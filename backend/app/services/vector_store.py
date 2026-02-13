from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .pdf_extract import chunk_text
from .rag import embedding_to_json, rank_chunks, text_to_embedding


def _backend() -> str:
    return os.getenv("VECTOR_BACKEND", "sqlite").strip().lower()


def _is_chroma_enabled() -> bool:
    return _backend() == "chroma"


def _get_chroma_collection():
    try:
        import chromadb
    except Exception:
        return None
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_dir = os.getenv("VECTOR_CHROMA_DIR") or os.path.join(base_dir, "data", "chroma")
    name = os.getenv("VECTOR_CHROMA_COLLECTION", "paper_chunks")
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        return client.get_or_create_collection(name=name)
    except Exception:
        return None


def index_paper_text(
    conn,
    paper_id: int,
    title: Optional[str],
    text: str,
    created_at: int,
    chunk_size: int = 900,
    overlap: int = 160,
) -> int:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))

    embeddings: List[List[float]] = []
    rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        vector = text_to_embedding(chunk["text"])
        embeddings.append(vector)
        rows.append(
            {
                "paper_id": paper_id,
                "chunk_index": chunk["index"],
                "chunk_text": chunk["text"],
                "chunk_embedding": embedding_to_json(vector),
                "created_at": created_at,
            }
        )

    for row in rows:
        conn.execute(
            """
            INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text, chunk_embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["paper_id"],
                row["chunk_index"],
                row["chunk_text"],
                row["chunk_embedding"],
                row["created_at"],
            ),
        )

    if _is_chroma_enabled():
        collection = _get_chroma_collection()
        if collection is not None:
            try:
                collection.delete(where={"paper_id": paper_id})
            except Exception:
                pass
            if rows:
                ids = [f"{paper_id}:{row['chunk_index']}" for row in rows]
                docs = [row["chunk_text"] for row in rows]
                metas = [
                    {
                        "paper_id": int(paper_id),
                        "chunk_index": int(row["chunk_index"]),
                        "title": title or "",
                    }
                    for row in rows
                ]
                try:
                    collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
                except Exception:
                    # Keep SQLite index as fallback, even if Chroma write fails.
                    pass

    return len(rows)


def search_paper_chunks(
    conn,
    query: str,
    paper_ids: Optional[List[int]] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    if _is_chroma_enabled():
        collection = _get_chroma_collection()
        if collection is not None:
            try:
                kwargs: Dict[str, Any] = {
                    "query_embeddings": [text_to_embedding(query)],
                    "n_results": max(1, min(top_k * 2, 20)),
                }
                if paper_ids:
                    if len(paper_ids) == 1:
                        kwargs["where"] = {"paper_id": int(paper_ids[0])}
                    else:
                        kwargs["where"] = {"paper_id": {"$in": [int(x) for x in paper_ids]}}
                result = collection.query(**kwargs)
                docs = (result.get("documents") or [[]])[0]
                metas = (result.get("metadatas") or [[]])[0]
                distances = (result.get("distances") or [[]])[0]
                items: List[Dict[str, Any]] = []
                for idx, doc in enumerate(docs):
                    meta = metas[idx] if idx < len(metas) else {}
                    distance = distances[idx] if idx < len(distances) else 1.0
                    score = max(0.0, 1.0 - float(distance))
                    items.append(
                        {
                            "paper_id": int(meta.get("paper_id", 0)),
                            "chunk_index": int(meta.get("chunk_index", idx)),
                            "chunk_text": doc,
                            "title": meta.get("title") or "",
                            "score": score,
                        }
                    )
                items.sort(key=lambda x: x["score"], reverse=True)
                return items[: max(1, top_k)]
            except Exception:
                pass

    if paper_ids:
        placeholders = ",".join(["?"] * len(paper_ids))
        rows = conn.execute(
            f"""
            SELECT
                c.paper_id,
                c.chunk_index,
                c.chunk_text,
                c.chunk_embedding,
                p.title
            FROM paper_chunks c
            JOIN papers p ON p.id = c.paper_id
            WHERE c.paper_id IN ({placeholders})
            """,
            paper_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                c.paper_id,
                c.chunk_index,
                c.chunk_text,
                c.chunk_embedding,
                p.title
            FROM paper_chunks c
            JOIN papers p ON p.id = c.paper_id
            ORDER BY c.created_at DESC
            LIMIT 3000
            """
        ).fetchall()
    chunks = [dict(r) for r in rows]
    return rank_chunks(query, chunks, top_k=top_k)
