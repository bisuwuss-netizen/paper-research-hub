from __future__ import annotations

import os
import re
import shutil
import threading
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import hashlib

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from .db import get_conn
from .repository import add_citation_edge, update_paper as repo_update_paper, upsert_paper
from .schemas import (
    Paper,
    PaperUpdate,
    Subfield,
    SubfieldCreate,
    SubfieldUpdate,
    PaperNote,
    PaperNoteUpdate,
    ReadingTask,
    ReadingTaskCreate,
    ReadingTaskUpdate,
    Experiment,
    ExperimentCreate,
    ExperimentUpdate,
)
from .services.crossref import lookup_doi
from .services.llm_client import (
    enrich_metadata,
    summarize_one_line,
    extract_references,
    chat_with_context,
    stream_chat_with_context,
    classify_citation_intent,
    extract_reported_metrics,
    extract_event_schema,
)
from .services.pdf_extract import (
    extract_metadata,
    extract_text,
    extract_references_text,
    extract_full_text,
    extract_markdown,
    extract_tables_text,
    extract_ee_metrics,
)
from .services.semantic_scholar import (
    extract_paper_from_edge,
    get_citations,
    get_references,
    search_match,
    to_paper_record,
)
from .services.zotero import (
    apply_paper_updates,
    extract_library_info,
    get_item,
    search_item_by_title,
    update_item,
)
from .services.recommendations import build_clusters, build_foundation_path, build_sota_list
from .services.ccf import classify_venue, learn_alias
from .services.sync_queue import (
    enqueue_all,
    enqueue_paper,
    fetch_due,
    mark_failure,
    mark_running,
    mark_success,
    queue_stats,
)
from .services.search_quality import (
    SearchDocument,
    classify_edge_confidence,
    detect_conflicts,
    detect_duplicate_groups,
    hybrid_search,
    title_similarity,
)
from .services.analytics import build_progress_snapshot, build_topic_evolution, to_period
from .services.rag import text_to_embedding, embedding_to_json
from .services.vector_store import index_paper_text, search_paper_chunks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage", "pdfs")

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="PaperTrail API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


_sync_worker_started = False
_last_cleanup_at = 0


def _cleanup_citations(conn) -> dict:
    before = conn.execute("SELECT COUNT(*) AS c FROM citations").fetchone()["c"]
    conn.execute(
        "DELETE FROM citations WHERE source_paper_id NOT IN (SELECT id FROM papers)"
    )
    conn.execute(
        "DELETE FROM citations WHERE target_paper_id NOT IN (SELECT id FROM papers)"
    )
    conn.execute("DELETE FROM citations WHERE source_paper_id = target_paper_id")
    conn.execute("DELETE FROM paper_chunks WHERE paper_id NOT IN (SELECT id FROM papers)")
    conn.execute("DELETE FROM ee_metrics WHERE paper_id NOT IN (SELECT id FROM papers)")
    conn.execute("DELETE FROM paper_schemas WHERE paper_id NOT IN (SELECT id FROM papers)")
    conn.execute("DELETE FROM note_links WHERE source_paper_id NOT IN (SELECT id FROM papers)")
    conn.execute("DELETE FROM note_links WHERE target_paper_id NOT IN (SELECT id FROM papers)")
    after = conn.execute("SELECT COUNT(*) AS c FROM citations").fetchone()["c"]
    return {"before": before, "after": after, "removed": max(0, before - after)}


def _maybe_cleanup(conn) -> None:
    global _last_cleanup_at
    if os.getenv("SYNC_CLEANUP_ENABLED", "true").lower() != "true":
        return
    interval = int(os.getenv("SYNC_CLEANUP_INTERVAL", "86400"))
    now = int(time.time())
    if now - _last_cleanup_at < interval:
        return
    _cleanup_citations(conn)
    _last_cleanup_at = now


def _sync_worker_loop():
    interval = int(os.getenv("SYNC_POLL_INTERVAL", "60"))
    batch_size = int(os.getenv("SYNC_BATCH_SIZE", "5"))
    expand = os.getenv("SYNC_EXPAND", "true").lower() == "true"
    while True:
        try:
            with get_conn() as conn:
                _maybe_cleanup(conn)
                due = fetch_due(conn, batch_size)
                for paper_id in due:
                    mark_running(conn, paper_id)
                    try:
                        row = conn.execute(
                            "SELECT * FROM papers WHERE id = ?", (paper_id,)
                        ).fetchone()
                        if not row:
                            mark_success(conn, paper_id)
                            continue
                        paper = dict(row)
                        if not paper.get("s2_paper_id"):
                            s2_record = _resolve_semantic_scholar(
                                paper.get("title"), paper.get("doi")
                            )
                            if s2_record:
                                paper = upsert_paper(conn, s2_record)
                        new_ids = _sync_citations(conn, paper, paper.get("s2_paper_id"))
                        if expand and new_ids:
                            for nid in new_ids:
                                enqueue_paper(conn, nid, delay_seconds=600)
                        mark_success(conn, paper_id)
                    except Exception as exc:
                        mark_failure(conn, paper_id, str(exc))
        except Exception:
            pass
        time.sleep(interval)


@app.on_event("startup")
def start_sync_worker():
    global _sync_worker_started
    with get_conn() as conn:
        _ensure_periodic_reports(conn)
        _sync_search_fts(conn)
    if _sync_worker_started:
        return
    if os.getenv("SYNC_ENABLED", "true").lower() != "true":
        return
    if os.getenv("SYNC_AUTO_ENQUEUE", "true").lower() == "true":
        with get_conn() as conn:
            enqueue_all(conn)
    thread = threading.Thread(target=_sync_worker_loop, daemon=True)
    thread.start()
    _sync_worker_started = True


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _now_ts() -> int:
    return int(time.time())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _fetch_note(conn, paper_id: int) -> Dict[str, Any]:
    row = conn.execute("SELECT * FROM paper_notes WHERE paper_id = ?", (paper_id,)).fetchone()
    if row:
        return dict(row)
    return {
        "paper_id": paper_id,
        "method": None,
        "datasets": None,
        "conclusions": None,
        "reproducibility": None,
        "risks": None,
        "notes": None,
        "created_at": None,
        "updated_at": None,
    }


def _upsert_note(conn, paper_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    existing = conn.execute("SELECT paper_id FROM paper_notes WHERE paper_id = ?", (paper_id,)).fetchone()
    now_ts = _now_ts()
    if existing:
        sets = []
        values: List[Any] = []
        for key, value in payload.items():
            sets.append(f"{key} = ?")
            values.append(value)
        sets.append("updated_at = ?")
        values.append(now_ts)
        values.append(paper_id)
        conn.execute(f"UPDATE paper_notes SET {', '.join(sets)} WHERE paper_id = ?", values)
    else:
        data = {
            "paper_id": paper_id,
            "method": payload.get("method"),
            "datasets": payload.get("datasets"),
            "conclusions": payload.get("conclusions"),
            "reproducibility": payload.get("reproducibility"),
            "risks": payload.get("risks"),
            "notes": payload.get("notes"),
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        conn.execute(
            """
            INSERT INTO paper_notes (
                paper_id, method, datasets, conclusions, reproducibility, risks, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["paper_id"],
                data["method"],
                data["datasets"],
                data["conclusions"],
                data["reproducibility"],
                data["risks"],
                data["notes"],
                data["created_at"],
                data["updated_at"],
            ),
        )
    return _fetch_note(conn, paper_id)


def _extract_note_link_tokens(text: str) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"\[\[([^\[\]]+)\]\]", text)
    tokens: List[str] = []
    seen = set()
    for raw in matches:
        token = raw.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token[:160])
    return tokens


def _resolve_note_link_target(conn, token: str) -> Optional[int]:
    if token.isdigit():
        row = conn.execute("SELECT id FROM papers WHERE id = ?", (int(token),)).fetchone()
        return int(row["id"]) if row else None
    row = conn.execute("SELECT id FROM papers WHERE LOWER(title) = LOWER(?)", (token,)).fetchone()
    if row:
        return int(row["id"])
    rows = conn.execute(
        "SELECT id, title FROM papers WHERE title LIKE ? LIMIT 50",
        (f"%{token}%",),
    ).fetchall()
    best_id = None
    best_score = 0.0
    for item in rows:
        score = title_similarity(token, item["title"])
        if score > best_score:
            best_score = score
            best_id = int(item["id"])
    if best_score >= 0.72:
        return best_id
    return None


def _refresh_note_links(conn, source_paper_id: int, note_text: str) -> int:
    conn.execute("DELETE FROM note_links WHERE source_paper_id = ?", (source_paper_id,))
    tokens = _extract_note_link_tokens(note_text)
    if not tokens:
        return 0
    now_ts = _now_ts()
    inserted = 0
    for token in tokens:
        target_id = _resolve_note_link_target(conn, token)
        if not target_id or target_id == source_paper_id:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO note_links (
                source_paper_id, target_paper_id, link_text, context, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_paper_id, target_id, token, None, now_ts, now_ts),
        )
        inserted += 1
    return inserted


def _fetch_backlinks(conn, paper_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            l.source_paper_id,
            l.target_paper_id,
            l.link_text,
            l.updated_at,
            p.title AS source_title,
            p.year AS source_year,
            p.sub_field AS source_sub_field
        FROM note_links l
        JOIN papers p ON p.id = l.source_paper_id
        WHERE l.target_paper_id = ?
        ORDER BY COALESCE(l.updated_at, 0) DESC
        LIMIT ?
        """,
        (paper_id, max(1, min(limit, 200))),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_paper_schema(conn, paper_id: int) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM paper_schemas WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if not row:
        return {
            "paper_id": paper_id,
            "event_types": [],
            "role_types": [],
            "schema_notes": None,
            "confidence": None,
            "source": None,
            "updated_at": None,
        }
    data = dict(row)
    return {
        "paper_id": paper_id,
        "event_types": _json_loads(data.get("event_types_json"), []),
        "role_types": _json_loads(data.get("role_types_json"), []),
        "schema_notes": data.get("schema_notes"),
        "confidence": data.get("confidence"),
        "source": data.get("source"),
        "updated_at": data.get("updated_at"),
    }


def _upsert_paper_schema(
    conn,
    paper_id: int,
    schema: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    now_ts = _now_ts()
    conn.execute(
        """
        INSERT INTO paper_schemas (
            paper_id, event_types_json, role_types_json, schema_notes, confidence, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            event_types_json = excluded.event_types_json,
            role_types_json = excluded.role_types_json,
            schema_notes = excluded.schema_notes,
            confidence = excluded.confidence,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        (
            paper_id,
            _json_dumps(schema.get("event_types") or []),
            _json_dumps(schema.get("role_types") or []),
            schema.get("schema_notes"),
            schema.get("confidence"),
            source,
            now_ts,
        ),
    )
    return _fetch_paper_schema(conn, paper_id)


def _extract_and_store_schema(
    conn,
    paper_id: int,
    full_text: str,
    source: str,
    force: bool = False,
) -> Dict[str, Any]:
    existing = _fetch_paper_schema(conn, paper_id)
    if not force and existing.get("event_types"):
        return existing
    schema = extract_event_schema(full_text, max_types=24)
    return _upsert_paper_schema(conn, paper_id, schema=schema, source=source)


def _sync_search_fts(conn) -> None:
    try:
        conn.execute("DELETE FROM paper_search_fts")
    except Exception:
        return
    rows = conn.execute(
        """
        SELECT
            p.id,
            p.title,
            p.abstract,
            p.proposed_method_name,
            p.dynamic_tags,
            p.open_sub_field,
            p.keywords,
            n.method,
            n.datasets,
            n.conclusions,
            n.reproducibility,
            n.risks,
            n.notes
        FROM papers p
        LEFT JOIN paper_notes n ON n.paper_id = p.id
        """
    ).fetchall()
    for row in rows:
        note_blob = " ".join(
            [
                row["method"] or "",
                row["datasets"] or "",
                row["conclusions"] or "",
                row["reproducibility"] or "",
                row["risks"] or "",
                row["notes"] or "",
                row["proposed_method_name"] or "",
                row["open_sub_field"] or "",
                row["keywords"] or "",
                " ".join(_normalize_open_tags(row["dynamic_tags"])),
            ]
        ).strip()
        conn.execute(
            "INSERT INTO paper_search_fts (paper_id, title, abstract, notes) VALUES (?, ?, ?, ?)",
            (row["id"], row["title"] or "", row["abstract"] or "", note_blob),
        )


def _collect_search_documents(conn) -> List[SearchDocument]:
    rows = conn.execute(
        """
        SELECT
            p.id,
            p.title,
            p.abstract,
            p.proposed_method_name,
            p.dynamic_tags,
            p.open_sub_field,
            p.keywords,
            n.method,
            n.datasets,
            n.conclusions,
            n.reproducibility,
            n.risks,
            n.notes
        FROM papers p
        LEFT JOIN paper_notes n ON n.paper_id = p.id
        """
    ).fetchall()
    docs: List[SearchDocument] = []
    for r in rows:
        notes = " ".join(
            [
                r["method"] or "",
                r["datasets"] or "",
                r["conclusions"] or "",
                r["reproducibility"] or "",
                r["risks"] or "",
                r["notes"] or "",
                r["proposed_method_name"] or "",
                r["open_sub_field"] or "",
                r["keywords"] or "",
                " ".join(_normalize_open_tags(r["dynamic_tags"])),
            ]
        ).strip()
        docs.append(
            SearchDocument(
                doc_id=r["id"],
                title=r["title"] or "",
                abstract=r["abstract"] or "",
                notes=notes,
            )
        )
    return docs


def _index_paper_chunks(conn, paper_id: int, full_text: str) -> int:
    now_ts = _now_ts()
    row = conn.execute("SELECT title FROM papers WHERE id = ?", (paper_id,)).fetchone()
    title = row["title"] if row else None
    count = index_paper_text(conn, paper_id, title=title, text=full_text, created_at=now_ts)
    if not full_text:
        return 0
    # Keep a coarse paper-level embedding in SQLite for lightweight ranking/fallback.
    vector = text_to_embedding(full_text[:6000])
    if vector:
        repo_update_paper(conn, paper_id, {"embedding": embedding_to_json(vector)})
    return count


def _retrieve_rag_chunks(
    conn,
    query: str,
    paper_ids: Optional[List[int]] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    return search_paper_chunks(conn, query=query, paper_ids=paper_ids, top_k=top_k)


def _upsert_ee_metrics(
    conn,
    paper_id: int,
    metrics: List[Dict[str, Any]],
    source: str,
) -> int:
    if not metrics:
        return 0
    now_ts = _now_ts()
    inserted = 0
    # Keep the latest extraction snapshot by source.
    conn.execute(
        "DELETE FROM ee_metrics WHERE paper_id = ? AND source = ?",
        (paper_id, source),
    )
    for item in metrics:
        dataset = item.get("dataset_name")
        if not dataset:
            continue
        conn.execute(
            """
            INSERT INTO ee_metrics (
                paper_id, dataset_name, precision, recall, f1, trigger_f1, argument_f1, source, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                dataset,
                item.get("precision"),
                item.get("recall"),
                item.get("f1"),
                item.get("trigger_f1"),
                item.get("argument_f1"),
                source,
                item.get("confidence"),
                now_ts,
            ),
        )
        inserted += 1
    return inserted


def _sync_auto_experiments_from_metrics(conn, paper_id: int, metrics: List[Dict[str, Any]]) -> int:
    if not metrics:
        conn.execute(
            "DELETE FROM experiments WHERE paper_id = ? AND name LIKE 'AUTO_METRIC:%'",
            (paper_id,),
        )
        return 0
    conn.execute(
        "DELETE FROM experiments WHERE paper_id = ? AND name LIKE 'AUTO_METRIC:%'",
        (paper_id,),
    )
    now_ts = _now_ts()
    inserted = 0
    metric_fields = [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("trigger_f1", "Trigger F1"),
        ("argument_f1", "Argument F1"),
    ]
    for item in metrics:
        dataset = item.get("dataset_name") or "Unknown"
        for key, label in metric_fields:
            value = item.get(key)
            if value is None:
                continue
            try:
                val = float(value)
            except Exception:
                continue
            conn.execute(
                """
                INSERT INTO experiments (
                    paper_id, name, model, params_json, metrics_json, result_summary, artifact_path,
                    dataset_name, trigger_f1, argument_f1, precision, recall, f1,
                    dataset, split, metric_name, metric_value, is_sota,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    f"AUTO_METRIC:{dataset}:{label}",
                    "auto_extractor",
                    None,
                    _json_dumps(item),
                    f"Auto extracted {label} on {dataset}",
                    None,
                    dataset,
                    item.get("trigger_f1"),
                    item.get("argument_f1"),
                    item.get("precision"),
                    item.get("recall"),
                    item.get("f1"),
                    dataset,
                    "test",
                    label,
                    val,
                    1 if key in {"f1", "trigger_f1", "argument_f1"} and val >= 80 else 0,
                    now_ts,
                    now_ts,
                ),
            )
            inserted += 1
    return inserted


def _normalize_open_tags(tags: Any) -> List[str]:
    if isinstance(tags, str):
        parsed = _json_loads(tags, None)
        if isinstance(parsed, list):
            tags = parsed
        else:
            tags = [x.strip() for x in tags.split(",") if x.strip()]
    if not isinstance(tags, list):
        return []
    seen = set()
    out = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean = tag.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean[:60])
    return out[:12]


def _discover_open_subfields(conn, limit: int = 100) -> List[str]:
    rows = conn.execute(
        """
        SELECT dynamic_tags, keywords, sub_field, open_sub_field
        FROM papers
        WHERE dynamic_tags IS NOT NULL OR keywords IS NOT NULL OR open_sub_field IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    freq: Dict[str, int] = {}
    existing = {r["name"].lower() for r in conn.execute("SELECT name FROM subfields").fetchall()}
    for row in rows:
        tags = _normalize_open_tags(row["dynamic_tags"])
        if row.get("open_sub_field"):
            tags = [row["open_sub_field"], *tags]
        if not tags and row["keywords"]:
            tags = [x.strip() for x in str(row["keywords"]).split(",") if x.strip()]
        for tag in tags:
            low = tag.lower()
            if len(low) < 4:
                continue
            if low in existing:
                continue
            if low in {"event extraction", "nlp"}:
                continue
            freq[tag] = freq.get(tag, 0) + 1
    candidates = [tag for tag, c in sorted(freq.items(), key=lambda x: x[1], reverse=True) if c >= 2]
    return candidates[:20]


def _cluster_open_tags(candidates: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for tag in candidates:
        key = re.sub(r"[^a-z0-9]+", " ", tag.lower()).strip()
        if not key:
            continue
        tokens = [t for t in key.split() if len(t) >= 3]
        if not tokens:
            continue
        cluster_key = " ".join(tokens[:2])
        row = groups.setdefault(cluster_key, {"label": tag, "members": [], "count": 0})
        row["members"].append(tag)
        row["count"] += 1
        if len(tag) < len(row["label"]):
            row["label"] = tag
    ranked = sorted(groups.values(), key=lambda x: (x["count"], len(x["members"])), reverse=True)
    return ranked[:20]


def _build_sota_board(conn, dataset: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    params: List[Any] = []
    where = ""
    if dataset:
        where = "WHERE LOWER(m.dataset_name) = LOWER(?)"
        params.append(dataset)
    query = f"""
        SELECT
            m.*,
            p.title,
            p.year,
            p.sub_field,
            p.ccf_level,
            p.doi,
            p.url
        FROM ee_metrics m
        JOIN papers p ON p.id = m.paper_id
        {where}
        ORDER BY COALESCE(m.f1, m.argument_f1, m.trigger_f1, 0) DESC, COALESCE(p.year, 0) DESC
        LIMIT ?
    """
    params.append(max(1, min(limit, 200)))
    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    datasets = sorted({(r.get("dataset_name") or "").strip() for r in rows if r.get("dataset_name")})
    return {"dataset": dataset, "items": rows, "datasets": datasets, "count": len(rows)}


def _build_metric_leaderboard(
    conn,
    dataset: Optional[str] = None,
    metric: str = "f1",
    limit: int = 50,
) -> Dict[str, Any]:
    metric = (metric or "f1").strip().lower()
    metric_column = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "trigger_f1": "trigger_f1",
        "argument_f1": "argument_f1",
    }.get(metric, "f1")
    params: List[Any] = []
    where = f"WHERE m.{metric_column} IS NOT NULL"
    if dataset:
        where += " AND LOWER(m.dataset_name) = LOWER(?)"
        params.append(dataset)
    query = f"""
        SELECT
            m.paper_id,
            m.dataset_name,
            m.{metric_column} AS metric_value,
            p.title,
            p.year,
            p.sub_field,
            p.ccf_level
        FROM ee_metrics m
        JOIN papers p ON p.id = m.paper_id
        {where}
        ORDER BY m.{metric_column} DESC, COALESCE(p.year, 0) DESC
        LIMIT ?
    """
    params.append(max(1, min(limit, 300)))
    top_rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    trend_query = f"""
        SELECT
            COALESCE(p.year, 0) AS year,
            MAX(m.{metric_column}) AS best_value,
            AVG(m.{metric_column}) AS avg_value,
            COUNT(*) AS sample_count
        FROM ee_metrics m
        JOIN papers p ON p.id = m.paper_id
        {where}
        GROUP BY COALESCE(p.year, 0)
        ORDER BY year ASC
    """
    trend_rows = [dict(r) for r in conn.execute(trend_query, params[:-1] if params else []).fetchall()]
    trend_rows = [row for row in trend_rows if row.get("year")]
    return {
        "dataset": dataset,
        "metric": metric_column,
        "top_items": top_rows,
        "trend": trend_rows,
    }


def _get_shortest_path(
    conn,
    source_id: int,
    target_id: int,
    direction: str = "any",
) -> Dict[str, Any]:
    rows = conn.execute("SELECT source_paper_id, target_paper_id FROM citations").fetchall()
    edges = [(int(r["source_paper_id"]), int(r["target_paper_id"])) for r in rows]
    if not edges:
        return {"nodes": [], "edges": [], "distance": None}
    try:
        import networkx as nx
    except Exception:
        return {"nodes": [], "edges": [], "distance": None}

    if direction == "out":
        graph = nx.DiGraph()
    elif direction == "in":
        graph = nx.DiGraph()
        edges = [(b, a) for a, b in edges]
    else:
        graph = nx.Graph()
    graph.add_edges_from(edges)
    try:
        node_path = nx.shortest_path(graph, source=source_id, target=target_id)
    except Exception:
        return {"nodes": [], "edges": [], "distance": None}
    if len(node_path) < 2:
        return {"nodes": node_path, "edges": [], "distance": 0}
    edge_path = [(int(node_path[i]), int(node_path[i + 1])) for i in range(len(node_path) - 1)]
    return {"nodes": node_path, "edges": edge_path, "distance": len(edge_path)}


def _spaced_next_interval(current_days: int) -> int:
    if current_days <= 0:
        return 1
    if current_days < 3:
        return 3
    if current_days < 7:
        return 7
    if current_days < 14:
        return 14
    if current_days < 30:
        return 30
    return 60


def _insert_zotero_log(
    conn,
    paper_id: Optional[int],
    direction: str,
    action: str,
    status: str,
    conflict_strategy: Optional[str],
    details: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO zotero_sync_logs (
            paper_id, direction, action, status, conflict_strategy, details, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            direction,
            action,
            status,
            conflict_strategy,
            _json_dumps(details),
            _now_ts(),
        ),
    )


def _apply_zotero_mapping(paper: Dict[str, Any], mapping: Dict[str, str], item: Dict[str, Any]) -> Dict[str, Any]:
    data = item.get("data") or {}
    updates: Dict[str, Any] = {}
    for local_field, zotero_field in mapping.items():
        if local_field in {"authors"}:
            if zotero_field == "creators":
                creators = data.get("creators") or []
                names: List[str] = []
                for c in creators:
                    first = (c.get("firstName") or "").strip()
                    last = (c.get("lastName") or "").strip()
                    full = " ".join([first, last]).strip()
                    if full:
                        names.append(full)
                if names:
                    updates["authors"] = ", ".join(names)
            continue
        raw = data.get(zotero_field)
        if raw in (None, ""):
            continue
        if local_field == "year":
            year_match = re.search(r"\b(19|20)\d{2}\b", str(raw))
            if year_match:
                updates["year"] = int(year_match.group(0))
            continue
        if local_field == "summary_one" and zotero_field == "extra":
            updates["summary_one"] = str(raw)[:300]
            continue
        updates[local_field] = raw
    return updates


def _default_zotero_mapping(conn) -> Dict[str, str]:
    row = conn.execute(
        "SELECT mapping_json FROM zotero_mapping_templates ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    mapping = _json_loads(row["mapping_json"], {})
    return mapping if isinstance(mapping, dict) else {}


def _generate_report_payload(conn, period_type: str, period_start: str, period_end: str) -> Dict[str, Any]:
    start_dt = datetime.fromisoformat(period_start)
    end_dt = datetime.fromisoformat(period_end)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.replace(hour=23, minute=59, second=59).timestamp())

    papers = [dict(r) for r in conn.execute("SELECT * FROM papers").fetchall()]
    progress = build_progress_snapshot(papers)
    new_papers = [
        {"id": p["id"], "title": p.get("title"), "year": p.get("year")}
        for p in papers
        if (p.get("created_at") or 0) >= start_ts and (p.get("created_at") or 0) <= end_ts
    ][:20]

    read_updates = conn.execute(
        """
        SELECT id, title, updated_at FROM papers
        WHERE read_status = 1 AND updated_at BETWEEN ? AND ?
        ORDER BY updated_at DESC
        LIMIT 20
        """,
        (start_ts, end_ts),
    ).fetchall()
    read_items = [{"id": r["id"], "title": r["title"], "updated_at": r["updated_at"]} for r in read_updates]

    task_overdue = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM reading_tasks
        WHERE status != 'done'
          AND due_date IS NOT NULL
          AND due_date < ?
        """,
        (period_end,),
    ).fetchone()

    rec_rows = conn.execute("SELECT * FROM papers ORDER BY citation_count DESC LIMIT 80").fetchall()
    rec_papers = [dict(r) for r in rec_rows]
    node_ids = {p["id"] for p in rec_papers}
    edge_rows = conn.execute("SELECT source_paper_id, target_paper_id FROM citations").fetchall()
    edges = [
        (r["source_paper_id"], r["target_paper_id"])
        for r in edge_rows
        if r["source_paper_id"] in node_ids and r["target_paper_id"] in node_ids
    ]
    foundation = build_foundation_path(rec_papers, edges)[:8]

    next_suggestions = []
    if task_overdue and task_overdue["c"] > 0:
        next_suggestions.append(f"处理 {task_overdue['c']} 个已逾期阅读任务")
    if progress["unread"] > 0:
        next_suggestions.append("优先完成高引用未读论文")
    if not next_suggestions:
        next_suggestions.append("保持当前阅读节奏，扩展新子方向候选论文")

    return {
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        "progress": progress,
        "new_papers": new_papers,
        "read_updates": read_items,
        "task_overdue": task_overdue["c"] if task_overdue else 0,
        "key_path_changes": foundation,
        "next_suggestions": next_suggestions,
    }


def _ensure_periodic_reports(conn) -> None:
    for period in ("weekly", "monthly"):
        start, end = to_period(period)
        period_start = start.isoformat()
        period_end = end.isoformat()
        exists = conn.execute(
            """
            SELECT id FROM reports
            WHERE period_type = ? AND period_start = ? AND period_end = ?
            """,
            (period, period_start, period_end),
        ).fetchone()
        if exists:
            continue
        payload = _generate_report_payload(conn, period, period_start, period_end)
        conn.execute(
            """
            INSERT INTO reports (period_type, period_start, period_end, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (period, period_start, period_end, _json_dumps(payload), _now_ts()),
        )


class MergeRequest(BaseModel):
    source_paper_id: int
    target_paper_id: int


class ZoteroSyncRequest(BaseModel):
    direction: str = "both"  # both | pull | push
    conflict_strategy: str = "prefer_local"  # prefer_local | prefer_zotero | manual
    limit: int = 20


class ZoteroMappingTemplateRequest(BaseModel):
    name: str = "default"
    mapping: Dict[str, str]


class ChatQueryRequest(BaseModel):
    paper_ids: List[int] = []
    query: str
    top_k: int = 6
    language: str = "zh"


class CitationIntentRequest(BaseModel):
    edge_ids: List[str] | None = None
    limit: int = 50


class ComparePapersRequest(BaseModel):
    paper_ids: List[int]


@app.get("/api/papers", response_model=List[Paper])
def list_papers():
    with get_conn() as conn:
        _ensure_periodic_reports(conn)
        rows = conn.execute("SELECT * FROM papers ORDER BY id DESC").fetchall()
        return [Paper(**dict(row)) for row in rows]


@app.get("/api/search")
def search_papers(q: str, top_k: int = 20):
    query = (q or "").strip()
    if not query:
        return {"query": q, "results": []}
    with get_conn() as conn:
        docs = _collect_search_documents(conn)
        ranked = hybrid_search(query, docs, top_k=max(1, min(top_k, 100)))
        chunk_hits = _retrieve_rag_chunks(conn, query, top_k=max(6, min(top_k * 2, 30)))
        chunk_score_map: Dict[int, float] = {}
        best_chunk_map: Dict[int, Dict[str, Any]] = {}
        for hit in chunk_hits:
            pid = int(hit.get("paper_id"))
            score = float(hit.get("score") or 0.0)
            if score > chunk_score_map.get(pid, 0.0):
                chunk_score_map[pid] = score
                best_chunk_map[pid] = hit

        # Merge doc-level and chunk-level scores.
        rank_map: Dict[int, Dict[str, float]] = {}
        for item in ranked:
            pid = int(item["doc_id"])
            rank_map[pid] = {
                "score": float(item.get("score", 0.0)),
                "bm25_score": float(item.get("bm25_score", 0.0)),
                "semantic_score": float(item.get("semantic_score", 0.0)),
                "chunk_score": chunk_score_map.get(pid, 0.0),
            }
        for pid, chunk_score in chunk_score_map.items():
            if pid not in rank_map:
                rank_map[pid] = {
                    "score": 0.0,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                    "chunk_score": chunk_score,
                }
            rank_map[pid]["score"] = max(rank_map[pid]["score"], rank_map[pid]["score"] * 0.72 + chunk_score * 0.48)

        rank_items = sorted(rank_map.items(), key=lambda x: x[1]["score"], reverse=True)[: max(1, min(top_k, 100))]
        if not rank_items:
            fallback_rows = conn.execute(
                """
                SELECT * FROM papers
                ORDER BY COALESCE(citation_count, 0) DESC, COALESCE(updated_at, 0) DESC
                LIMIT ?
                """,
                (max(1, min(top_k, 20)),),
            ).fetchall()
            fallback = []
            for row in fallback_rows:
                paper = dict(row)
                fallback.append(
                    {
                        "paper": paper,
                        "score": 0.0,
                        "bm25_score": 0.0,
                        "semantic_score": 0.0,
                        "snippet": (paper.get("abstract") or paper.get("title") or "")[:220],
                    }
                )
            return {"query": query, "results": fallback, "fallback": True}
        ids = [pid for pid, _ in rank_items]
        doc_scores = {pid: score for pid, score in rank_items}
        placeholders = ",".join(["?"] * len(ids))
        rows = conn.execute(
            f"SELECT * FROM papers WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        notes_map = {
            r["paper_id"]: dict(r)
            for r in conn.execute(
                f"SELECT * FROM paper_notes WHERE paper_id IN ({placeholders})",
                ids,
            ).fetchall()
        }
        paper_map = {r["id"]: dict(r) for r in rows}
        results = []
        for pid in ids:
            paper = paper_map.get(pid)
            if not paper:
                continue
            score = doc_scores[pid]
            notes = notes_map.get(pid) or {}
            note_text = notes.get("notes") or notes.get("conclusions") or notes.get("method") or ""
            best_chunk = best_chunk_map.get(pid)
            snippet_source = (
                (best_chunk or {}).get("chunk_text")
                or paper.get("abstract")
                or note_text
                or paper.get("title")
                or ""
            )
            snippet = snippet_source[:220]
            results.append(
                {
                    "paper": paper,
                    "score": score.get("score", 0.0),
                    "bm25_score": score.get("bm25_score", 0.0),
                    "semantic_score": max(score.get("semantic_score", 0.0), score.get("chunk_score", 0.0)),
                    "chunk_score": score.get("chunk_score", 0.0),
                    "snippet": snippet,
                    "chunk_index": best_chunk.get("chunk_index") if best_chunk else None,
                }
            )
        return {"query": query, "results": results, "fallback": False}


@app.get("/api/papers/{paper_id}/notes", response_model=PaperNote)
def get_paper_notes(paper_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        note = _fetch_note(conn, paper_id)
        return PaperNote(**note)


@app.put("/api/papers/{paper_id}/notes", response_model=PaperNote)
def save_paper_notes(paper_id: int, payload: PaperNoteUpdate):
    updates = payload.model_dump()
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        note = _upsert_note(conn, paper_id, updates)
        _refresh_note_links(conn, paper_id, note.get("notes") or "")
        _sync_search_fts(conn)
        return PaperNote(**note)


@app.get("/api/papers/{paper_id}/backlinks")
def get_paper_backlinks(paper_id: int, limit: int = 100):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        items = _fetch_backlinks(conn, paper_id, limit=limit)
        return {"paper_id": paper_id, "count": len(items), "items": items}


@app.get("/api/papers/{paper_id}/schema")
def get_paper_schema(paper_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        return _fetch_paper_schema(conn, paper_id)


@app.post("/api/papers/{paper_id}/schema/extract")
def extract_paper_schema(paper_id: int, force: bool = False):
    with get_conn() as conn:
        row = conn.execute("SELECT id, file_path FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        paper = dict(row)
        file_path = paper.get("file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="Paper has no uploaded PDF")
        text = extract_markdown(file_path) or extract_full_text(file_path)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract PDF text")
        schema = _extract_and_store_schema(conn, paper_id, text, source="manual_extract", force=force)
        return schema


@app.get("/api/schemas/search")
def search_schemas(keyword: str, limit: int = 50):
    key = (keyword or "").strip().lower()
    if not key:
        return {"keyword": keyword, "count": 0, "items": []}
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                s.paper_id,
                s.event_types_json,
                s.role_types_json,
                s.schema_notes,
                p.title,
                p.year,
                p.sub_field
            FROM paper_schemas s
            JOIN papers p ON p.id = s.paper_id
            ORDER BY COALESCE(s.updated_at, 0) DESC
            LIMIT ?
            """,
            (max(1, min(limit * 4, 500)),),
        ).fetchall()
        items: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            event_types = _json_loads(data.get("event_types_json"), [])
            role_types = _json_loads(data.get("role_types_json"), [])
            flat = " ".join(
                [
                    str(data.get("title") or ""),
                    str(data.get("schema_notes") or ""),
                    " ".join([str(r) for r in role_types]),
                    " ".join([str(e.get("name") or "") for e in event_types if isinstance(e, dict)]),
                ]
            ).lower()
            if key not in flat:
                continue
            items.append(
                {
                    "paper_id": data["paper_id"],
                    "title": data.get("title"),
                    "year": data.get("year"),
                    "sub_field": data.get("sub_field"),
                    "event_types": event_types,
                    "role_types": role_types,
                }
            )
            if len(items) >= limit:
                break
        return {"keyword": keyword, "count": len(items), "items": items}


@app.get("/api/tasks", response_model=List[ReadingTask])
def list_tasks(paper_id: Optional[int] = None, status: Optional[str] = None):
    with get_conn() as conn:
        query = "SELECT * FROM reading_tasks WHERE 1=1"
        values: List[Any] = []
        if paper_id:
            query += " AND paper_id = ?"
            values.append(paper_id)
        if status:
            query += " AND status = ?"
            values.append(status)
        query += " ORDER BY COALESCE(due_date, '9999-12-31') ASC, priority DESC, id DESC"
        rows = conn.execute(query, values).fetchall()
        return [ReadingTask(**dict(r)) for r in rows]


@app.post("/api/tasks", response_model=ReadingTask)
def create_task(payload: ReadingTaskCreate):
    with get_conn() as conn:
        paper = conn.execute("SELECT id FROM papers WHERE id = ?", (payload.paper_id,)).fetchone()
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        now_ts = _now_ts()
        conn.execute(
            """
            INSERT INTO reading_tasks (
                paper_id, title, status, priority, due_date, next_review_at, interval_days, created_at, updated_at
            ) VALUES (?, ?, 'todo', ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.paper_id,
                payload.title,
                payload.priority,
                payload.due_date,
                payload.next_review_at,
                payload.interval_days,
                now_ts,
                now_ts,
            ),
        )
        row = conn.execute("SELECT * FROM reading_tasks ORDER BY id DESC LIMIT 1").fetchone()
        return ReadingTask(**dict(row))


@app.patch("/api/tasks/{task_id}", response_model=ReadingTask)
def update_task(task_id: int, payload: ReadingTaskUpdate):
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reading_tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        fields = []
        values = []
        for key, value in updates.items():
            fields.append(f"{key} = ?")
            values.append(value)
        fields.append("updated_at = ?")
        values.append(_now_ts())
        values.append(task_id)
        conn.execute(f"UPDATE reading_tasks SET {', '.join(fields)} WHERE id = ?", values)
        updated = conn.execute("SELECT * FROM reading_tasks WHERE id = ?", (task_id,)).fetchone()
        return ReadingTask(**dict(updated))


@app.post("/api/tasks/{task_id}/complete-review", response_model=ReadingTask)
def complete_review(task_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reading_tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        task = dict(row)
        now_ts = _now_ts()
        current_interval = task.get("interval_days") or 1
        next_interval = _spaced_next_interval(int(current_interval))
        next_review = now_ts + next_interval * 86400
        conn.execute(
            """
            UPDATE reading_tasks
            SET status = 'done',
                last_review_at = ?,
                next_review_at = ?,
                interval_days = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (now_ts, next_review, next_interval, now_ts, task_id),
        )
        updated = conn.execute("SELECT * FROM reading_tasks WHERE id = ?", (task_id,)).fetchone()
        return ReadingTask(**dict(updated))


@app.get("/api/experiments", response_model=List[Experiment])
def list_experiments(paper_id: Optional[int] = None):
    with get_conn() as conn:
        if paper_id:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE paper_id = ? ORDER BY id DESC",
                (paper_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
        return [Experiment(**dict(r)) for r in rows]


@app.post("/api/experiments", response_model=Experiment)
def create_experiment(payload: ExperimentCreate):
    with get_conn() as conn:
        paper = conn.execute("SELECT id FROM papers WHERE id = ?", (payload.paper_id,)).fetchone()
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        now_ts = _now_ts()
        conn.execute(
            """
            INSERT INTO experiments (
                paper_id, name, model, params_json, metrics_json, result_summary, artifact_path,
                dataset_name, trigger_f1, argument_f1, precision, recall, f1,
                dataset, split, metric_name, metric_value, is_sota,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.paper_id,
                payload.name,
                payload.model,
                payload.params_json,
                payload.metrics_json,
                payload.result_summary,
                payload.artifact_path,
                payload.dataset_name,
                payload.trigger_f1,
                payload.argument_f1,
                payload.precision,
                payload.recall,
                payload.f1,
                payload.dataset,
                payload.split,
                payload.metric_name,
                payload.metric_value,
                payload.is_sota,
                now_ts,
                now_ts,
            ),
        )
        row = conn.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT 1").fetchone()
        return Experiment(**dict(row))


@app.patch("/api/experiments/{experiment_id}", response_model=Experiment)
def update_experiment(experiment_id: int, payload: ExperimentUpdate):
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Experiment not found")
        fields = []
        values = []
        for key, value in updates.items():
            fields.append(f"{key} = ?")
            values.append(value)
        fields.append("updated_at = ?")
        values.append(_now_ts())
        values.append(experiment_id)
        conn.execute(f"UPDATE experiments SET {', '.join(fields)} WHERE id = ?", values)
        updated = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        return Experiment(**dict(updated))


@app.delete("/api/experiments/{experiment_id}")
def delete_experiment(experiment_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        return {"status": "deleted"}


@app.get("/api/quality/duplicates")
def quality_duplicates(title_threshold: float = 0.9):
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT id, title, doi, year FROM papers").fetchall()]
        groups = detect_duplicate_groups(rows, title_threshold=title_threshold)
        return {"groups": groups, "count": len(groups)}


@app.get("/api/quality/conflicts")
def quality_conflicts():
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT id, title, doi, year FROM papers").fetchall()]
        conflicts = detect_conflicts(rows)
        return {"conflicts": conflicts, "count": len(conflicts)}


@app.post("/api/quality/merge")
def quality_merge(payload: MergeRequest):
    source_id = payload.source_paper_id
    target_id = payload.target_paper_id
    if source_id == target_id:
        raise HTTPException(status_code=400, detail="source and target must differ")
    with get_conn() as conn:
        source_row = conn.execute("SELECT * FROM papers WHERE id = ?", (source_id,)).fetchone()
        target_row = conn.execute("SELECT * FROM papers WHERE id = ?", (target_id,)).fetchone()
        if not source_row or not target_row:
            raise HTTPException(status_code=404, detail="Paper not found")
        source = dict(source_row)
        target = dict(target_row)

        out_edges = conn.execute(
            """
            SELECT target_paper_id, confidence, edge_source, evidence, intent, intent_confidence
            FROM citations
            WHERE source_paper_id = ?
            """,
            (source_id,),
        ).fetchall()
        in_edges = conn.execute(
            """
            SELECT source_paper_id, confidence, edge_source, evidence, intent, intent_confidence
            FROM citations
            WHERE target_paper_id = ?
            """,
            (source_id,),
        ).fetchall()
        for e in out_edges:
            if e["target_paper_id"] == target_id:
                continue
            add_citation_edge(
                conn,
                target_id,
                e["target_paper_id"],
                confidence=e["confidence"] or 0.8,
                edge_source=e["edge_source"],
                evidence=e["evidence"],
                intent=e["intent"],
                intent_confidence=e["intent_confidence"],
            )
        for e in in_edges:
            if e["source_paper_id"] == target_id:
                continue
            add_citation_edge(
                conn,
                e["source_paper_id"],
                target_id,
                confidence=e["confidence"] or 0.8,
                edge_source=e["edge_source"],
                evidence=e["evidence"],
                intent=e["intent"],
                intent_confidence=e["intent_confidence"],
            )

        conn.execute(
            "DELETE FROM citations WHERE source_paper_id = ? OR target_paper_id = ?",
            (source_id, source_id),
        )

        # Merge textual side tables.
        source_note = _fetch_note(conn, source_id)
        target_note = _fetch_note(conn, target_id)
        merged_note = {}
        for field in ["method", "datasets", "conclusions", "reproducibility", "risks", "notes"]:
            merged_note[field] = target_note.get(field) or source_note.get(field)
        _upsert_note(conn, target_id, merged_note)
        conn.execute("DELETE FROM paper_notes WHERE paper_id = ?", (source_id,))

        conn.execute("UPDATE reading_tasks SET paper_id = ? WHERE paper_id = ?", (target_id, source_id))
        conn.execute("UPDATE experiments SET paper_id = ? WHERE paper_id = ?", (target_id, source_id))
        conn.execute("UPDATE ee_metrics SET paper_id = ? WHERE paper_id = ?", (target_id, source_id))
        conn.execute("UPDATE paper_chunks SET paper_id = ? WHERE paper_id = ?", (target_id, source_id))
        conn.execute("UPDATE note_links SET source_paper_id = ? WHERE source_paper_id = ?", (target_id, source_id))
        conn.execute("UPDATE note_links SET target_paper_id = ? WHERE target_paper_id = ?", (target_id, source_id))
        conn.execute(
            """
            DELETE FROM note_links
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM note_links
                GROUP BY source_paper_id, target_paper_id, link_text
            )
            """
        )
        source_schema = conn.execute(
            "SELECT * FROM paper_schemas WHERE paper_id = ?",
            (source_id,),
        ).fetchone()
        target_schema = conn.execute(
            "SELECT * FROM paper_schemas WHERE paper_id = ?",
            (target_id,),
        ).fetchone()
        if source_schema and not target_schema:
            schema = dict(source_schema)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_schemas (
                    paper_id, event_types_json, role_types_json, schema_notes, confidence, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    schema.get("event_types_json"),
                    schema.get("role_types_json"),
                    schema.get("schema_notes"),
                    schema.get("confidence"),
                    schema.get("source"),
                    schema.get("updated_at"),
                ),
            )
        conn.execute("DELETE FROM paper_schemas WHERE paper_id = ?", (source_id,))
        conn.execute("UPDATE zotero_sync_logs SET paper_id = ? WHERE paper_id = ?", (target_id, source_id))
        conn.execute("DELETE FROM sync_queue WHERE paper_id = ?", (source_id,))

        updates: Dict[str, Any] = {}
        for field in [
            "title",
            "authors",
            "year",
            "journal_conf",
            "ccf_level",
            "source_type",
            "sub_field",
            "file_path",
            "doi",
            "abstract",
            "s2_paper_id",
            "citation_count",
            "reference_count",
            "url",
            "venue",
            "summary_one",
            "file_hash",
            "proposed_method_name",
            "dynamic_tags",
            "open_sub_field",
        ]:
            if not target.get(field) and source.get(field):
                updates[field] = source.get(field)
        if updates:
            repo_update_paper(conn, target_id, updates)

        conn.execute("DELETE FROM papers WHERE id = ?", (source_id,))
        _sync_search_fts(conn)
        return {"status": "merged", "source": source_id, "target": target_id}


@app.post("/api/quality/auto-merge")
def quality_auto_merge(limit: int = 20, title_threshold: float = 0.96):
    merged = 0
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM papers").fetchall()]
        groups = detect_duplicate_groups(rows, title_threshold=title_threshold)
        for group in groups[:limit]:
            ids = group.get("paper_ids") or []
            if len(ids) < 2:
                continue
            paper_rows = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM papers WHERE id IN ({','.join(['?'] * len(ids))})",
                    ids,
                ).fetchall()
            ]
            if len(paper_rows) < 2:
                continue
            paper_rows.sort(
                key=lambda p: (
                    1 if p.get("file_path") else 0,
                    p.get("citation_count") or 0,
                    p.get("updated_at") or 0,
                ),
                reverse=True,
            )
            keeper = paper_rows[0]["id"]
            for item in paper_rows[1:]:
                source = item["id"]
                out_edges = conn.execute(
                    """
                    SELECT target_paper_id, confidence, edge_source, evidence, intent, intent_confidence
                    FROM citations
                    WHERE source_paper_id = ?
                    """,
                    (source,),
                ).fetchall()
                in_edges = conn.execute(
                    """
                    SELECT source_paper_id, confidence, edge_source, evidence, intent, intent_confidence
                    FROM citations
                    WHERE target_paper_id = ?
                    """,
                    (source,),
                ).fetchall()
                for e in out_edges:
                    if e["target_paper_id"] == keeper:
                        continue
                    add_citation_edge(
                        conn,
                        keeper,
                        e["target_paper_id"],
                        confidence=e["confidence"] or 0.8,
                        edge_source=e["edge_source"],
                        evidence=e["evidence"],
                        intent=e["intent"],
                        intent_confidence=e["intent_confidence"],
                    )
                for e in in_edges:
                    if e["source_paper_id"] == keeper:
                        continue
                    add_citation_edge(
                        conn,
                        e["source_paper_id"],
                        keeper,
                        confidence=e["confidence"] or 0.8,
                        edge_source=e["edge_source"],
                        evidence=e["evidence"],
                        intent=e["intent"],
                        intent_confidence=e["intent_confidence"],
                    )
                conn.execute(
                    "DELETE FROM citations WHERE source_paper_id = ? OR target_paper_id = ?",
                    (source, source),
                )
                conn.execute("UPDATE reading_tasks SET paper_id = ? WHERE paper_id = ?", (keeper, source))
                conn.execute("UPDATE experiments SET paper_id = ? WHERE paper_id = ?", (keeper, source))
                conn.execute("UPDATE ee_metrics SET paper_id = ? WHERE paper_id = ?", (keeper, source))
                conn.execute("UPDATE paper_chunks SET paper_id = ? WHERE paper_id = ?", (keeper, source))
                conn.execute("UPDATE note_links SET source_paper_id = ? WHERE source_paper_id = ?", (keeper, source))
                conn.execute("UPDATE note_links SET target_paper_id = ? WHERE target_paper_id = ?", (keeper, source))
                conn.execute(
                    """
                    DELETE FROM note_links
                    WHERE id NOT IN (
                        SELECT MIN(id)
                        FROM note_links
                        GROUP BY source_paper_id, target_paper_id, link_text
                    )
                    """
                )
                source_schema = conn.execute(
                    "SELECT * FROM paper_schemas WHERE paper_id = ?",
                    (source,),
                ).fetchone()
                keeper_schema = conn.execute(
                    "SELECT * FROM paper_schemas WHERE paper_id = ?",
                    (keeper,),
                ).fetchone()
                if source_schema and not keeper_schema:
                    schema = dict(source_schema)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO paper_schemas (
                            paper_id, event_types_json, role_types_json, schema_notes, confidence, source, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            keeper,
                            schema.get("event_types_json"),
                            schema.get("role_types_json"),
                            schema.get("schema_notes"),
                            schema.get("confidence"),
                            schema.get("source"),
                            schema.get("updated_at"),
                        ),
                    )
                conn.execute("DELETE FROM paper_schemas WHERE paper_id = ?", (source,))
                conn.execute("UPDATE zotero_sync_logs SET paper_id = ? WHERE paper_id = ?", (keeper, source))
                conn.execute("DELETE FROM paper_notes WHERE paper_id = ?", (source,))
                conn.execute("DELETE FROM sync_queue WHERE paper_id = ?", (source,))
                conn.execute("DELETE FROM papers WHERE id = ?", (source,))
                merged += 1
        if merged:
            _sync_search_fts(conn)
    return {"merged": merged}


@app.get("/api/analytics/topic-evolution")
def topic_evolution():
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM papers").fetchall()]
        return build_topic_evolution(rows)


@app.get("/api/analytics/topic-river")
def topic_river():
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM papers").fetchall()]
        evolution = build_topic_evolution(rows)
        return {
            "river": evolution.get("river", []),
            "years": evolution.get("years", []),
            "sub_fields": evolution.get("sub_fields", []),
            "bursts": evolution.get("bursts", []),
        }


@app.get("/api/subfields/discover-open-tags")
def discover_open_tags(limit: int = 200, add_to_subfields: bool = False):
    with get_conn() as conn:
        candidates = _discover_open_subfields(conn, limit=max(20, min(limit, 1000)))
        clusters = _cluster_open_tags(candidates)
        added = 0
        if add_to_subfields:
            existing = {r["name"].lower() for r in conn.execute("SELECT name FROM subfields").fetchall()}
            for row in clusters:
                label = row.get("label")
                if not isinstance(label, str):
                    continue
                name = label.strip()
                if not name or name.lower() in existing:
                    continue
                conn.execute(
                    "INSERT INTO subfields (name, description, active, created_at) VALUES (?, ?, 1, ?)",
                    (name, "Auto-discovered from open tags", _now_ts()),
                )
                existing.add(name.lower())
                added += 1
        return {"candidates": candidates, "clusters": clusters, "added": added}


@app.post("/api/chat/papers")
def chat_with_papers(payload: ChatQueryRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    with get_conn() as conn:
        paper_ids = [int(pid) for pid in payload.paper_ids if pid]
        contexts = _retrieve_rag_chunks(
            conn,
            query,
            paper_ids=paper_ids or None,
            top_k=max(2, min(payload.top_k, 12)),
        )
        result = chat_with_context(query, contexts, language=payload.language)
        return {
            "query": query,
            "paper_ids": paper_ids,
            "answer": result.get("answer") or "",
            "sources": result.get("sources") or [],
            "context_count": len(contexts),
        }


@app.post("/api/chat/papers/stream")
def chat_with_papers_stream(payload: ChatQueryRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    with get_conn() as conn:
        paper_ids = [int(pid) for pid in payload.paper_ids if pid]
        contexts = _retrieve_rag_chunks(
            conn,
            query,
            paper_ids=paper_ids or None,
            top_k=max(2, min(payload.top_k, 12)),
        )

    def event_stream():
        init_payload = {
            "type": "meta",
            "query": query,
            "paper_ids": paper_ids,
            "context_count": len(contexts),
        }
        yield f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
        for event in stream_chat_with_context(query, contexts, language=payload.language):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/metrics/sota")
def metrics_sota(dataset: Optional[str] = None, limit: int = 50):
    with get_conn() as conn:
        return _build_sota_board(conn, dataset=dataset, limit=limit)


@app.get("/api/metrics/leaderboard")
def metrics_leaderboard(
    dataset: Optional[str] = None,
    metric: str = "f1",
    limit: int = 50,
):
    with get_conn() as conn:
        return _build_metric_leaderboard(conn, dataset=dataset, metric=metric, limit=limit)


@app.post("/api/papers/compare")
def compare_papers(payload: ComparePapersRequest):
    paper_ids = [int(pid) for pid in payload.paper_ids if pid]
    if not paper_ids:
        raise HTTPException(status_code=400, detail="paper_ids is required")
    with get_conn() as conn:
        placeholders = ",".join(["?"] * len(paper_ids))
        papers = [dict(r) for r in conn.execute(
            f"SELECT * FROM papers WHERE id IN ({placeholders})",
            paper_ids,
        ).fetchall()]
        metrics_rows = [dict(r) for r in conn.execute(
            f"""
            SELECT
                m.paper_id,
                m.dataset_name,
                m.precision,
                m.recall,
                m.f1,
                m.trigger_f1,
                m.argument_f1,
                m.confidence
            FROM ee_metrics m
            WHERE m.paper_id IN ({placeholders})
            ORDER BY COALESCE(m.f1, m.argument_f1, m.trigger_f1, 0) DESC
            """,
            paper_ids,
        ).fetchall()]
        metrics_map: Dict[int, List[Dict[str, Any]]] = {}
        for row in metrics_rows:
            metrics_map.setdefault(int(row["paper_id"]), []).append(row)
        result = []
        for paper in papers:
            result.append(
                {
                    "paper": paper,
                    "metrics": metrics_map.get(int(paper["id"]), []),
                    "tags": _normalize_open_tags(paper.get("dynamic_tags")),
                }
            )
        return {"items": result, "count": len(result)}


@app.post("/api/citations/intent/classify")
def classify_citation_intents(payload: CitationIntentRequest):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                c.source_paper_id,
                c.target_paper_id,
                c.intent,
                c.intent_confidence,
                sp.title AS source_title,
                sp.abstract AS source_abstract,
                tp.title AS target_title
            FROM citations c
            JOIN papers sp ON sp.id = c.source_paper_id
            JOIN papers tp ON tp.id = c.target_paper_id
            ORDER BY COALESCE(c.last_verified_at, 0) DESC
            """
        ).fetchall()

        edge_filter = set(payload.edge_ids or [])
        pending: List[Dict[str, Any]] = []
        for row in rows:
            edge_id = f"{row['source_paper_id']}->{row['target_paper_id']}"
            if edge_filter and edge_id not in edge_filter:
                continue
            if not edge_filter and row["intent"] and row["intent_confidence"] is not None:
                continue
            pending.append(dict(row))
            if len(pending) >= max(1, min(payload.limit, 200)):
                break

        updated = 0
        results: List[Dict[str, Any]] = []
        for item in pending:
            out = classify_citation_intent(
                item.get("source_title"),
                item.get("source_abstract"),
                item.get("target_title"),
            )
            intent = out.get("intent") or "mention"
            confidence = float(out.get("confidence") or 0.0)
            conn.execute(
                """
                UPDATE citations
                SET intent = ?, intent_confidence = ?, last_verified_at = ?
                WHERE source_paper_id = ? AND target_paper_id = ?
                """,
                (
                    intent,
                    confidence,
                    _now_ts(),
                    item["source_paper_id"],
                    item["target_paper_id"],
                ),
            )
            updated += 1
            results.append(
                {
                    "id": f"{item['source_paper_id']}->{item['target_paper_id']}",
                    "source": item["source_paper_id"],
                    "target": item["target_paper_id"],
                    "intent": intent,
                    "intent_confidence": confidence,
                }
            )
        return {"updated": updated, "items": results}


@app.get("/api/graph/shortest-path")
def graph_shortest_path(source_id: int, target_id: int, direction: str = "any"):
    direction = direction.lower().strip()
    if direction not in {"any", "out", "in"}:
        raise HTTPException(status_code=400, detail="direction must be any/out/in")
    with get_conn() as conn:
        path = _get_shortest_path(conn, source_id=source_id, target_id=target_id, direction=direction)
        if not path["nodes"]:
            return {"source_id": source_id, "target_id": target_id, "path": path, "papers": []}
        placeholders = ",".join(["?"] * len(path["nodes"]))
        rows = conn.execute(
            f"SELECT id, title, year, sub_field FROM papers WHERE id IN ({placeholders})",
            path["nodes"],
        ).fetchall()
        paper_map = {r["id"]: dict(r) for r in rows}
        papers = [paper_map.get(pid, {"id": pid}) for pid in path["nodes"]]
        return {"source_id": source_id, "target_id": target_id, "path": path, "papers": papers}


@app.post("/api/reports/generate")
def generate_report(period: str = "weekly"):
    period = (period or "weekly").lower()
    if period not in {"weekly", "monthly"}:
        raise HTTPException(status_code=400, detail="period must be weekly or monthly")
    with get_conn() as conn:
        start, end = to_period(period)
        payload = _generate_report_payload(conn, period, start.isoformat(), end.isoformat())
        conn.execute(
            """
            INSERT INTO reports (period_type, period_start, period_end, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (period, start.isoformat(), end.isoformat(), _json_dumps(payload), _now_ts()),
        )
        row = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT 1").fetchone()
        out = dict(row)
        out["payload"] = _json_loads(out.get("payload_json"), {})
        return out


@app.get("/api/reports")
def list_reports(period: Optional[str] = None, limit: int = 20):
    with get_conn() as conn:
        _ensure_periodic_reports(conn)
        if period:
            rows = conn.execute(
                "SELECT * FROM reports WHERE period_type = ? ORDER BY created_at DESC LIMIT ?",
                (period, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["payload"] = _json_loads(data.get("payload_json"), {})
            result.append(data)
        return result


@app.get("/api/zotero/mapping-template")
def zotero_mapping_template(name: str = "default"):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM zotero_mapping_templates WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Template not found")
        data = dict(row)
        data["mapping"] = _json_loads(data.get("mapping_json"), {})
        return data


@app.put("/api/zotero/mapping-template")
def upsert_zotero_mapping_template(payload: ZoteroMappingTemplateRequest):
    name = payload.name
    mapping = payload.mapping or {}
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM zotero_mapping_templates WHERE name = ?",
            (name,),
        ).fetchone()
        now_ts = _now_ts()
        if row:
            conn.execute(
                "UPDATE zotero_mapping_templates SET mapping_json = ?, updated_at = ? WHERE name = ?",
                (_json_dumps(mapping), now_ts, name),
            )
        else:
            conn.execute(
                """
                INSERT INTO zotero_mapping_templates (name, mapping_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, _json_dumps(mapping), now_ts, now_ts),
            )
        return {"name": name, "mapping": mapping}


@app.get("/api/zotero/sync-logs")
def zotero_sync_logs(limit: int = 100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM zotero_sync_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        logs = []
        for row in rows:
            data = dict(row)
            data["details"] = _json_loads(data.get("details"), {})
            logs.append(data)
        return logs


@app.post("/api/zotero/sync-incremental")
def zotero_sync_incremental(payload: ZoteroSyncRequest):
    direction = payload.direction.lower()
    strategy = payload.conflict_strategy.lower()
    if direction not in {"both", "pull", "push"}:
        raise HTTPException(status_code=400, detail="direction must be both/pull/push")
    if strategy not in {"prefer_local", "prefer_zotero", "manual"}:
        raise HTTPException(status_code=400, detail="invalid conflict strategy")
    synced = 0
    conflicts = 0
    with get_conn() as conn:
        mapping = _default_zotero_mapping(conn)
        rows = conn.execute(
            """
            SELECT * FROM papers
            WHERE zotero_item_key IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(1, min(payload.limit, 200)),),
        ).fetchall()
        for row in rows:
            paper = dict(row)
            item = get_item(paper.get("zotero_item_key"))
            if not item:
                _insert_zotero_log(
                    conn,
                    paper["id"],
                    direction="pull",
                    action="fetch_item",
                    status="failed",
                    conflict_strategy=strategy,
                    details={"reason": "item_not_found"},
                )
                continue

            pulled = _apply_zotero_mapping(paper, mapping, item)
            field_conflicts = {}
            for field, remote_value in pulled.items():
                local_value = paper.get(field)
                if local_value and remote_value and str(local_value).strip() != str(remote_value).strip():
                    field_conflicts[field] = {"local": local_value, "zotero": remote_value}
            if field_conflicts and strategy == "manual":
                conflicts += 1
                _insert_zotero_log(
                    conn,
                    paper["id"],
                    direction=direction,
                    action="sync",
                    status="conflict",
                    conflict_strategy=strategy,
                    details={"fields": field_conflicts},
                )
                continue

            updates_local: Dict[str, Any] = {}
            if direction in {"both", "pull"}:
                for field, remote_value in pulled.items():
                    if remote_value in (None, ""):
                        continue
                    local_value = paper.get(field)
                    if field in field_conflicts and strategy == "prefer_local":
                        continue
                    if field in field_conflicts and strategy == "prefer_zotero":
                        updates_local[field] = remote_value
                    elif not local_value:
                        updates_local[field] = remote_value
            if updates_local:
                repo_update_paper(conn, paper["id"], updates_local)

            if direction in {"both", "push"}:
                updated_item = apply_paper_updates(item, paper)
                if not update_item(paper.get("zotero_item_key"), updated_item):
                    _insert_zotero_log(
                        conn,
                        paper["id"],
                        direction="push",
                        action="update_item",
                        status="failed",
                        conflict_strategy=strategy,
                        details={"reason": "update_failed"},
                    )
                    continue

            _insert_zotero_log(
                conn,
                paper["id"],
                direction=direction,
                action="sync",
                status="ok",
                conflict_strategy=strategy,
                details={"updated_local_fields": list(updates_local.keys()), "conflicts": list(field_conflicts.keys())},
            )
            synced += 1
    return {"synced": synced, "conflicts": conflicts}


@app.get("/api/subfields", response_model=List[Subfield])
def list_subfields(active_only: bool = True):
    with get_conn() as conn:
        if active_only:
            rows = conn.execute("SELECT * FROM subfields WHERE active = 1 ORDER BY name").fetchall()
        else:
            rows = conn.execute("SELECT * FROM subfields ORDER BY name").fetchall()
        return [Subfield(**dict(r)) for r in rows]


@app.post("/api/subfields", response_model=Subfield)
def create_subfield(payload: SubfieldCreate):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO subfields (name, description, active, created_at) VALUES (?, ?, ?, ?)",
            (payload.name, payload.description, payload.active, int(time.time())),
        )
        row = conn.execute(
            "SELECT * FROM subfields WHERE name = ?", (payload.name,)
        ).fetchone()
        return Subfield(**dict(row))


@app.patch("/api/subfields/{subfield_id}", response_model=Subfield)
def update_subfield(subfield_id: int, payload: SubfieldUpdate):
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    with get_conn() as conn:
        fields = []
        values = []
        for k, v in updates.items():
            if k == "id":
                continue
            fields.append(f"{k} = ?")
            values.append(v)
        values.append(subfield_id)
        conn.execute(f"UPDATE subfields SET {', '.join(fields)} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM subfields WHERE id = ?", (subfield_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Subfield not found")
        return Subfield(**dict(row))


@app.delete("/api/subfields/{subfield_id}")
def delete_subfield(subfield_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM subfields WHERE id = ?", (subfield_id,))
        return {"status": "deleted"}


@app.get("/api/papers/{paper_id}", response_model=Paper)
def get_paper(paper_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        return Paper(**dict(row))


@app.patch("/api/papers/{paper_id}", response_model=Paper)
def update_paper(paper_id: int, payload: PaperUpdate):
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        repo_update_paper(conn, paper_id, updates)
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        paper = dict(row)
        if "sub_field" in updates and updates.get("sub_field"):
            text = f"{paper.get('title') or ''}\n{paper.get('abstract') or ''}".strip()
            if text:
                _record_subfield_feedback(conn, paper_id, text, updates.get("sub_field"))
        if updates.get("zotero_item_key") or (
            paper.get("zotero_item_key") and not paper.get("zotero_item_id")
        ):
            _sync_zotero_meta(conn, paper)
            row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        _sync_search_fts(conn)
        return Paper(**dict(row))


@app.post("/api/ccf/refresh")
def refresh_ccf():
    with get_conn() as conn:
        rows = conn.execute("SELECT id, journal_conf, venue FROM papers").fetchall()
        updated = 0
        for r in rows:
            venue = r["journal_conf"] or r["venue"]
            level = classify_venue(venue)
            if level:
                repo_update_paper(conn, r["id"], {"ccf_level": level})
                updated += 1
        return {"updated": updated}


@app.post("/api/ccf/learn-aliases")
def learn_ccf_aliases(limit: int = 20):
    learned = 0
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE zotero_item_key IS NOT NULL AND ccf_level IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
        for row in rows:
            paper = dict(row)
            item = get_item(paper.get("zotero_item_key"))
            if not item:
                continue
            data = item.get("data") or {}
            zotero_venue = data.get("publicationTitle") or data.get("proceedingsTitle") or data.get("conferenceName")
            if zotero_venue and learn_alias(zotero_venue, paper.get("ccf_level")):
                learned += 1
    return {"learned": learned}


@app.post("/api/papers/{paper_id}/zotero-match", response_model=Paper)
def match_zotero(paper_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        paper = dict(row)
        item = search_item_by_title(paper.get("title") or "")
        if not item:
            raise HTTPException(status_code=404, detail="No Zotero match found")
        item_key = item.get("key")
        if not item_key:
            raise HTTPException(status_code=404, detail="No Zotero key in response")
        data = item.get("data") or {}
        zotero_venue = data.get("publicationTitle") or data.get("proceedingsTitle") or data.get("conferenceName")
        if zotero_venue and paper.get("ccf_level"):
            learn_alias(zotero_venue, paper.get("ccf_level"))
        library, item_id = extract_library_info(item)
        updates = {
            "zotero_item_key": item_key,
            "zotero_library": library,
            "zotero_item_id": item_id,
        }
        repo_update_paper(conn, paper_id, updates)
        _insert_zotero_log(
            conn,
            paper_id,
            direction="pull",
            action="match_by_title",
            status="ok",
            conflict_strategy=None,
            details={"item_key": item_key},
        )
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return Paper(**dict(row))


@app.post("/api/zotero/match-all")
def zotero_match_all(limit: int = 20):
    matched = 0
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE zotero_item_key IS NULL LIMIT ?", (limit,)
        ).fetchall()
        for row in rows:
            paper = dict(row)
            item = search_item_by_title(paper.get("title") or "")
            if not item or not item.get("key"):
                continue
            data = item.get("data") or {}
            zotero_venue = data.get("publicationTitle") or data.get("proceedingsTitle") or data.get("conferenceName")
            if zotero_venue and paper.get("ccf_level"):
                learn_alias(zotero_venue, paper.get("ccf_level"))
            library, item_id = extract_library_info(item)
            repo_update_paper(
                conn,
                paper["id"],
                {
                    "zotero_item_key": item.get("key"),
                    "zotero_library": library,
                    "zotero_item_id": item_id,
                },
            )
            matched += 1
    return {"matched": matched}


@app.post("/api/zotero/sync-ids")
def zotero_sync_ids(limit: int = 50):
    synced = 0
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM papers
            WHERE zotero_item_key IS NOT NULL
              AND (zotero_item_id IS NULL OR zotero_library IS NULL)
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        for row in rows:
            paper = dict(row)
            if _sync_zotero_meta(conn, paper):
                synced += 1
    return {"synced": synced}


@app.post("/api/zotero/push/{paper_id}")
def zotero_push(paper_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        paper = dict(row)
        if not paper.get("zotero_item_key"):
            raise HTTPException(status_code=400, detail="No Zotero item key")
        item = get_item(paper.get("zotero_item_key"))
        if not item:
            raise HTTPException(status_code=404, detail="Zotero item not found")
        data = item.get("data") or {}
        zotero_venue = data.get("publicationTitle") or data.get("proceedingsTitle") or data.get("conferenceName")
        if zotero_venue and paper.get("ccf_level"):
            learn_alias(zotero_venue, paper.get("ccf_level"))
        updated = apply_paper_updates(item, paper)
        if not update_item(paper.get("zotero_item_key"), updated):
            _insert_zotero_log(
                conn,
                paper_id,
                direction="push",
                action="update_item",
                status="failed",
                conflict_strategy=None,
                details={"reason": "update_failed"},
            )
            raise HTTPException(status_code=500, detail="Failed to update Zotero item")
        _insert_zotero_log(
            conn,
            paper_id,
            direction="push",
            action="update_item",
            status="ok",
            conflict_strategy=None,
            details={"item_key": paper.get("zotero_item_key")},
        )
        return {"status": "updated"}


@app.post("/api/zotero/push-all")
def zotero_push_all(limit: int = 20):
    updated = 0
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE zotero_item_key IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
        for row in rows:
            paper = dict(row)
            item = get_item(paper.get("zotero_item_key"))
            if not item:
                continue
            updated_item = apply_paper_updates(item, paper)
            if update_item(paper.get("zotero_item_key"), updated_item):
                updated += 1
                _insert_zotero_log(
                    conn,
                    paper["id"],
                    direction="push",
                    action="batch_push",
                    status="ok",
                    conflict_strategy=None,
                    details={"item_key": paper.get("zotero_item_key")},
                )
    return {"updated": updated}


def _filter_papers(
    papers,
    sub_field: Optional[str],
    year_from: Optional[int],
    year_to: Optional[int],
    uploaded_only: bool,
):
    def matches(p):
        if uploaded_only and not p.get("file_path"):
            return False
        if sub_field and p.get("sub_field") != sub_field:
            return False
        year = p.get("year")
        if year_from and (not year or year < year_from):
            return False
        if year_to and (not year or year > year_to):
            return False
        return True

    return [p for p in papers if matches(p)]


@app.get("/api/graph")
def get_graph(
    sub_field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    size_by: str = "citations",
    edge_recent_years: Optional[int] = None,
    min_indegree: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    sort_by: str = "citation_count",
    uploaded_only: bool = False,
    edge_min_confidence: float = 0.0,
):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM papers").fetchall()
        papers = [dict(r) for r in rows]

        nodes_raw = _filter_papers(papers, sub_field, year_from, year_to, uploaded_only)
        node_ids = {p["id"] for p in nodes_raw}

        edge_rows = conn.execute(
            """
            SELECT
                source_paper_id,
                target_paper_id,
                confidence,
                edge_source,
                intent,
                intent_confidence
            FROM citations
            """
        ).fetchall()
        edges = []
        current_year = datetime.now().year
        for r in edge_rows:
            source = r["source_paper_id"]
            target = r["target_paper_id"]
            if source in node_ids and target in node_ids:
                conf = r["confidence"] if r["confidence"] is not None else 1.0
                if conf < edge_min_confidence:
                    continue
                edges.append(
                    {
                        "id": f"{source}->{target}",
                        "source": source,
                        "target": target,
                        "confidence": conf,
                        "edge_source": r["edge_source"],
                        "intent": r["intent"],
                        "intent_confidence": r["intent_confidence"],
                    }
                )

        if edge_recent_years:
            threshold = current_year - edge_recent_years
            year_map = {p["id"]: p.get("year") for p in nodes_raw}
            edges = [
                e
                for e in edges
                if (year_map.get(e["source"]) or 0) >= threshold
            ]

        pagerank_full: Dict[int, float] = {}
        if sort_by == "pagerank":
            try:
                import networkx as nx

                G = nx.DiGraph()
                for p in nodes_raw:
                    G.add_node(p["id"])
                for e in edges:
                    G.add_edge(e["source"], e["target"])
                if G.number_of_nodes() > 0:
                    pagerank_full = nx.pagerank(G)
                nodes_raw.sort(key=lambda p: pagerank_full.get(p["id"], 0.0), reverse=True)
            except Exception:
                nodes_raw.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
        elif sort_by == "year":
            nodes_raw.sort(key=lambda p: p.get("year") or 0, reverse=True)
        else:
            nodes_raw.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)

        if min_indegree:
            indegree: Dict[int, int] = {}
            for e in edges:
                indegree[e["target"]] = indegree.get(e["target"], 0) + 1
            node_ids = {nid for nid in node_ids if indegree.get(nid, 0) >= min_indegree}
            nodes_raw = [p for p in nodes_raw if p["id"] in node_ids]
            edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

        total_nodes = len(nodes_raw)
        if limit is not None:
            nodes_raw = nodes_raw[offset : offset + limit]
            node_ids = {p["id"] for p in nodes_raw}
            edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

        def color_for(p):
            if not p.get("file_path") or p.get("read_status") == 0:
                return "#94a3b8"  # gray
            if p.get("ccf_level") in {"A", "B"}:
                return "#ef4444"  # red
            if p.get("source_type") == "CNKI":
                return "#22c55e"  # green
            return "#3b82f6"  # blue

        pagerank_map: Dict[int, float] = pagerank_full if size_by == "pagerank" else {}
        if size_by == "pagerank":
            try:
                import networkx as nx

                G = nx.DiGraph()
                for p in nodes_raw:
                    G.add_node(p["id"])
                for e in edges:
                    G.add_edge(e["source"], e["target"])
                if G.number_of_nodes() > 0:
                    pagerank_map = nx.pagerank(G)
            except Exception:
                pagerank_map = {}

        def size_for(p):
            if size_by == "pagerank":
                score = pagerank_map.get(p["id"], 0.0)
                return 20 + min(50, score * 800)
            count = p.get("citation_count") or 0
            return 20 + min(40, (count ** 0.5) * 4)

        nodes = []
        task_stats = {
            r["paper_id"]: {"open": r["open_count"] or 0, "overdue": r["overdue_count"] or 0}
            for r in conn.execute(
                """
                SELECT
                    paper_id,
                    SUM(CASE WHEN status != 'done' THEN 1 ELSE 0 END) AS open_count,
                    SUM(
                        CASE
                            WHEN status != 'done'
                             AND due_date IS NOT NULL
                             AND due_date < DATE('now')
                            THEN 1 ELSE 0
                        END
                    ) AS overdue_count
                FROM reading_tasks
                GROUP BY paper_id
                """
            ).fetchall()
        }
        exp_stats = {
            r["paper_id"]: r["c"] or 0
            for r in conn.execute(
                "SELECT paper_id, COUNT(*) AS c FROM experiments GROUP BY paper_id"
            ).fetchall()
        }
        for p in nodes_raw:
            task = task_stats.get(p["id"], {"open": 0, "overdue": 0})
            nodes.append(
                {
                    "id": p["id"],
                    "label": p.get("title") or f"Paper {p['id']}",
                    "size": size_for(p),
                    "color": color_for(p),
                    "year": p.get("year"),
                    "authors": p.get("authors"),
                    "abstract": p.get("abstract"),
                    "sub_field": p.get("sub_field"),
                    "read_status": p.get("read_status"),
                    "ccf_level": p.get("ccf_level"),
                    "source_type": p.get("source_type"),
                    "citation_count": p.get("citation_count"),
                    "reference_count": p.get("reference_count"),
                    "pagerank": pagerank_map.get(p["id"]),
                    "citation_velocity": p.get("citation_velocity"),
                    "doi": p.get("doi"),
                    "url": p.get("url"),
                    "zotero_item_key": p.get("zotero_item_key"),
                    "zotero_library": p.get("zotero_library"),
                    "zotero_item_id": p.get("zotero_item_id"),
                    "summary_one": p.get("summary_one"),
                    "proposed_method_name": p.get("proposed_method_name"),
                    "dynamic_tags": _normalize_open_tags(p.get("dynamic_tags")),
                    "open_sub_field": p.get("open_sub_field"),
                    "open_tasks": task["open"],
                    "overdue_tasks": task["overdue"],
                    "experiment_count": exp_stats.get(p["id"], 0),
                }
            )

        sub_fields = sorted({p.get("sub_field") for p in papers if p.get("sub_field")})
        years = [p.get("year") for p in papers if p.get("year")]
        year_counts: Dict[int, int] = {}
        for p in nodes_raw:
            year = p.get("year")
            if not year:
                continue
            year_counts[year] = year_counts.get(year, 0) + 1
        all_year_counts: Dict[int, int] = {}
        for p in papers:
            year = p.get("year")
            if not year:
                continue
            all_year_counts[year] = all_year_counts.get(year, 0) + 1
        meta = {
            "sub_fields": sub_fields,
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "year_counts": [
                {"year": y, "count": year_counts[y]} for y in sorted(year_counts.keys())
            ],
            "year_min_all": min(years) if years else None,
            "year_max_all": max(years) if years else None,
            "year_counts_all": [
                {"year": y, "count": all_year_counts[y]} for y in sorted(all_year_counts.keys())
            ],
            "total_nodes": total_nodes,
            "limit": limit,
            "offset": offset,
            "size_by": size_by,
        }

        return {"nodes": nodes, "edges": edges, "meta": meta}


@app.get("/api/graph/neighbors")
def graph_neighbors(node_id: int, depth: int = 1, limit: int = 200):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM papers").fetchall()
        papers = {r["id"]: dict(r) for r in rows}

        edge_rows = conn.execute(
            """
            SELECT
                source_paper_id,
                target_paper_id,
                confidence,
                edge_source,
                intent,
                intent_confidence
            FROM citations
            """
        ).fetchall()
        adjacency: Dict[int, List[int]] = {}
        reverse: Dict[int, List[int]] = {}
        for r in edge_rows:
            adjacency.setdefault(r["source_paper_id"], []).append(r["target_paper_id"])
            reverse.setdefault(r["target_paper_id"], []).append(r["source_paper_id"])

        seen = {node_id}
        frontier = {node_id}
        for _ in range(max(1, depth)):
            next_frontier = set()
            for nid in frontier:
                next_frontier.update(adjacency.get(nid, []))
                next_frontier.update(reverse.get(nid, []))
            next_frontier -= seen
            seen.update(next_frontier)
            frontier = next_frontier
            if len(seen) >= limit:
                break

        node_ids = list(seen)[:limit]
        nodes_raw = [papers[nid] for nid in node_ids if nid in papers]
        node_set = {p["id"] for p in nodes_raw}
        edges = [
            {
                "id": f"{r['source_paper_id']}->{r['target_paper_id']}",
                "source": r["source_paper_id"],
                "target": r["target_paper_id"],
                "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
                "edge_source": r["edge_source"],
                "intent": r["intent"],
                "intent_confidence": r["intent_confidence"],
            }
            for r in edge_rows
            if r["source_paper_id"] in node_set and r["target_paper_id"] in node_set
        ]

        def color_for(p):
            if not p.get("file_path") or p.get("read_status") == 0:
                return "#94a3b8"
            if p.get("ccf_level") in {"A", "B"}:
                return "#ef4444"
            if p.get("source_type") == "CNKI":
                return "#22c55e"
            return "#3b82f6"

        def size_for(p):
            count = p.get("citation_count") or 0
            return 20 + min(40, (count ** 0.5) * 4)

        nodes = []
        task_stats = {
            r["paper_id"]: {"open": r["open_count"] or 0, "overdue": r["overdue_count"] or 0}
            for r in conn.execute(
                """
                SELECT
                    paper_id,
                    SUM(CASE WHEN status != 'done' THEN 1 ELSE 0 END) AS open_count,
                    SUM(
                        CASE
                            WHEN status != 'done'
                             AND due_date IS NOT NULL
                             AND due_date < DATE('now')
                            THEN 1 ELSE 0
                        END
                    ) AS overdue_count
                FROM reading_tasks
                GROUP BY paper_id
                """
            ).fetchall()
        }
        exp_stats = {
            r["paper_id"]: r["c"] or 0
            for r in conn.execute(
                "SELECT paper_id, COUNT(*) AS c FROM experiments GROUP BY paper_id"
            ).fetchall()
        }
        for p in nodes_raw:
            task = task_stats.get(p["id"], {"open": 0, "overdue": 0})
            nodes.append(
                {
                    "id": p["id"],
                    "label": p.get("title") or f"Paper {p['id']}",
                    "size": size_for(p),
                    "color": color_for(p),
                    "year": p.get("year"),
                    "authors": p.get("authors"),
                    "abstract": p.get("abstract"),
                    "sub_field": p.get("sub_field"),
                    "read_status": p.get("read_status"),
                    "ccf_level": p.get("ccf_level"),
                    "source_type": p.get("source_type"),
                    "citation_count": p.get("citation_count"),
                    "reference_count": p.get("reference_count"),
                    "doi": p.get("doi"),
                    "url": p.get("url"),
                    "summary_one": p.get("summary_one"),
                    "proposed_method_name": p.get("proposed_method_name"),
                    "dynamic_tags": _normalize_open_tags(p.get("dynamic_tags")),
                    "open_sub_field": p.get("open_sub_field"),
                    "open_tasks": task["open"],
                    "overdue_tasks": task["overdue"],
                    "experiment_count": exp_stats.get(p["id"], 0),
                }
            )

        return {"nodes": nodes, "edges": edges}


@app.get("/api/recommendations")
def get_recommendations(
    strategy: str,
    sub_field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM papers").fetchall()
        papers = [dict(r) for r in rows]
        filtered = _filter_papers(papers, sub_field, year_from, year_to, False)
        node_ids = {p["id"] for p in filtered}
        edge_rows = conn.execute("SELECT source_paper_id, target_paper_id FROM citations").fetchall()
        edges = [
            (r["source_paper_id"], r["target_paper_id"])
            for r in edge_rows
            if r["source_paper_id"] in node_ids and r["target_paper_id"] in node_ids
        ]

        strategy = (strategy or "").lower()
        if strategy == "foundation":
            path = build_foundation_path(filtered, edges)
            return {"strategy": "foundation", "highlight_nodes": path}
        if strategy == "sota":
            nodes = build_sota_list(filtered)
            return {"strategy": "sota", "highlight_nodes": nodes}
        if strategy == "cluster":
            clusters = build_clusters(filtered)
            return {"strategy": "cluster", "clusters": clusters}

        raise HTTPException(status_code=400, detail="Unknown strategy")


@app.get("/api/recommendations/path")
def export_recommendation_path(
    strategy: str,
    sub_field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM papers").fetchall()
        papers = [dict(r) for r in rows]
        filtered = _filter_papers(papers, sub_field, year_from, year_to, False)
        id_map = {p["id"]: p for p in filtered}
        node_ids = set(id_map.keys())
        edge_rows = conn.execute("SELECT source_paper_id, target_paper_id FROM citations").fetchall()
        edges = [
            (r["source_paper_id"], r["target_paper_id"])
            for r in edge_rows
            if r["source_paper_id"] in node_ids and r["target_paper_id"] in node_ids
        ]

        strategy = (strategy or "").lower()
        if strategy == "foundation":
            path_ids = build_foundation_path(filtered, edges)
        elif strategy == "sota":
            path_ids = build_sota_list(filtered)
        else:
            raise HTTPException(status_code=400, detail="Strategy must be foundation or sota")

        path = []
        for pid in path_ids:
            p = id_map.get(pid)
            if not p:
                continue
            path.append(
                {
                    "id": p["id"],
                    "title": p.get("title"),
                    "authors": p.get("authors"),
                    "year": p.get("year"),
                    "venue": p.get("venue") or p.get("journal_conf"),
                    "doi": p.get("doi"),
                    "url": p.get("url"),
                }
            )
        return {"strategy": strategy, "path": path}


@app.post("/api/papers/{paper_id}/sync", response_model=Paper)
def sync_paper(paper_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Paper not found")
        paper = dict(row)
        if not paper.get("s2_paper_id"):
            s2_record = _resolve_semantic_scholar(paper.get("title"), paper.get("doi"))
            if s2_record:
                paper = upsert_paper(conn, s2_record)
        new_ids = _sync_citations(conn, paper, paper.get("s2_paper_id"))
        if os.getenv("SYNC_EXPAND", "true").lower() == "true" and new_ids:
            for nid in new_ids:
                enqueue_paper(conn, nid, delay_seconds=600)
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return Paper(**dict(row))


@app.post("/api/sync/enqueue-all")
def sync_enqueue_all():
    with get_conn() as conn:
        count = enqueue_all(conn)
        return {"enqueued": count}


@app.post("/api/sync/enqueue/{paper_id}")
def sync_enqueue_one(paper_id: int):
    with get_conn() as conn:
        enqueue_paper(conn, paper_id, delay_seconds=0)
        return {"enqueued": paper_id}


@app.post("/api/sync/run")
def sync_run(limit: int = 5):
    results = {"processed": 0, "failed": 0}
    with get_conn() as conn:
        _maybe_cleanup(conn)
        due = fetch_due(conn, limit)
        expand = os.getenv("SYNC_EXPAND", "true").lower() == "true"
        for pid in due:
            mark_running(conn, pid)
            try:
                row = conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()
                if not row:
                    mark_success(conn, pid)
                    continue
                paper = dict(row)
                if not paper.get("s2_paper_id"):
                    s2_record = _resolve_semantic_scholar(paper.get("title"), paper.get("doi"))
                    if s2_record:
                        paper = upsert_paper(conn, s2_record)
                new_ids = _sync_citations(conn, paper, paper.get("s2_paper_id"))
                if expand and new_ids:
                    for nid in new_ids:
                        enqueue_paper(conn, nid, delay_seconds=600)
                mark_success(conn, pid)
                results["processed"] += 1
            except Exception as exc:
                mark_failure(conn, pid, str(exc))
                results["failed"] += 1
    return results


@app.get("/api/sync/status")
def sync_status():
    with get_conn() as conn:
        return queue_stats(conn)


@app.post("/api/maintenance/cleanup")
def maintenance_cleanup():
    with get_conn() as conn:
        stats = _cleanup_citations(conn)
        return {"status": "ok", **stats}


@app.post("/api/papers/backfill")
def backfill_papers(
    limit: int = 10,
    summary: bool = True,
    references: bool = True,
    embeddings: bool = True,
    metrics: bool = True,
    force: bool = False,
):
    processed = 0
    summary_added = 0
    references_parsed = 0
    chunks_indexed = 0
    metrics_upserted = 0
    auto_experiments = 0
    schemas_extracted = 0
    touched = False
    with get_conn() as conn:
        clauses = ["file_path IS NOT NULL"]
        if not force:
            if summary and references:
                clauses.append("(summary_one IS NULL OR summary_one = '' OR refs_parsed_at IS NULL)")
            elif summary:
                clauses.append("(summary_one IS NULL OR summary_one = '')")
            elif references:
                clauses.append("(refs_parsed_at IS NULL)")
        where = " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT * FROM papers WHERE {where} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        for row in rows:
            processed += 1
            paper = dict(row)
            updates: Dict[str, Any] = {}

            if summary and (force or not paper.get("summary_one")):
                title = paper.get("title")
                abstract = paper.get("abstract")
                if paper.get("file_path") and (not title or not abstract):
                    extracted = extract_metadata(paper["file_path"])
                    if not title and extracted.title:
                        title = extracted.title
                        updates["title"] = extracted.title
                    if not abstract and extracted.abstract:
                        abstract = extracted.abstract
                        updates["abstract"] = extracted.abstract
                summary_text = summarize_one_line(title, abstract)
                if summary_text:
                    updates["summary_one"] = summary_text
                    summary_added += 1

            if references and (force or not paper.get("refs_parsed_at")):
                ref_text = extract_references_text(paper.get("file_path") or "")
                if ref_text:
                    refs = extract_references(ref_text, max_items=30)
                    if not refs:
                        refs = _extract_dois_from_text(ref_text, limit=30)
                    _sync_references_from_text(conn, paper, refs)
                updates["refs_parsed_at"] = int(time.time())
                references_parsed += 1

            full_text = ""
            table_text = ""
            if paper.get("file_path") and (embeddings or metrics):
                full_text = extract_markdown(paper["file_path"]) or extract_full_text(paper["file_path"])
                table_text = extract_tables_text(paper["file_path"])

            if embeddings and full_text:
                chunks_indexed += _index_paper_chunks(conn, paper["id"], full_text)
                touched = True
                schema = _extract_and_store_schema(
                    conn,
                    paper_id=paper["id"],
                    full_text=full_text,
                    source="backfill",
                    force=force,
                )
                if schema.get("event_types"):
                    schemas_extracted += 1

            if metrics and full_text:
                metric_text = "\n".join([full_text, table_text]).strip()
                extracted_metrics = extract_ee_metrics(metric_text, limit=40)
                llm_metrics = extract_reported_metrics(metric_text[:18000], max_items=20)
                if llm_metrics:
                    extracted_metrics.extend(llm_metrics)
                metrics_upserted += _upsert_ee_metrics(
                    conn, paper["id"], extracted_metrics, source="backfill"
                )
                auto_experiments += _sync_auto_experiments_from_metrics(
                    conn, paper["id"], extracted_metrics
                )
                touched = True

            if updates:
                repo_update_paper(conn, paper["id"], updates)
                touched = True
        if touched:
            _sync_search_fts(conn)
    return {
        "processed": processed,
        "summary_added": summary_added,
        "references_parsed": references_parsed,
        "chunks_indexed": chunks_indexed,
        "metrics_upserted": metrics_upserted,
        "auto_experiments": auto_experiments,
        "schemas_extracted": schemas_extracted,
    }

def _contains_cjk(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _guess_source_type(title: Optional[str], abstract: Optional[str]) -> Optional[str]:
    if _contains_cjk(title) or _contains_cjk(abstract):
        return "CNKI"
    if title or abstract:
        return "Global"
    return None


def _get_active_subfields(conn) -> List[str]:
    rows = conn.execute("SELECT name FROM subfields WHERE active = 1").fetchall()
    return [r["name"] for r in rows]


def _get_subfield_examples(conn, limit: int = 6) -> List[Dict[str, str]]:
    rows = conn.execute(
        "SELECT text, sub_field FROM subfield_feedback ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    examples = []
    for r in rows:
        if r["text"] and r["sub_field"]:
            examples.append({"text": r["text"], "sub_field": r["sub_field"]})
    return examples


def _record_subfield_feedback(conn, paper_id: int, text: str, sub_field: str) -> None:
    if not text or not sub_field:
        return
    conn.execute(
        "INSERT INTO subfield_feedback (paper_id, text, sub_field, created_at) VALUES (?, ?, ?, ?)",
        (paper_id, text, sub_field, int(time.time())),
    )


def _infer_sub_field(text: str) -> Optional[str]:
    text_lower = (text or "").lower()
    if "few-shot" in text_lower:
        return "Few-shot"
    if "zero-shot" in text_lower:
        return "Zero-shot"
    if "document-level" in text_lower or "doc-level" in text_lower:
        return "Doc-level"
    if "argument" in text_lower:
        return "Argument Extraction"
    if "trigger" in text_lower:
        return "Trigger Identification"
    if "event detection" in text_lower:
        return "Event Detection"
    if "role labeling" in text_lower or "role labeling" in text_lower:
        return "Event Argument Role Labeling"
    if "joint" in text_lower:
        return "Joint Extraction"
    if "open-domain" in text_lower or "open domain" in text_lower:
        return "Open-domain"
    if "cross-lingual" in text_lower or "cross lingual" in text_lower:
        return "Cross-lingual"
    if "low-resource" in text_lower or "low resource" in text_lower:
        return "Low-resource"
    return None


def _extract_dois_from_text(text: str, limit: int = 30) -> List[Dict[str, Any]]:
    if not text:
        return []
    found = re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.IGNORECASE)
    items: List[Dict[str, Any]] = []
    seen = set()
    for doi in found:
        doi_norm = doi.strip().lower()
        if doi_norm in seen:
            continue
        seen.add(doi_norm)
        items.append({"title": None, "doi": doi})
        if len(items) >= limit:
            break
    return items


def _merge_metadata(
    extracted,
    llm_data: Dict[str, Any],
    s2_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["title"] = llm_data.get("title") or extracted.title
    data["authors"] = llm_data.get("authors") or extracted.authors
    data["year"] = llm_data.get("year") or extracted.year
    data["journal_conf"] = llm_data.get("journal_conf") or extracted.journal_conf
    data["doi"] = llm_data.get("doi") or extracted.doi
    data["abstract"] = llm_data.get("abstract") or extracted.abstract
    data["sub_field"] = llm_data.get("sub_field")
    data["open_sub_field"] = llm_data.get("open_sub_field")
    data["proposed_method_name"] = llm_data.get("proposed_method_name")
    dynamic_tags = _normalize_open_tags(llm_data.get("dynamic_tags"))
    if dynamic_tags:
        data["dynamic_tags"] = _json_dumps(dynamic_tags)
    if s2_data:
        data.update({k: v for k, v in s2_data.items() if v is not None})
    if not data.get("ccf_level"):
        venue = data.get("journal_conf") or data.get("venue")
        data["ccf_level"] = classify_venue(venue)
    return data


def _sync_zotero_meta(conn, paper: Dict[str, Any]) -> bool:
    key = paper.get("zotero_item_key")
    if not key:
        return False
    item = get_item(key)
    if not item:
        return False
    data = item.get("data") or {}
    zotero_venue = data.get("publicationTitle") or data.get("proceedingsTitle") or data.get(
        "conferenceName"
    )
    if zotero_venue and paper.get("ccf_level"):
        learn_alias(zotero_venue, paper.get("ccf_level"))
    library, item_id = extract_library_info(item)
    updates: Dict[str, Any] = {}
    if library and library != paper.get("zotero_library"):
        updates["zotero_library"] = library
    if item_id and item_id != paper.get("zotero_item_id"):
        updates["zotero_item_id"] = item_id
    if updates:
        repo_update_paper(conn, paper["id"], updates)
        return True
    return False


def _resolve_semantic_scholar(
    title: Optional[str], doi: Optional[str]
) -> Optional[Dict[str, Any]]:
    query = doi or title
    if not query:
        return None
    match = search_match(query)
    if isinstance(match, list):
        match = match[0] if match else None
    if not match:
        return None
    return to_paper_record(match)


def _infer_intent_if_enabled(
    source_title: Optional[str],
    source_abstract: Optional[str],
    target_title: Optional[str],
) -> Tuple[Optional[str], Optional[float]]:
    if os.getenv("CITATION_INTENT_AUTO", "false").lower() != "true":
        return None, None
    result = classify_citation_intent(source_title, source_abstract, target_title)
    intent = result.get("intent")
    confidence = result.get("confidence")
    try:
        conf_value = float(confidence) if confidence is not None else None
    except Exception:
        conf_value = None
    return intent, conf_value


def _sync_citations(conn, paper_row: Dict[str, Any], s2_paper_id: Optional[str]) -> List[int]:
    if not s2_paper_id:
        return []
    limit = int(os.getenv("S2_EDGE_LIMIT", "100"))
    references = get_references(s2_paper_id, limit=limit)
    citations = get_citations(s2_paper_id, limit=limit)
    new_ids: List[int] = []

    # References: current -> cited
    for edge in references:
        target_data = extract_paper_from_edge(edge, "citedPaper")
        if not target_data:
            continue
        record = to_paper_record(target_data)
        record["ccf_level"] = record.get("ccf_level") or classify_venue(
            record.get("venue") or record.get("journal_conf")
        )
        target = upsert_paper(conn, record)
        sim = title_similarity(record.get("title"), target.get("title"))
        intent, intent_conf = _infer_intent_if_enabled(
            paper_row.get("title"),
            paper_row.get("abstract"),
            target.get("title"),
        )
        add_citation_edge(
            conn,
            paper_row["id"],
            target["id"],
            confidence=classify_edge_confidence(
                "semantic_scholar",
                has_doi=bool(record.get("doi")),
                title_quality=sim,
            ),
            edge_source="semantic_scholar",
            evidence=record.get("doi") or record.get("title"),
            intent=intent,
            intent_confidence=intent_conf,
        )
        if target["id"] != paper_row["id"] and not target.get("file_path"):
            new_ids.append(target["id"])

    # Citations: citing -> current
    for edge in citations:
        source_data = extract_paper_from_edge(edge, "citingPaper")
        if not source_data:
            continue
        record = to_paper_record(source_data)
        record["ccf_level"] = record.get("ccf_level") or classify_venue(
            record.get("venue") or record.get("journal_conf")
        )
        source = upsert_paper(conn, record)
        sim = title_similarity(record.get("title"), source.get("title"))
        intent, intent_conf = _infer_intent_if_enabled(
            source.get("title"),
            source.get("abstract"),
            paper_row.get("title"),
        )
        add_citation_edge(
            conn,
            source["id"],
            paper_row["id"],
            confidence=classify_edge_confidence(
                "semantic_scholar",
                has_doi=bool(record.get("doi")),
                title_quality=sim,
            ),
            edge_source="semantic_scholar",
            evidence=record.get("doi") or record.get("title"),
            intent=intent,
            intent_confidence=intent_conf,
        )
        if source["id"] != paper_row["id"] and not source.get("file_path"):
            new_ids.append(source["id"])

    return new_ids


def _sync_references_from_text(
    conn, paper_row: Dict[str, Any], references: List[Dict[str, Any]]
) -> None:
    if not references:
        return
    for ref in references:
        title = ref.get("title")
        doi = ref.get("doi")
        if not title and not doi:
            continue
        s2_record = _resolve_semantic_scholar(title, doi)
        if s2_record:
            record = s2_record
        else:
            record = {"title": title, "doi": doi}
        record["ccf_level"] = record.get("ccf_level") or classify_venue(
            record.get("venue") or record.get("journal_conf")
        )
        target = upsert_paper(conn, record)
        title_sim = title_similarity(title, target.get("title")) if title else 0.6
        intent, intent_conf = _infer_intent_if_enabled(
            paper_row.get("title"),
            paper_row.get("abstract"),
            target.get("title"),
        )
        add_citation_edge(
            conn,
            paper_row["id"],
            target["id"],
            confidence=classify_edge_confidence(
                "llm_reference",
                has_doi=bool(doi),
                title_quality=title_sim,
            ),
            edge_source="llm_reference",
            evidence=doi or title,
            intent=intent,
            intent_confidence=intent_conf,
        )


@app.post("/api/papers/upload", response_model=Paper)
def upload_paper(file: UploadFile = File(...)):
    is_pdf = (file.content_type in {"application/pdf", "application/x-pdf"}) or (
        file.filename and file.filename.lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(STORAGE_DIR, exist_ok=True)
    safe_name = file.filename or "paper.pdf"
    file_id = uuid4().hex
    filename = f"{file_id}_{safe_name}"
    file_path = os.path.join(STORAGE_DIR, filename)

    with open(file_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    extracted = extract_metadata(file_path)
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    raw_text = extract_text(file_path, max_pages=3)
    full_text = extract_markdown(file_path) or extract_full_text(file_path)
    table_text = extract_tables_text(file_path)
    with get_conn() as conn:
        subfields = _get_active_subfields(conn)
        examples = _get_subfield_examples(conn)
    llm_data = enrich_metadata(raw_text, sub_fields=subfields, examples=examples) or {}

    merged = _merge_metadata(extracted, llm_data, None)
    if not merged.get("doi") and merged.get("title"):
        merged["doi"] = lookup_doi(merged["title"])

    s2_record = _resolve_semantic_scholar(merged.get("title"), merged.get("doi"))
    if s2_record:
        merged = _merge_metadata(extracted, llm_data, s2_record)

    merged["file_path"] = file_path
    merged["file_hash"] = file_hash
    if not merged.get("summary_one"):
        merged["summary_one"] = summarize_one_line(merged.get("title"), merged.get("abstract"))
    if not merged.get("sub_field") and merged.get("open_sub_field"):
        merged["sub_field"] = merged.get("open_sub_field")
    if not merged.get("sub_field"):
        merged["sub_field"] = _infer_sub_field(
            f"{merged.get('title') or ''} {merged.get('abstract') or ''}"
        )
    merged["source_type"] = merged.get("source_type") or _guess_source_type(
        merged.get("title"), merged.get("abstract")
    )

    with get_conn() as conn:
        merged.setdefault("read_status", 0)
        row = upsert_paper(conn, merged)
        if row.get("sub_field"):
            feedback_text = f"{row.get('title') or ''}\n{row.get('abstract') or ''}".strip()
            if feedback_text:
                _record_subfield_feedback(conn, row["id"], feedback_text, row["sub_field"])
        ref_text = extract_references_text(file_path)
        if ref_text:
            references = extract_references(ref_text, max_items=30)
            if not references:
                references = _extract_dois_from_text(ref_text, limit=30)
            _sync_references_from_text(conn, row, references)
        _sync_citations(conn, row, row.get("s2_paper_id"))
        if full_text:
            _index_paper_chunks(conn, row["id"], full_text)
            _extract_and_store_schema(
                conn,
                paper_id=row["id"],
                full_text=full_text,
                source="upload",
                force=False,
            )
            metric_text = "\n".join([full_text, table_text]).strip()
            metrics = extract_ee_metrics(metric_text, limit=40)
            llm_metrics = extract_reported_metrics(metric_text[:18000], max_items=20)
            if llm_metrics:
                metrics.extend(llm_metrics)
            _upsert_ee_metrics(conn, row["id"], metrics, source="upload")
            _sync_auto_experiments_from_metrics(conn, row["id"], metrics)
        enqueue_paper(conn, row["id"], delay_seconds=0)
        _sync_search_fts(conn)
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (row["id"],)).fetchone()

    return Paper(**dict(row))
