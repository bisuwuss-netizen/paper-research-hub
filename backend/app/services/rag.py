from __future__ import annotations

import json
import math
from typing import Dict, List, Iterable, Any


def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(
        n_features=512,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )


def text_to_embedding(text: str) -> List[float]:
    content = (text or "").strip()
    if not content:
        return []
    vec = _vectorizer().transform([content]).toarray()[0].tolist()
    return [float(v) for v in vec]


def embedding_to_json(embedding: List[float]) -> str:
    return json.dumps(embedding, ensure_ascii=False)


def embedding_from_json(value: str | None) -> List[float]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[float] = []
    for item in data:
        try:
            out.append(float(item))
        except Exception:
            out.append(0.0)
    return out


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        va = a[i]
        vb = b[i]
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def rank_chunks(query: str, chunks: Iterable[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    q = text_to_embedding(query)
    if not q:
        return []
    ranked: List[Dict[str, Any]] = []
    for chunk in chunks:
        emb = chunk.get("chunk_embedding")
        if isinstance(emb, str):
            vec = embedding_from_json(emb)
        elif isinstance(emb, list):
            vec = [float(x) for x in emb]
        else:
            vec = []
        score = cosine_similarity(q, vec)
        if score <= 0:
            continue
        ranked.append({**chunk, "score": float(score)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
