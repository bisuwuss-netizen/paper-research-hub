from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from .db import get_conn


PAPER_FIELDS = [
    "title",
    "authors",
    "year",
    "journal_conf",
    "ccf_level",
    "source_type",
    "sub_field",
    "read_status",
    "file_path",
    "doi",
    "abstract",
    "s2_paper_id",
    "s2_corpus_id",
    "citation_count",
    "reference_count",
    "influential_citation_count",
    "citation_velocity",
    "url",
    "venue",
    "keywords",
    "cluster_id",
    "zotero_item_key",
    "zotero_library",
    "zotero_item_id",
    "file_hash",
    "summary_one",
    "refs_parsed_at",
    "created_at",
    "updated_at",
    "proposed_method_name",
    "dynamic_tags",
    "embedding",
    "open_sub_field",
]


def _normalize_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    clean = re.sub(r"\s+", " ", title).strip().lower()
    return clean


def find_paper_by_doi(conn, doi: Optional[str]):
    if not doi:
        return None
    row = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
    return dict(row) if row else None


def find_paper_by_s2_id(conn, s2_paper_id: Optional[str]):
    if not s2_paper_id:
        return None
    row = conn.execute("SELECT * FROM papers WHERE s2_paper_id = ?", (s2_paper_id,)).fetchone()
    return dict(row) if row else None


def find_paper_by_title_year(conn, title: Optional[str], year: Optional[int]):
    if not title:
        return None
    row = conn.execute(
        "SELECT * FROM papers WHERE title = ? AND (? IS NULL OR year = ?)",
        (title, year, year),
    ).fetchone()
    return dict(row) if row else None


def _suspicious_title(title: Optional[str]) -> bool:
    if not title:
        return True
    lower = title.lower()
    if "@" in title or "http" in lower or "www" in lower:
        return True
    if re.search(r"\b(university|institute|laboratory|department|school|college)\b", lower):
        return True
    if len(title) < 8:
        return True
    return False


def insert_paper(conn, data: Dict[str, Any]) -> int:
    fields = PAPER_FIELDS
    values = [data.get(field) for field in fields]
    now_ts = int(time.time())
    if "created_at" in fields:
        idx = fields.index("created_at")
        if values[idx] is None:
            values[idx] = now_ts
    if "updated_at" in fields:
        idx = fields.index("updated_at")
        if values[idx] is None:
            values[idx] = now_ts
    # Keep status non-null for response models and UI logic.
    if "read_status" in fields:
        idx = fields.index("read_status")
        if values[idx] is None:
            values[idx] = 0
    placeholders = ",".join(["?"] * len(fields))
    cur = conn.execute(
        f"INSERT INTO papers ({','.join(fields)}) VALUES ({placeholders})",
        values,
    )
    return cur.lastrowid


def update_paper(conn, paper_id: int, data: Dict[str, Any]) -> None:
    fields = []
    values = []
    updates = dict(data)
    updates["updated_at"] = int(time.time())
    for key, value in updates.items():
        if key == "read_status" and value is None:
            value = 0
        fields.append(f"{key} = ?")
        values.append(value)
    if not fields:
        return
    values.append(paper_id)
    conn.execute(f"UPDATE papers SET {', '.join(fields)} WHERE id = ?", values)


def upsert_paper(conn, data: Dict[str, Any]) -> Dict[str, Any]:
    existing = None
    file_hash = data.get("file_hash")
    if file_hash:
        row = conn.execute("SELECT * FROM papers WHERE file_hash = ?", (file_hash,)).fetchone()
        existing = dict(row) if row else None
    if not existing:
        existing = find_paper_by_doi(conn, data.get("doi")) or find_paper_by_s2_id(
            conn, data.get("s2_paper_id")
        )
    if not existing:
        if not _suspicious_title(data.get("title")):
            existing = find_paper_by_title_year(conn, data.get("title"), data.get("year"))

    if not existing:
        paper_id = insert_paper(conn, data)
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return dict(row)

    # Merge: prefer existing non-null; fill gaps with new data
    merged = dict(existing)
    for key, value in data.items():
        if value is None:
            continue
        if merged.get(key) is None or merged.get(key) == "":
            merged[key] = value
        elif key in {
            "citation_count",
            "reference_count",
            "influential_citation_count",
            "citation_velocity",
        }:
            merged[key] = value
        elif key in {"file_path", "file_hash"}:
            merged[key] = value
    update_fields = {key: merged.get(key) for key in data.keys()}
    update_paper(conn, existing["id"], update_fields)
    row = conn.execute("SELECT * FROM papers WHERE id = ?", (existing["id"],)).fetchone()
    return dict(row)


def add_citation_edge(
    conn,
    source_id: int,
    target_id: int,
    confidence: float = 1.0,
    edge_source: str | None = None,
    evidence: str | None = None,
    intent: str | None = None,
    intent_confidence: float | None = None,
    context_snippet: str | None = None,
    intent_source: str | None = None,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO citations (
                source_paper_id, target_paper_id, confidence, edge_source, evidence,
                intent, intent_confidence, context_snippet, intent_source, last_verified_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
            ON CONFLICT(source_paper_id, target_paper_id) DO UPDATE SET
                confidence = CASE
                    WHEN excluded.confidence > COALESCE(citations.confidence, 0)
                    THEN excluded.confidence
                    ELSE citations.confidence
                END,
                edge_source = COALESCE(excluded.edge_source, citations.edge_source),
                evidence = COALESCE(excluded.evidence, citations.evidence),
                intent = COALESCE(excluded.intent, citations.intent),
                intent_confidence = COALESCE(excluded.intent_confidence, citations.intent_confidence),
                context_snippet = COALESCE(excluded.context_snippet, citations.context_snippet),
                intent_source = COALESCE(excluded.intent_source, citations.intent_source),
                last_verified_at = strftime('%s','now')
            """,
            (
                source_id,
                target_id,
                confidence,
                edge_source,
                evidence,
                intent,
                intent_confidence,
                context_snippet,
                intent_source,
            ),
        )
    except Exception:
        return
