from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")

DEFAULT_SUBFIELDS = [
    "Few-shot",
    "Zero-shot",
    "Doc-level",
    "Argument Extraction",
    "Trigger Identification",
    "Event Detection",
    "Event Argument Role Labeling",
    "Joint Extraction",
    "Open-domain",
    "Cross-lingual",
    "Low-resource",
]


PAPERS_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "title": "TEXT",
    "authors": "TEXT",
    "year": "INTEGER",
    "journal_conf": "TEXT",
    "ccf_level": "TEXT",
    "source_type": "TEXT",
    "sub_field": "TEXT",
    "read_status": "INTEGER DEFAULT 0",
    "file_path": "TEXT",
    "doi": "TEXT",
    "abstract": "TEXT",
    "s2_paper_id": "TEXT",
    "s2_corpus_id": "TEXT",
    "citation_count": "INTEGER",
    "reference_count": "INTEGER",
    "influential_citation_count": "INTEGER",
    "citation_velocity": "REAL",
    "url": "TEXT",
    "venue": "TEXT",
    "keywords": "TEXT",
    "cluster_id": "TEXT",
    "zotero_item_key": "TEXT",
    "zotero_library": "TEXT",
    "zotero_item_id": "INTEGER",
    "file_hash": "TEXT",
    "summary_one": "TEXT",
    "refs_parsed_at": "INTEGER",
    "created_at": "INTEGER",
    "updated_at": "INTEGER",
}


CITATIONS_COLUMNS = {
    "confidence": "REAL DEFAULT 1.0",
    "edge_source": "TEXT",
    "last_verified_at": "INTEGER",
    "evidence": "TEXT",
}


def _get_existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = _get_existing_columns(conn, table)
    for name, definition in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def ensure_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                year INTEGER,
                journal_conf TEXT,
                ccf_level TEXT,
                source_type TEXT,
                sub_field TEXT,
                read_status INTEGER DEFAULT 0,
                file_path TEXT,
                doi TEXT,
                abstract TEXT
            );
            """
        )
        _ensure_columns(conn, "papers", PAPERS_COLUMNS)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citations (
                source_paper_id INTEGER,
                target_paper_id INTEGER,
                FOREIGN KEY(source_paper_id) REFERENCES papers(id),
                FOREIGN KEY(target_paper_id) REFERENCES papers(id)
            );
            """
        )
        _ensure_columns(conn, "citations", CITATIONS_COLUMNS)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_queue (
                paper_id INTEGER PRIMARY KEY,
                status TEXT DEFAULT 'idle',
                attempts INTEGER DEFAULT 0,
                next_run_at INTEGER,
                last_run_at INTEGER,
                last_error TEXT,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subfields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT,
                active INTEGER DEFAULT 1,
                created_at INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subfield_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                text TEXT,
                sub_field TEXT,
                created_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_notes (
                paper_id INTEGER PRIMARY KEY,
                method TEXT,
                datasets TEXT,
                conclusions TEXT,
                reproducibility TEXT,
                risks TEXT,
                notes TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reading_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                title TEXT,
                status TEXT DEFAULT 'todo',
                priority INTEGER DEFAULT 2,
                due_date TEXT,
                next_review_at INTEGER,
                interval_days INTEGER DEFAULT 1,
                last_review_at INTEGER,
                created_at INTEGER,
                updated_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                name TEXT,
                model TEXT,
                params_json TEXT,
                metrics_json TEXT,
                result_summary TEXT,
                artifact_path TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS zotero_mapping_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                mapping_json TEXT,
                created_at INTEGER,
                updated_at INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS zotero_sync_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                direction TEXT,
                action TEXT,
                status TEXT,
                conflict_strategy TEXT,
                details TEXT,
                created_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_type TEXT,
                period_start TEXT,
                period_end TEXT,
                payload_json TEXT,
                created_at INTEGER
            );
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS paper_search_fts
                USING fts5(
                    paper_id UNINDEXED,
                    title,
                    abstract,
                    notes
                );
                """
            )
        except sqlite3.OperationalError:
            # FTS5 may be unavailable in some SQLite builds.
            pass
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_citations_unique "
            "ON citations(source_paper_id, target_paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_conf "
            "ON citations(confidence)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_queue_next "
            "ON sync_queue(next_run_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_s2 ON papers(s2_paper_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_updated_at ON papers(updated_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_subfield ON subfield_feedback(sub_field)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_paper_status ON reading_tasks(paper_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_review_due ON reading_tasks(next_review_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_paper ON experiments(paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_zotero_logs_created ON zotero_sync_logs(created_at)"
        )
        conn.execute("UPDATE papers SET read_status = 0 WHERE read_status IS NULL")
        conn.execute(
            "UPDATE papers SET created_at = strftime('%s','now') WHERE created_at IS NULL"
        )
        conn.execute(
            "UPDATE papers SET updated_at = strftime('%s','now') WHERE updated_at IS NULL"
        )
        template_count = conn.execute(
            "SELECT COUNT(*) AS c FROM zotero_mapping_templates"
        ).fetchone()
        if template_count and template_count[0] == 0:
            conn.execute(
                """
                INSERT INTO zotero_mapping_templates (name, mapping_json, created_at, updated_at)
                VALUES (
                    'default',
                    '{"title":"title","abstract":"abstractNote","authors":"creators","doi":"DOI","venue":"publicationTitle","year":"date","summary_one":"extra"}',
                    strftime('%s','now'),
                    strftime('%s','now')
                )
                """
            )
        existing = conn.execute("SELECT COUNT(*) AS c FROM subfields").fetchone()
        if existing and existing[0] == 0:
            for name in DEFAULT_SUBFIELDS:
                conn.execute(
                    "INSERT INTO subfields (name, description, active, created_at) VALUES (?, ?, ?, strftime('%s','now'))",
                    (name, None, 1),
                )
        conn.commit()


@contextmanager
def get_conn():
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
