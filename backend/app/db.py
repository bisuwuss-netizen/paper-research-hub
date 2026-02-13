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
    "proposed_method_name": "TEXT",
    "dynamic_tags": "TEXT",
    "embedding": "TEXT",
    "open_sub_field": "TEXT",
}


CITATIONS_COLUMNS = {
    "confidence": "REAL DEFAULT 1.0",
    "edge_source": "TEXT",
    "last_verified_at": "INTEGER",
    "evidence": "TEXT",
    "intent": "TEXT",
    "intent_confidence": "REAL",
    "context_snippet": "TEXT",
    "intent_source": "TEXT",
}


EXPERIMENTS_COLUMNS = {
    "dataset_name": "TEXT",
    "trigger_f1": "REAL",
    "argument_f1": "REAL",
    "precision": "REAL",
    "recall": "REAL",
    "f1": "REAL",
    "dataset": "TEXT",
    "split": "TEXT",
    "metric_name": "TEXT",
    "metric_value": "REAL",
    "is_sota": "INTEGER DEFAULT 0",
}


EE_METRICS_COLUMNS = {
    "table_id": "TEXT",
    "row_index": "INTEGER",
    "col_index": "INTEGER",
    "cell_text": "TEXT",
    "provenance_json": "TEXT",
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
            CREATE TABLE IF NOT EXISTS note_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id INTEGER,
                target_paper_id INTEGER,
                link_text TEXT,
                context TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                FOREIGN KEY(source_paper_id) REFERENCES papers(id),
                FOREIGN KEY(target_paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_schemas (
                paper_id INTEGER PRIMARY KEY,
                event_types_json TEXT,
                role_types_json TEXT,
                schema_notes TEXT,
                confidence REAL,
                source TEXT,
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
        _ensure_columns(conn, "experiments", EXPERIMENTS_COLUMNS)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                chunk_index INTEGER,
                chunk_text TEXT,
                chunk_embedding TEXT,
                created_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ee_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                dataset_name TEXT,
                precision REAL,
                recall REAL,
                f1 REAL,
                trigger_f1 REAL,
                argument_f1 REAL,
                source TEXT,
                confidence REAL,
                created_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        _ensure_columns(conn, "ee_metrics", EE_METRICS_COLUMNS)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ee_metric_cells (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                metric_key TEXT,
                metric_value REAL,
                dataset_name TEXT,
                table_id TEXT,
                row_index INTEGER,
                col_index INTEGER,
                cell_text TEXT,
                parser TEXT,
                confidence REAL,
                created_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT UNIQUE,
                concept_type TEXT DEFAULT 'event',
                aliases_json TEXT,
                created_at INTEGER,
                updated_at INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_schema_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                schema_name TEXT,
                canonical_concept_id INTEGER,
                concept_type TEXT DEFAULT 'event',
                confidence REAL,
                created_at INTEGER,
                updated_at INTEGER,
                FOREIGN KEY(paper_id) REFERENCES papers(id),
                FOREIGN KEY(canonical_concept_id) REFERENCES schema_concepts(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idea_capsules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                status TEXT DEFAULT 'seed',
                priority INTEGER DEFAULT 2,
                linked_papers_json TEXT,
                tags_json TEXT,
                source_note_paper_id INTEGER,
                created_at INTEGER,
                updated_at INTEGER,
                FOREIGN KEY(source_note_paper_id) REFERENCES papers(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                language TEXT DEFAULT 'zh',
                created_at INTEGER,
                updated_at INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                role TEXT,
                content TEXT,
                paper_ids_json TEXT,
                trace_score REAL,
                sources_json TEXT,
                created_at INTEGER,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT,
                status TEXT,
                payload_json TEXT,
                result_json TEXT,
                attempts INTEGER DEFAULT 0,
                error TEXT,
                started_at INTEGER,
                finished_at INTEGER,
                next_retry_at INTEGER,
                created_at INTEGER,
                updated_at INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                source_job_id INTEGER,
                message TEXT,
                payload_json TEXT,
                resolved INTEGER DEFAULT 0,
                created_at INTEGER,
                resolved_at INTEGER,
                FOREIGN KEY(source_job_id) REFERENCES job_runs(id)
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
            "CREATE INDEX IF NOT EXISTS idx_citations_intent "
            "ON citations(intent, intent_confidence)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_citations_context "
            "ON citations(source_paper_id, target_paper_id, context_snippet)"
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
            "CREATE INDEX IF NOT EXISTS idx_note_links_target ON note_links(target_paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_note_links_source ON note_links(source_paper_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_links_unique "
            "ON note_links(source_paper_id, target_paper_id, link_text)"
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_paper ON paper_chunks(paper_id, chunk_index)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ee_metrics_paper ON ee_metrics(paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ee_metric_cells_paper ON ee_metric_cells(paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ee_metric_cells_dataset ON ee_metric_cells(dataset_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_schema_concepts_name ON schema_concepts(canonical_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_schema_concepts_paper ON paper_schema_concepts(paper_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_schema_concepts_concept ON paper_schema_concepts(canonical_concept_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_capsules_status ON idea_capsules(status, priority)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs(status, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON system_alerts(resolved, created_at)"
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
