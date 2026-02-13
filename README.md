# PaperTrail MVP

This repo contains a full-stack skeleton for PaperTrail:
- FastAPI + SQLite backend (PDF upload, metadata extraction, citation graph)
- React + TypeScript + Ant Design frontend (list + graph + recommendations)

## Backend

```bash
cd /Users/bisuv/Developer/paper-research-hub/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Optional LLM enrichment
- Copy `backend/.env.example` to `backend/.env`
- Set `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_ENDPOINT`, `LLM_MODEL`
- Optional: `LLM_RESPONSE_JSON=true`, `LLM_MAX_RETRIES=2`

### Optional Semantic Scholar / Crossref
- `S2_API_KEY` (if you have one)
- `S2_EDGE_LIMIT` (default 100)
- `CROSSREF_MAILTO` (polite usage for DOI lookup)

### Optional Zotero title matching
- `ZOTERO_API_KEY`
- `ZOTERO_USER_ID` or `ZOTERO_GROUP_ID`
- `ZOTERO_LIBRARY_TYPE` = `user` or `group`

### Optional sync worker
- `SYNC_ENABLED=true` to start background sync
- `SYNC_POLL_INTERVAL`, `SYNC_BATCH_SIZE`, `SYNC_RESYNC_INTERVAL`
- `SYNC_RETRY_BASE`, `SYNC_EXPAND`, `SYNC_AUTO_ENQUEUE`

## Frontend

```bash
cd /Users/bisuv/Developer/paper-research-hub/frontend
npm install
npm run dev
```

UI runs at `http://localhost:5173` and the API at `http://localhost:8000`.

## Features
- Upload PDF
- Extract title/authors/year/abstract
- Enrich via LLM (structured schema + retries)
- Fetch citation network (Semantic Scholar)
- Graph view with filters + recommendation strategies
- Timeline visualization (drag to preview, confirm to apply)
- Layout switch (force / hierarchy / timeline)
- Reading path export (CSV, Markdown, PDF, BibTeX)
- Zotero key save + open protocol + push updates
- CCF auto-tagging (full list + alias/abbreviation matching)
- Lazy load / neighbor expansion for large graphs
- Sub-field management + label correction

## Data
- SQLite: `backend/data/app.db`
- PDFs: `backend/storage/pdfs`

## CCF List
- Full list: `backend/app/resources/ccf_list.json`
- Alias list: `backend/app/resources/ccf_aliases.json`
- Refresh existing records via `POST /api/ccf/refresh`
