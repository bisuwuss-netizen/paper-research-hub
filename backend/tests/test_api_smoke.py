from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SYNC_ENABLED", "false")
    monkeypatch.setenv("SYNC_AUTO_ENQUEUE", "false")

    from app import db as db_module

    monkeypatch.setattr(db_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "app.db"))
    db_module.ensure_db()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json().get("status") == "ok"


def test_idea_capsule_crud(client: TestClient):
    create = client.post(
        "/api/idea-capsules",
        json={
            "title": "Capsule A",
            "content": "Test idea",
            "status": "seed",
            "priority": 3,
            "linked_papers": [],
            "tags": ["ee", "few-shot"],
        },
    )
    assert create.status_code == 200
    capsule_id = create.json()["id"]

    listing = client.get("/api/idea-capsules")
    assert listing.status_code == 200
    assert listing.json()["count"] >= 1

    update = client.patch(f"/api/idea-capsules/{capsule_id}", json={"status": "incubating"})
    assert update.status_code == 200
    assert update.json()["status"] == "incubating"

    remove = client.delete(f"/api/idea-capsules/{capsule_id}")
    assert remove.status_code == 200


def test_chat_sessions(client: TestClient):
    created = client.post("/api/chat/sessions", json={"title": "smoke", "language": "zh"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    listed = client.get("/api/chat/sessions")
    assert listed.status_code == 200
    assert isinstance(listed.json(), list)

    messages = client.get(f"/api/chat/sessions/{session_id}/messages")
    assert messages.status_code == 200
    assert isinstance(messages.json(), list)


def test_jobs_and_schema_endpoints(client: TestClient):
    jobs = client.get("/api/jobs/history")
    assert jobs.status_code == 200
    assert "items" in jobs.json()

    alerts = client.get("/api/jobs/alerts")
    assert alerts.status_code == 200
    assert "items" in alerts.json()

    align = client.post("/api/schemas/align", params={"limit": 20})
    assert align.status_code == 200
    assert "papers_processed" in align.json()

    ontology = client.get("/api/schemas/ontology", params={"min_support": 1})
    assert ontology.status_code == 200
    assert "nodes" in ontology.json()

    provenance = client.get("/api/metrics/provenance")
    assert provenance.status_code == 200
    assert "items" in provenance.json()
