from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

BASE_URL = "https://api.semanticscholar.org/graph/v1"

DEFAULT_FIELDS = [
    "paperId",
    "corpusId",
    "title",
    "authors",
    "year",
    "venue",
    "abstract",
    "doi",
    "citationCount",
    "citationVelocity",
    "referenceCount",
    "influentialCitationCount",
    "url",
    "s2FieldsOfStudy",
    "externalIds",
]


def _headers() -> Dict[str, str]:
    api_key = os.getenv("S2_API_KEY", "").strip()
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def search_match(query: str, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    params = {
        "query": query,
        "fields": ",".join(fields or DEFAULT_FIELDS),
    }
    url = f"{BASE_URL}/paper/search/match"
    try:
        with httpx.Client(timeout=20) as client:
            res = client.get(url, params=params, headers=_headers())
            res.raise_for_status()
            data = res.json()
    except Exception:
        return None
    return data.get("data")


def get_paper(paper_id: str, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    if not paper_id:
        return None
    params = {
        "fields": ",".join(fields or DEFAULT_FIELDS),
    }
    url = f"{BASE_URL}/paper/{paper_id}"
    try:
        with httpx.Client(timeout=20) as client:
            res = client.get(url, params=params, headers=_headers())
            res.raise_for_status()
            return res.json()
    except Exception:
        return None


def get_references(paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    params = {
        "fields": ",".join(DEFAULT_FIELDS),
        "limit": limit,
    }
    url = f"{BASE_URL}/paper/{paper_id}/references"
    try:
        with httpx.Client(timeout=30) as client:
            res = client.get(url, params=params, headers=_headers())
            res.raise_for_status()
            data = res.json()
            return data.get("data", [])
    except Exception:
        return []


def get_citations(paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    params = {
        "fields": ",".join(DEFAULT_FIELDS),
        "limit": limit,
    }
    url = f"{BASE_URL}/paper/{paper_id}/citations"
    try:
        with httpx.Client(timeout=30) as client:
            res = client.get(url, params=params, headers=_headers())
            res.raise_for_status()
            data = res.json()
            return data.get("data", [])
    except Exception:
        return []


def extract_paper_from_edge(edge: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    item = edge.get(key)
    if not item:
        return None
    return item


def normalize_authors(authors: Any) -> Optional[str]:
    if not authors:
        return None
    if isinstance(authors, list):
        names = [a.get("name") for a in authors if isinstance(a, dict)]
        names = [n for n in names if n]
        return ", ".join(names) if names else None
    if isinstance(authors, str):
        return authors
    return None


def extract_year(data: Dict[str, Any]) -> Optional[int]:
    year = data.get("year")
    if isinstance(year, int):
        return year
    return None


def extract_doi(data: Dict[str, Any]) -> Optional[str]:
    doi = data.get("doi")
    if doi:
        return doi
    external = data.get("externalIds") or {}
    return external.get("DOI")


def extract_keywords(data: Dict[str, Any]) -> Optional[str]:
    fields = data.get("s2FieldsOfStudy")
    if not fields:
        return None
    if isinstance(fields, list):
        names = []
        for f in fields:
            if isinstance(f, dict) and f.get("category"):
                names.append(f.get("category"))
            elif isinstance(f, str):
                names.append(f)
        if names:
            return ",".join(sorted(set(names)))
    return None


def to_paper_record(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": data.get("title"),
        "authors": normalize_authors(data.get("authors")),
        "year": extract_year(data),
        "journal_conf": data.get("venue"),
        "venue": data.get("venue"),
        "doi": extract_doi(data),
        "abstract": data.get("abstract"),
        "s2_paper_id": data.get("paperId"),
        "s2_corpus_id": str(data.get("corpusId")) if data.get("corpusId") else None,
        "citation_count": data.get("citationCount"),
        "citation_velocity": data.get("citationVelocity"),
        "reference_count": data.get("referenceCount"),
        "influential_citation_count": data.get("influentialCitationCount"),
        "url": data.get("url"),
        "keywords": extract_keywords(data),
    }
