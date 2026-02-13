from __future__ import annotations

import os
from typing import Optional

import httpx

BASE_URL = "https://api.crossref.org/works"


def lookup_doi(title: str) -> Optional[str]:
    if not title:
        return None
    params = {
        "query.title": title,
        "rows": 1,
    }
    mailto = os.getenv("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    try:
        with httpx.Client(timeout=20) as client:
            res = client.get(BASE_URL, params=params)
            res.raise_for_status()
            data = res.json()
    except Exception:
        return None

    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    doi = items[0].get("DOI")
    return doi
