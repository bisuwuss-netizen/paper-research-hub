from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

import httpx

BASE_URL = "https://api.zotero.org"


def _library_path() -> Optional[str]:
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user").strip().lower()
    if library_type == "group":
        group_id = os.getenv("ZOTERO_GROUP_ID", "").strip()
        if not group_id:
            return None
        return f"/groups/{group_id}"
    user_id = os.getenv("ZOTERO_USER_ID", "").strip()
    if not user_id:
        return None
    return f"/users/{user_id}"


def _headers(version: Optional[str] = None) -> Dict[str, str]:
    api_key = os.getenv("ZOTERO_API_KEY", "").strip()
    if not api_key:
        return {}
    headers = {"Zotero-API-Key": api_key}
    if version:
        headers["If-Unmodified-Since-Version"] = str(version)
    return headers


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def search_item_by_title(title: str) -> Optional[Dict[str, Any]]:
    if not title:
        return None
    library_path = _library_path()
    if not library_path:
        return None

    params = {
        "q": title,
        "qmode": "titleCreatorYear",
        "limit": 5,
    }
    url = f"{BASE_URL}{library_path}/items"
    try:
        with httpx.Client(timeout=20) as client:
            res = client.get(url, params=params, headers=_headers())
            res.raise_for_status()
            items = res.json()
    except Exception:
        return None

    if not isinstance(items, list) or not items:
        return None

    normalized_title = _normalize(title)
    best = None
    for item in items:
        data = item.get("data") or {}
        item_title = data.get("title") or ""
        if _normalize(item_title) == normalized_title:
            best = item
            break
    return best or items[0]


def get_item(item_key: str) -> Optional[Dict[str, Any]]:
    library_path = _library_path()
    if not library_path or not item_key:
        return None
    url = f"{BASE_URL}{library_path}/items/{item_key}"
    try:
        with httpx.Client(timeout=20) as client:
            res = client.get(url, headers=_headers())
            res.raise_for_status()
            return res.json()
    except Exception:
        return None


def update_item(item_key: str, item: Dict[str, Any]) -> bool:
    library_path = _library_path()
    if not library_path or not item_key:
        return False
    version = item.get("version")
    url = f"{BASE_URL}{library_path}/items/{item_key}"
    try:
        with httpx.Client(timeout=20) as client:
            res = client.put(url, headers=_headers(version), json=item)
            res.raise_for_status()
            return True
    except Exception:
        return False


def build_zotero_url(item_key: str) -> Optional[str]:
    if not item_key:
        return None
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user").strip().lower()
    if library_type == "group":
        group_id = os.getenv("ZOTERO_GROUP_ID", "").strip()
        if not group_id:
            return None
        return f"zotero://select/groups/{group_id}/items/{item_key}"
    return f"zotero://select/library/items/{item_key}"


def _parse_creators(authors: Optional[str]) -> list[Dict[str, str]]:
    if not authors:
        return []
    creators = []
    for name in authors.split(","):
        name = name.strip()
        if not name:
            continue
        parts = name.split()
        if len(parts) == 1:
            creators.append({"creatorType": "author", "lastName": parts[0], "firstName": ""})
        else:
            creators.append({"creatorType": "author", "lastName": parts[-1], "firstName": " ".join(parts[:-1])})
    return creators


def apply_paper_updates(item: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    data = item.get("data", {})
    title = paper.get("title")
    if title and not data.get("title"):
        data["title"] = title
    creators = _parse_creators(paper.get("authors"))
    if creators and not data.get("creators"):
        data["creators"] = creators
    if paper.get("year") and not data.get("date"):
        data["date"] = str(paper.get("year"))
    if paper.get("doi") and not data.get("DOI"):
        data["DOI"] = paper.get("doi")
    if paper.get("url") and not data.get("url"):
        data["url"] = paper.get("url")

    venue = paper.get("venue") or paper.get("journal_conf")
    if venue:
        if data.get("itemType") == "journalArticle" and not data.get("publicationTitle"):
            data["publicationTitle"] = venue
        if data.get("itemType") == "conferencePaper" and not data.get("proceedingsTitle"):
            data["proceedingsTitle"] = venue

    item["data"] = data
    return item


def extract_library_info(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    library = item.get("library") or {}
    lib_type = library.get("type")
    lib_id = library.get("id")
    if lib_type == "group" and lib_id:
        return f"group:{lib_id}", item.get("data", {}).get("itemID")
    return "user", item.get("data", {}).get("itemID")
