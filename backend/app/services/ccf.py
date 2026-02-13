from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
CCF_PATH = os.path.join(RESOURCES_DIR, "ccf_list.json")
CCF_ALIAS_PATH = os.path.join(RESOURCES_DIR, "ccf_aliases.json")


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


STOPWORDS = {
    "of",
    "the",
    "and",
    "for",
    "on",
    "in",
    "to",
    "a",
    "an",
    "international",
    "conference",
    "symposium",
    "workshop",
    "annual",
    "acm",
    "ieee",
    "sig",
    "transactions",
    "journal",
}


def _acronym(text: str) -> Optional[str]:
    words = re.findall(r"[A-Za-z0-9]+", text)
    letters = [w[0].upper() for w in words if w.lower() not in STOPWORDS]
    if len(letters) >= 2:
        return "".join(letters)
    return None


def _extract_candidates(venue: str) -> set[str]:
    candidates: set[str] = set()
    for m in re.findall(r"\(([A-Z0-9]{2,10})\)", venue):
        candidates.add(m)
    for m in re.findall(r"\b[A-Z]{2,10}\b", venue):
        candidates.add(m)
    ac = _acronym(venue)
    if ac:
        candidates.add(ac)
    return candidates


@lru_cache(maxsize=1)
def _load_ccf() -> dict:
    try:
        with open(CCF_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"A": [], "B": [], "C": []}


@lru_cache(maxsize=1)
def _load_aliases() -> dict:
    try:
        with open(CCF_ALIAS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return { _normalize(k): v for k, v in data.items() }
    except Exception:
        return {}


def learn_alias(alias: str, target: str) -> bool:
    if not alias or not target:
        return False
    try:
        with open(CCF_ALIAS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    if alias in data:
        return False
    data[alias] = target
    with open(CCF_ALIAS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _load_aliases.cache_clear()
    _build_index.cache_clear()
    return True


@lru_cache(maxsize=1)
def _build_index():
    data = _load_ccf()
    aliases = _load_aliases()
    canonical_level = {}
    index = {}
    for level in ["A", "B", "C"]:
        names = []
        abbrs = set()
        for name in data.get(level, []):
            if not name:
                continue
            norm = _normalize(name)
            if norm:
                names.append(norm)
                canonical_level[norm] = level
            raw = name.strip()
            if raw and " " not in raw and len(raw) <= 12:
                abbrs.add(raw.upper())
            if re.fullmatch(r"[A-Z0-9]{2,12}", raw):
                abbrs.add(raw.upper())
        index[level] = {"names": names, "abbrs": abbrs}

    # apply aliases to the detected level or direct level mapping
    for alias_norm, target in aliases.items():
        if not alias_norm or not target:
            continue
        if isinstance(target, str) and target.upper() in {"A", "B", "C"}:
            index[target.upper()]["names"].append(alias_norm)
            if re.fullmatch(r"[A-Z0-9]{2,12}", alias_norm.replace(" ", "").upper()):
                index[target.upper()]["abbrs"].add(alias_norm.replace(" ", "").upper())
            continue
        target_norm = _normalize(str(target))
        target_level = canonical_level.get(target_norm)
        if target_level:
            index[target_level]["names"].append(alias_norm)
            if re.fullmatch(r"[A-Z0-9]{2,12}", alias_norm.replace(" ", "").upper()):
                index[target_level]["abbrs"].add(alias_norm.replace(" ", "").upper())
    return index


def classify_venue(venue: Optional[str]) -> Optional[str]:
    if not venue:
        return None
    venue_norm = _normalize(venue)
    index = _build_index()
    for level in ["A", "B", "C"]:
        for name_norm in index[level]["names"]:
            if name_norm and (name_norm == venue_norm or name_norm in venue_norm or venue_norm in name_norm):
                return level
        candidates = _extract_candidates(venue)
        for cand in candidates:
            if cand.upper() in index[level]["abbrs"]:
                return level
    return None
