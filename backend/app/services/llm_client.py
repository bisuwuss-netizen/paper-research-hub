from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List

import httpx
from pydantic import BaseModel, ValidationError


SUB_FIELDS = [
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


class MetadataSchema(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    journal_conf: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    sub_field: Optional[str] = None


class ReferenceSchema(BaseModel):
    title: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None


def _build_prompt(
    raw_text: str, strict: bool, sub_fields: list[str], examples: Optional[List[Dict[str, str]]] = None
) -> str:
    fields = ", ".join(sub_fields)
    prefix = "Return ONLY valid JSON." if strict else "Return JSON."
    example_block = ""
    if examples:
        lines = ["Examples:"]
        for ex in examples[:6]:
            text = (ex.get("text") or "").strip().replace("\n", " ")
            label = ex.get("sub_field") or "null"
            if text:
                lines.append(f"TEXT: {text[:400]}")
                lines.append(f"SUB_FIELD: {label}")
        example_block = "\n".join(lines) + "\n\n"
    return (
        "You are extracting metadata from an academic paper. "
        f"{prefix} Keys: title, authors, year, journal_conf, doi, abstract, sub_field.\n"
        "If unsure, use null. For authors, return a comma-separated string.\n"
        f"Choose sub_field from: [{fields}]. If none match, return null.\n\n"
        f"{example_block}TEXT:\n{raw_text[:12000]}"
    )


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def _normalize(data: Dict[str, Any], sub_fields: list[str]) -> Dict[str, Any]:
    if not data:
        return {}
    out: Dict[str, Any] = {}
    for key in ["title", "authors", "journal_conf", "doi", "abstract", "sub_field"]:
        value = data.get(key)
        if isinstance(value, str):
            value = value.strip() or None
        out[key] = value
    year = data.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    if isinstance(year, (int, float)):
        year = int(year)
    else:
        year = None
    out["year"] = year
    if out.get("sub_field"):
        for option in sub_fields:
            if out["sub_field"].lower() == option.lower():
                out["sub_field"] = option
                break
    return out


def _build_payload(
    model: str,
    raw_text: str,
    strict: bool,
    sub_fields: list[str],
    examples: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a metadata extraction assistant."},
            {"role": "user", "content": _build_prompt(raw_text, strict, sub_fields, examples)},
        ],
        "temperature": 0.2,
    }
    if strict and os.getenv("LLM_RESPONSE_JSON", "").strip().lower() == "true":
        payload["response_format"] = {"type": "json_object"}
    return payload


def _resolve_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    if "/chat/completions" not in endpoint:
        return endpoint.rstrip("/") + "/chat/completions"
    return endpoint


def _call_llm(endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    endpoint = _resolve_endpoint(endpoint)
    with httpx.Client(timeout=40) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


def enrich_metadata(
    raw_text: str,
    sub_fields: list[str] | None = None,
    examples: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    endpoint = os.getenv("LLM_ENDPOINT", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not provider or not api_key or not endpoint or not model:
        return {}

    headers = {"Authorization": f"Bearer {api_key}"}
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

    last_valid: Dict[str, Any] = {}
    fields = sub_fields or SUB_FIELDS
    for attempt in range(max_retries + 1):
        strict = attempt > 0
        payload = _build_payload(model, raw_text, strict, fields, examples)
        try:
            content = _call_llm(endpoint, headers, payload)
        except Exception:
            # retry once without response_format for compatibility
            if payload.get("response_format"):
                payload.pop("response_format", None)
                try:
                    content = _call_llm(endpoint, headers, payload)
                except Exception:
                    continue
            else:
                continue

        data = _normalize(_parse_json(content), fields)
        if not data:
            continue

        try:
            validated = MetadataSchema.model_validate(data)
            last_valid = validated.model_dump()
            break
        except ValidationError:
            last_valid = data
            continue

    return last_valid


def summarize_one_line(title: Optional[str], abstract: Optional[str]) -> Optional[str]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    endpoint = os.getenv("LLM_ENDPOINT", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not provider or not api_key or not endpoint or not model:
        return None
    if not abstract and not title:
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        "Write ONE concise sentence summary (<= 30 words) of the paper in Chinese. "
        "If info is insufficient, return null.\n\n"
        f"TITLE: {title or ''}\nABSTRACT: {abstract or ''}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    try:
        content = _call_llm(endpoint, headers, payload).strip()
    except Exception:
        return None
    if not content or content.lower() in {"null", "none"}:
        return None
    # Try to strip quotes if returned as JSON string
    content = content.strip().strip("\"")
    return content[:300]


def extract_references(raw_text: str, max_items: int = 30) -> List[Dict[str, Any]]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    endpoint = os.getenv("LLM_ENDPOINT", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not provider or not api_key or not endpoint or not model:
        return []
    if not raw_text:
        return []
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        "Extract reference items from the following References section. "
        "Return ONLY JSON array. Each item: {\"title\":..., \"doi\":..., \"year\":...}. "
        f"Return at most {max_items} items. If unknown, use null.\n\n"
        f"TEXT:\n{raw_text[:12000]}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract citations from academic references."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    if os.getenv("LLM_RESPONSE_JSON", "").strip().lower() == "true":
        payload["response_format"] = {"type": "json_object"}
    try:
        content = _call_llm(endpoint, headers, payload)
    except Exception:
        if payload.get("response_format"):
            payload.pop("response_format", None)
            try:
                content = _call_llm(endpoint, headers, payload)
            except Exception:
                return []
        else:
            return []
    data = _parse_json(content)
    # If LLM returns wrapped object like {"items": [...]}
    if isinstance(data, dict):
        items = data.get("items") or data.get("references") or data.get("data")
    else:
        items = data
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        try:
            validated = ReferenceSchema.model_validate(item)
            out.append(validated.model_dump())
        except ValidationError:
            out.append({k: item.get(k) for k in ["title", "doi", "year"]})
    return out
