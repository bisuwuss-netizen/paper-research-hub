from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import fitz  # PyMuPDF

try:
    import pymupdf4llm  # type: ignore
except Exception:
    pymupdf4llm = None

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    import gmft  # type: ignore
except Exception:
    gmft = None


@dataclass
class ExtractedMetadata:
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    journal_conf: Optional[str] = None


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _guess_title(lines: list[str]) -> Optional[str]:
    candidates = []
    for line in lines[:30]:
        clean = _clean_line(line)
        if not clean:
            continue
        if len(clean) < 10 or len(clean) > 200:
            continue
        lower = clean.lower()
        if "abstract" in lower:
            continue
        if "@" in clean or "http" in lower or "www" in lower:
            continue
        if re.search(r"\b(university|institute|laboratory|department|school|college)\b", lower):
            continue
        candidates.append(clean)
    if not candidates:
        return None
    # Prefer the longest reasonable line as title
    return max(candidates, key=len)


def _guess_title_by_font(pdf_path: str) -> Optional[str]:
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return None
            page = doc.load_page(0)
            info = page.get_text("dict")
    except Exception:
        return None

    lines = []
    max_size = 0.0
    for block in info.get("blocks", []):
        for line in block.get("lines", []):
            text_parts = []
            sizes = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    text_parts.append(text)
                sizes.append(span.get("size", 0))
            if not text_parts or not sizes:
                continue
            line_text = _clean_line("".join(text_parts))
            if not line_text:
                continue
            size = max(sizes)
            max_size = max(max_size, size)
            lines.append((line.get("bbox", [0, 0, 0, 0])[1], size, line_text))

    if not lines or max_size == 0:
        return None

    threshold = max_size - 0.5
    top_lines = [(y, t) for (y, size, t) in lines if size >= threshold]
    if not top_lines:
        return None
    top_lines.sort(key=lambda x: x[0])
    title = _clean_line(" ".join([t for _, t in top_lines]))
    if 10 <= len(title) <= 200:
        return title
    return None


def _guess_authors(lines: list[str], title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    try:
        title_index = next(i for i, l in enumerate(lines) if _clean_line(l) == title)
    except StopIteration:
        title_index = 0
    # Authors usually appear within the next 1-3 lines
    for i in range(title_index + 1, min(title_index + 5, len(lines))):
        clean = _clean_line(lines[i])
        if not clean:
            continue
        if "abstract" in clean.lower():
            return None
        if re.search(r"\b(university|institute|lab|department)\b", clean.lower()):
            continue
        # Heuristic: if commas or "and" appear, likely authors
        if "," in clean or " and " in clean.lower():
            return clean
        # If short and capitalized, also plausible
        if len(clean.split()) <= 8:
            return clean
    return None


def _guess_year(text: str) -> Optional[int]:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not years:
        return None
    # pick the most recent year in range
    years_int = [int(y) for y in years]
    years_int = [y for y in years_int if 1980 <= y <= 2035]
    if not years_int:
        return None
    return max(years_int)


def _guess_abstract(text: str) -> Optional[str]:
    match = re.search(r"\babstract\b\s*[:\n]", text, re.IGNORECASE)
    if not match:
        return None
    start = match.end()
    snippet = text[start:]
    # stop at common section headers
    stop_match = re.search(r"\n\s*(1\s+introduction|introduction|keywords)\b", snippet, re.IGNORECASE)
    if stop_match:
        snippet = snippet[: stop_match.start()]
    abstract = re.sub(r"\s+", " ", snippet).strip()
    if len(abstract) < 50:
        return None
    return abstract[:2000]


def _guess_doi(text: str) -> Optional[str]:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def _guess_journal(lines: list[str]) -> Optional[str]:
    for line in lines[:50]:
        clean = _clean_line(line)
        if not clean:
            continue
        if re.search(r"\b(Proceedings|Conference|Journal|Transactions)\b", clean, re.IGNORECASE):
            return clean
    return None


def extract_text(pdf_path: str, max_pages: int = 2) -> str:
    with fitz.open(pdf_path) as doc:
        texts: list[str] = []
        for i in range(min(max_pages, doc.page_count)):
            texts.append(doc.load_page(i).get_text("text"))
    return "\n".join(texts)


def extract_full_text(pdf_path: str, max_pages: Optional[int] = None) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            texts: list[str] = []
            page_cap = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
            for i in range(page_cap):
                texts.append(doc.load_page(i).get_text("text"))
    except Exception:
        return ""
    return "\n".join(texts)


def extract_markdown(pdf_path: str, max_pages: Optional[int] = None) -> str:
    if pymupdf4llm is None:
        return ""
    try:
        if max_pages is not None:
            pages = list(range(max_pages))
            output = pymupdf4llm.to_markdown(pdf_path, pages=pages)
        else:
            output = pymupdf4llm.to_markdown(pdf_path)
    except Exception:
        return ""
    if not isinstance(output, str):
        return ""
    return output


def extract_tables_text(pdf_path: str, max_pages: Optional[int] = None) -> str:
    if pdfplumber is None:
        return ""
    rows: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            for i in range(page_count):
                page = pdf.pages[i]
                for table in page.extract_tables() or []:
                    for row in table or []:
                        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
                        line = " | ".join([cell for cell in cells if cell])
                        if line:
                            rows.append(line)
    except Exception:
        return ""
    return "\n".join(rows)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 160) -> List[Dict[str, Any]]:
    content = re.sub(r"\s+", " ", text or "").strip()
    if not content:
        return []
    if chunk_size <= overlap:
        chunk_size = overlap + 120
    chunks: List[Dict[str, Any]] = []
    start = 0
    index = 0
    length = len(content)
    while start < length:
        end = min(length, start + chunk_size)
        # Try to cut on sentence boundary.
        if end < length:
            pivot = max(
                content.rfind("。", start, end),
                content.rfind(".", start, end),
                content.rfind("!", start, end),
                content.rfind("?", start, end),
            )
            if pivot > start + 120:
                end = pivot + 1
        chunk = content[start:end].strip()
        if chunk:
            chunks.append({"index": index, "text": chunk})
            index += 1
        if end >= length:
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_metadata(pdf_path: str, max_pages: int = 2) -> ExtractedMetadata:
    text = extract_text(pdf_path, max_pages=max_pages)
    lines = [line for line in text.splitlines() if _clean_line(line)]

    title = _guess_title_by_font(pdf_path) or _guess_title(lines)
    authors = _guess_authors(lines, title)
    year = _guess_year(text)
    abstract = _guess_abstract(text)
    doi = _guess_doi(text)
    journal_conf = _guess_journal(lines)

    return ExtractedMetadata(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        journal_conf=journal_conf,
    )


def extract_references_text(pdf_path: str, tail_pages: int = 4) -> Optional[str]:
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return None
            start = max(0, doc.page_count - tail_pages)
            pages = []
            for i in range(start, doc.page_count):
                pages.append(doc.load_page(i).get_text("text"))
    except Exception:
        return None
    text = "\n".join(pages)
    if not text:
        return None
    lower = text.lower()
    markers = ["references", "bibliography", "参考文献"]
    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            return text[idx:]
    if len(text) < 400:
        return None
    return text


def _to_metric(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    if num > 100:
        return None
    if 0 < num <= 1:
        return round(num * 100, 3)
    return round(num, 3)


def extract_ee_metrics(text: str, limit: int = 40) -> List[Dict[str, Any]]:
    if not text:
        return []
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    dataset_aliases = [
        "ACE2005",
        "ACE 2005",
        "ACE05",
        "MAVEN",
        "RAMS",
        "WikiEvents",
        "RichERE",
        "ERE",
        "CASIE",
        "FewEvent",
    ]
    out: List[Dict[str, Any]] = []
    seen = set()
    for line in lines:
        low = line.lower()
        dataset = None
        for alias in dataset_aliases:
            if alias.lower() in low:
                dataset = alias.replace(" ", "")
                break
        if not dataset:
            continue

        # P/R/F1 pattern
        prf = re.search(
            r"\bP(?:recision)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b.*?\bR(?:ecall)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b.*?\bF1?\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
            line,
            re.IGNORECASE,
        )
        if prf:
            precision = _to_metric(prf.group(1))
            recall = _to_metric(prf.group(2))
            f1 = _to_metric(prf.group(3))
            key = (dataset, precision, recall, f1, None, None)
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "dataset_name": dataset,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "trigger_f1": None,
                        "argument_f1": None,
                        "source": line[:280],
                        "confidence": 0.8,
                    }
                )
            if len(out) >= limit:
                break

        # Trigger/Argument F1 pattern
        trig = re.search(r"trigger\s*f1\s*[:=]?\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
        arg = re.search(r"argument\s*f1\s*[:=]?\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
        if trig or arg:
            trigger_f1 = _to_metric(trig.group(1)) if trig else None
            argument_f1 = _to_metric(arg.group(1)) if arg else None
            key = (dataset, None, None, None, trigger_f1, argument_f1)
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "dataset_name": dataset,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "trigger_f1": trigger_f1,
                        "argument_f1": argument_f1,
                        "source": line[:280],
                        "confidence": 0.72,
                    }
                )
            if len(out) >= limit:
                break
    return out


def _normalize_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _to_metric_from_cell(value: Any) -> Optional[float]:
    text = _normalize_cell(value)
    if not text:
        return None
    matched = re.search(r"(\d+(?:\.\d+)?)", text)
    if not matched:
        return None
    return _to_metric(matched.group(1))


def _is_dataset_cell(text: str) -> bool:
    low = text.lower()
    return any(
        key in low
        for key in [
            "ace2005",
            "ace 2005",
            "ace05",
            "maven",
            "rams",
            "wikievents",
            "richere",
            "ere",
            "casie",
            "fewevent",
        ]
    )


def _guess_metric_columns(header: List[str]) -> Dict[str, int]:
    columns: Dict[str, int] = {}
    for idx, raw in enumerate(header):
        token = raw.lower()
        if "dataset" in token or "corpus" in token or token in {"data", "bench"}:
            columns.setdefault("dataset_name", idx)
        if token in {"p", "prec", "precision"} or "precision" in token:
            columns.setdefault("precision", idx)
        if token in {"r", "recall"} or "recall" in token:
            columns.setdefault("recall", idx)
        if token in {"f1", "f1-score", "f"} or "f1" in token:
            columns.setdefault("f1", idx)
        if "trigger" in token and "f1" in token:
            columns.setdefault("trigger_f1", idx)
        if "argument" in token and "f1" in token:
            columns.setdefault("argument_f1", idx)
    return columns


def _iter_pdf_tables(pdf_path: str, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
                for page_index in range(page_count):
                    page = pdf.pages[page_index]
                    raw_tables = page.extract_tables() or []
                    for table_index, table in enumerate(raw_tables):
                        if not table:
                            continue
                        rows = [
                            [_normalize_cell(cell) for cell in row]
                            for row in table
                            if row and any(_normalize_cell(cell) for cell in row)
                        ]
                        if not rows:
                            continue
                        tables.append(
                            {
                                "parser": "pdfplumber",
                                "table_id": f"p{page_index+1}_t{table_index+1}",
                                "page": page_index + 1,
                                "rows": rows,
                            }
                        )
        except Exception:
            pass

    # Optional gmft enhancement, used when available.
    if gmft is not None:
        try:
            if hasattr(gmft, "extract_tables"):
                gmft_tables = gmft.extract_tables(pdf_path)  # type: ignore[attr-defined]
            elif hasattr(gmft, "parse"):
                gmft_tables = gmft.parse(pdf_path)  # type: ignore[attr-defined]
            else:
                gmft_tables = []
            for idx, table in enumerate(gmft_tables or []):
                rows = getattr(table, "rows", None) or table.get("rows") if isinstance(table, dict) else None
                if not rows:
                    continue
                cleaned = [
                    [_normalize_cell(cell) for cell in row]
                    for row in rows
                    if row and any(_normalize_cell(cell) for cell in row)
                ]
                if not cleaned:
                    continue
                page = getattr(table, "page", None) or (table.get("page") if isinstance(table, dict) else None)
                tables.append(
                    {
                        "parser": "gmft",
                        "table_id": f"gmft_{idx+1}",
                        "page": int(page) if page else None,
                        "rows": cleaned,
                    }
                )
        except Exception:
            # keep pdfplumber outputs if gmft parsing fails
            pass
    return tables


def extract_structured_table_metrics(
    pdf_path: str,
    max_pages: Optional[int] = None,
    limit: int = 80,
) -> Dict[str, Any]:
    tables = _iter_pdf_tables(pdf_path, max_pages=max_pages)
    metrics: List[Dict[str, Any]] = []
    cells: List[Dict[str, Any]] = []
    seen = set()

    for table in tables:
        rows = table.get("rows") or []
        if len(rows) < 2:
            continue
        header = [_normalize_cell(cell) for cell in rows[0]]
        metric_cols = _guess_metric_columns(header)
        if "dataset_name" not in metric_cols:
            # fallback: detect dataset directly in row body
            metric_cols["dataset_name"] = 0

        for row_idx, row in enumerate(rows[1:], start=1):
            if not row:
                continue
            dataset_idx = metric_cols.get("dataset_name", 0)
            dataset_raw = _normalize_cell(row[dataset_idx] if dataset_idx < len(row) else "")
            if not _is_dataset_cell(dataset_raw):
                # locate dataset in any cell as fallback
                dataset_raw = ""
                for cell in row:
                    candidate = _normalize_cell(cell)
                    if _is_dataset_cell(candidate):
                        dataset_raw = candidate
                        break
            if not dataset_raw:
                continue

            metric_item: Dict[str, Any] = {
                "dataset_name": dataset_raw.replace(" ", ""),
                "precision": None,
                "recall": None,
                "f1": None,
                "trigger_f1": None,
                "argument_f1": None,
                "source": f"{table.get('table_id')} row {row_idx}",
                "confidence": 0.86 if table.get("parser") == "gmft" else 0.78,
                "table_id": table.get("table_id"),
                "row_index": row_idx,
                "col_index": None,
                "cell_text": None,
                "provenance": {
                    "parser": table.get("parser"),
                    "page": table.get("page"),
                    "table_id": table.get("table_id"),
                    "header": header,
                    "cells": [],
                },
            }

            for key in ["precision", "recall", "f1", "trigger_f1", "argument_f1"]:
                col_idx = metric_cols.get(key)
                if col_idx is None or col_idx >= len(row):
                    continue
                cell_val = row[col_idx]
                parsed = _to_metric_from_cell(cell_val)
                if parsed is None:
                    continue
                metric_item[key] = parsed
                metric_item["col_index"] = col_idx
                metric_item["cell_text"] = _normalize_cell(cell_val)
                metric_item["provenance"]["cells"].append(
                    {"metric": key, "row": row_idx, "col": col_idx, "text": _normalize_cell(cell_val)}
                )
                cells.append(
                    {
                        "metric_key": key,
                        "metric_value": parsed,
                        "dataset_name": metric_item["dataset_name"],
                        "table_id": table.get("table_id"),
                        "row_index": row_idx,
                        "col_index": col_idx,
                        "cell_text": _normalize_cell(cell_val),
                        "parser": table.get("parser"),
                        "confidence": metric_item["confidence"],
                    }
                )

            if all(metric_item.get(k) is None for k in ["precision", "recall", "f1", "trigger_f1", "argument_f1"]):
                continue

            signature = (
                metric_item["dataset_name"],
                metric_item.get("precision"),
                metric_item.get("recall"),
                metric_item.get("f1"),
                metric_item.get("trigger_f1"),
                metric_item.get("argument_f1"),
            )
            if signature in seen:
                continue
            seen.add(signature)
            metrics.append(metric_item)
            if len(metrics) >= limit:
                break
        if len(metrics) >= limit:
            break

    return {"metrics": metrics, "cells": cells, "tables": len(tables)}


def extract_citation_context_map(
    full_text: str,
    references: List[Dict[str, Any]],
    max_context_per_ref: int = 1,
) -> Dict[str, str]:
    text = full_text or ""
    if not text or not references:
        return {}
    body = text
    marker = re.search(r"\n\s*(references|bibliography|参考文献)\b", text, flags=re.IGNORECASE)
    if marker:
        body = text[: marker.start()]
    body_low = body.lower()

    def sentence_window(idx: int) -> str:
        left = max(body.rfind(".", 0, idx), body.rfind("。", 0, idx), body.rfind("!", 0, idx), body.rfind("?", 0, idx))
        right_candidates = [body.find(".", idx), body.find("。", idx), body.find("!", idx), body.find("?", idx)]
        right_candidates = [x for x in right_candidates if x >= 0]
        right = min(right_candidates) if right_candidates else min(len(body), idx + 260)
        start = max(0, left + 1)
        end = min(len(body), right + 1)
        return _normalize_cell(body[start:end])[:360]

    mapping: Dict[str, str] = {}
    for ref in references:
        title = _normalize_cell(ref.get("title"))
        doi = _normalize_cell(ref.get("doi"))
        targets: List[Tuple[str, str]] = []
        if title:
            targets.append(("title", title))
        if doi:
            targets.append(("doi", doi))
        if not targets:
            continue
        contexts: List[str] = []
        for mode, token in targets:
            needle = token.lower()
            if mode == "title":
                title_tokens = [x for x in re.findall(r"[a-z0-9]{4,}", needle) if x not in {"using", "model", "method"}]
                if not title_tokens:
                    continue
                anchors = title_tokens[:3]
                first_hit = -1
                for anchor in anchors:
                    first_hit = body_low.find(anchor)
                    if first_hit >= 0:
                        break
                if first_hit < 0:
                    continue
                contexts.append(sentence_window(first_hit))
            else:
                hit = body_low.find(needle.lower())
                if hit >= 0:
                    contexts.append(sentence_window(hit))
            if len(contexts) >= max_context_per_ref:
                break
        if contexts:
            key = (title or doi).lower()
            mapping[key] = contexts[0]
    return mapping
