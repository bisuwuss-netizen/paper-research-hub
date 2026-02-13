from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF


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
