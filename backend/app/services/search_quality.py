from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")

ZH_EN_SYNONYMS = {
    "事件抽取": "event extraction",
    "论元抽取": "argument extraction",
    "触发词": "trigger",
    "小样本": "few-shot",
    "零样本": "zero-shot",
    "文档级": "document-level",
    "跨文档": "cross-document",
    "跨语言": "cross-lingual",
    "低资源": "low-resource",
    "提示学习": "prompt learning",
    "指令微调": "instruction tuning",
    "大模型": "llm",
    "关系抽取": "relation extraction",
}


def normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def tokenize(value: Optional[str]) -> List[str]:
    text = normalize_text(value)
    return TOKEN_RE.findall(text)


def expand_query_aliases(query: str) -> List[str]:
    base = normalize_text(query)
    if not base:
        return []
    items = [base]
    translated = base
    for zh, en in ZH_EN_SYNONYMS.items():
        if zh in translated:
            translated = translated.replace(zh, f"{zh} {en}")
    if translated != base:
        items.append(translated)
    for zh, en in ZH_EN_SYNONYMS.items():
        if zh in base:
            items.append(en)
    return list(dict.fromkeys([i.strip() for i in items if i.strip()]))


@dataclass
class SearchDocument:
    doc_id: int
    title: str
    abstract: str
    notes: str

    @property
    def text(self) -> str:
        return " ".join([self.title or "", self.abstract or "", self.notes or ""]).strip()


def bm25_scores(
    query: str,
    docs: Iterable[SearchDocument],
    k1: float = 1.6,
    b: float = 0.75,
) -> Dict[int, float]:
    docs_list = list(docs)
    q_terms = tokenize(query)
    if not docs_list or not q_terms:
        return {}

    doc_tokens: Dict[int, List[str]] = {d.doc_id: tokenize(d.text) for d in docs_list}
    doc_lens = {doc_id: len(tokens) for doc_id, tokens in doc_tokens.items()}
    avgdl = (sum(doc_lens.values()) / max(1, len(doc_lens))) or 1.0

    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        for t in set(tokens):
            df[t] += 1
    n_docs = len(doc_tokens)
    idf = {
        term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }

    scores: Dict[int, float] = {}
    for doc_id, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = max(1, doc_lens[doc_id])
        score = 0.0
        for term in q_terms:
            if term not in tf or term not in idf:
                continue
            freq = tf[term]
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * (numerator / denominator)
        if score > 0:
            scores[doc_id] = score
    return scores


def semantic_scores(query: str, docs: Iterable[SearchDocument]) -> Dict[int, float]:
    docs_list = list(docs)
    if not docs_list or not query.strip():
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return {}

    corpus = [d.text for d in docs_list]
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=6000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
    )
    scores: Dict[int, float] = {}
    for weight, vectorizer in [(0.65, word_vectorizer), (0.35, char_vectorizer)]:
        matrix = vectorizer.fit_transform(corpus + [query])
        doc_vectors = matrix[:-1]
        query_vector = matrix[-1]
        similarities = (doc_vectors @ query_vector.T).toarray().ravel().tolist()
        for idx, sim in enumerate(similarities):
            if sim <= 0:
                continue
            doc_id = docs_list[idx].doc_id
            scores[doc_id] = scores.get(doc_id, 0.0) + float(sim) * weight
    return scores


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {doc_id: value / max_score for doc_id, value in scores.items()}


def hybrid_search(
    query: str,
    docs: Iterable[SearchDocument],
    top_k: int = 20,
) -> List[Dict[str, float | int]]:
    docs_list = list(docs)
    expanded_queries = expand_query_aliases(query) or [query]
    bm25_raw: Dict[int, float] = {}
    semantic_raw: Dict[int, float] = {}
    for q in expanded_queries:
        for doc_id, val in bm25_scores(q, docs_list).items():
            bm25_raw[doc_id] = max(bm25_raw.get(doc_id, 0.0), val)
        for doc_id, val in semantic_scores(q, docs_list).items():
            semantic_raw[doc_id] = max(semantic_raw.get(doc_id, 0.0), val)

    bm25 = _normalize_scores(bm25_raw)
    semantic = _normalize_scores(semantic_raw)
    doc_ids = set(bm25.keys()) | set(semantic.keys())

    # Lexical fallback: allow direct substring hits even when vectorizers miss.
    norm_query = normalize_text(query)
    if norm_query:
        q_tokens = tokenize(norm_query)
        for d in docs_list:
            text = normalize_text(d.text)
            if not text:
                continue
            overlap = 0
            if norm_query in text:
                overlap = 1
            elif q_tokens:
                overlap = sum(1 for tok in q_tokens if tok in text) / len(q_tokens)
            if overlap > 0:
                doc_ids.add(d.doc_id)
                bm25[d.doc_id] = max(bm25.get(d.doc_id, 0.0), overlap * 0.65)
                semantic[d.doc_id] = max(semantic.get(d.doc_id, 0.0), overlap * 0.55)

    ranked: List[Dict[str, float]] = []
    for doc_id in doc_ids:
        bm = bm25.get(doc_id, 0.0)
        sm = semantic.get(doc_id, 0.0)
        score = 0.58 * bm + 0.42 * sm
        ranked.append(
            {
                "doc_id": doc_id,
                "score": score,
                "bm25_score": bm,
                "semantic_score": sm,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def title_similarity(a: Optional[str], b: Optional[str]) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def detect_duplicate_groups(
    papers: Iterable[Dict],
    title_threshold: float = 0.9,
) -> List[Dict]:
    rows = list(papers)
    doi_groups: Dict[str, List[Dict]] = {}
    for p in rows:
        doi = normalize_text(p.get("doi"))
        if doi:
            doi_groups.setdefault(doi, []).append(p)

    groups: List[Dict] = []
    for doi, items in doi_groups.items():
        if len(items) > 1:
            groups.append(
                {
                    "type": "doi_exact",
                    "key": doi,
                    "paper_ids": [i["id"] for i in items],
                    "confidence": 1.0,
                }
            )

    used_pairs = set()
    for i in range(len(rows)):
        pi = rows[i]
        ti = pi.get("title")
        if not ti:
            continue
        for j in range(i + 1, len(rows)):
            pj = rows[j]
            tj = pj.get("title")
            if not tj:
                continue
            pair_key = tuple(sorted([pi["id"], pj["id"]]))
            if pair_key in used_pairs:
                continue
            # constrain fuzzy checks to close publication years when available
            yi = pi.get("year")
            yj = pj.get("year")
            if yi and yj and abs(int(yi) - int(yj)) > 2:
                continue
            sim = title_similarity(ti, tj)
            if sim >= title_threshold:
                used_pairs.add(pair_key)
                groups.append(
                    {
                        "type": "title_fuzzy",
                        "key": f"{pi['id']}:{pj['id']}",
                        "paper_ids": [pi["id"], pj["id"]],
                        "confidence": round(sim, 4),
                    }
                )
    return groups


def detect_conflicts(papers: Iterable[Dict]) -> List[Dict]:
    rows = list(papers)
    by_doi: Dict[str, List[Dict]] = {}
    for p in rows:
        doi = normalize_text(p.get("doi"))
        if doi:
            by_doi.setdefault(doi, []).append(p)

    conflicts: List[Dict] = []
    for doi, items in by_doi.items():
        if len(items) < 2:
            continue
        titles = {normalize_text(i.get("title")) for i in items if i.get("title")}
        years = {i.get("year") for i in items if i.get("year")}
        if len(titles) > 1 or len(years) > 1:
            conflicts.append(
                {
                    "doi": doi,
                    "paper_ids": [i["id"] for i in items],
                    "title_variants": sorted({i.get("title") for i in items if i.get("title")}),
                    "year_variants": sorted({i.get("year") for i in items if i.get("year")}),
                }
            )
    return conflicts


def classify_edge_confidence(source: str, has_doi: bool, title_quality: float) -> float:
    base = 0.75
    source = (source or "").lower()
    if source == "semantic_scholar":
        base = 0.95
    elif source == "crossref":
        base = 0.85
    elif source == "llm_reference":
        base = 0.7
    if has_doi:
        base += 0.05
    base += min(0.08, max(0.0, title_quality * 0.08))
    return round(min(0.99, max(0.2, base)), 4)
