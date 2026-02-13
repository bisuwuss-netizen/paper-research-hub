from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Tuple


def build_foundation_path(papers: List[Dict], edges: List[Tuple[int, int]]) -> List[int]:
    try:
        import networkx as nx
    except Exception:
        return []

    G = nx.DiGraph()
    for p in papers:
        G.add_node(p["id"])
    G.add_edges_from(edges)

    if G.number_of_nodes() == 0:
        return []

    try:
        pr = nx.pagerank(G)
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [n for n, _ in ranked]

    try:
        topo = list(nx.topological_sort(G))
        path = [n for n in topo if n in set(ranked_ids)][:15]
        if path:
            return path
    except Exception:
        pass

    return ranked_ids[:15]


def build_sota_list(papers: List[Dict]) -> List[int]:
    current_year = datetime.now().year
    candidates = []
    for p in papers:
        year = p.get("year")
        if not year:
            continue
        if year < current_year - 3:
            continue
        velocity = p.get("citation_velocity")
        if velocity is None:
            citation_count = p.get("citation_count") or 0
            age = max(1, current_year - year + 1)
            score = citation_count / age
        else:
            score = float(velocity)
        candidates.append((p["id"], score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in candidates[:15]]


def build_clusters(papers: List[Dict]) -> Dict[int, str]:
    texts = []
    ids = []
    for p in papers:
        text = p.get("keywords") or p.get("abstract") or p.get("title")
        if not text:
            continue
        texts.append(text)
        ids.append(p["id"])

    if len(texts) < 4:
        # fallback: use sub_field
        mapping = {}
        for p in papers:
            label = p.get("sub_field") or "Cluster 1"
            mapping[p["id"]] = label
        return mapping

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
    except Exception:
        return {p["id"]: (p.get("sub_field") or "Cluster 1") for p in papers}

    n_clusters = max(2, min(6, int(math.sqrt(len(texts)))))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    mapping = {}
    for pid, label in zip(ids, labels):
        mapping[pid] = f"Cluster {label + 1}"
    return mapping
