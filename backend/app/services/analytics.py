from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple


def to_period(period: str, now: date | None = None) -> Tuple[date, date]:
    today = now or datetime.now().date()
    if period == "monthly":
        start = today.replace(day=1)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1) - timedelta(days=1)
        else:
            end = start.replace(month=start.month + 1) - timedelta(days=1)
        return start, end
    # default weekly (Mon-Sun)
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start, end


def build_topic_evolution(papers: List[Dict]) -> Dict:
    by_year_subfield: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: Dict[int, int] = defaultdict(int)
    hot_papers_by_year: Dict[int, List[Dict]] = defaultdict(list)

    for p in papers:
        year = p.get("year")
        if not year:
            continue
        sub = p.get("sub_field") or "Unknown"
        by_year_subfield[int(year)][sub] += 1
        year_totals[int(year)] += 1

        velocity = p.get("citation_velocity")
        if velocity is None:
            c = p.get("citation_count") or 0
            age = max(1, datetime.now().year - int(year) + 1)
            velocity = c / age
        hot_papers_by_year[int(year)].append(
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "sub_field": sub,
                "velocity": float(velocity or 0),
                "citation_count": p.get("citation_count") or 0,
            }
        )

    trends = []
    for year in sorted(by_year_subfield.keys()):
        trends.append(
            {
                "year": year,
                "total": year_totals.get(year, 0),
                "sub_fields": by_year_subfield[year],
            }
        )

    hotspots = []
    for year, items in hot_papers_by_year.items():
        top = sorted(items, key=lambda x: x["velocity"], reverse=True)[:5]
        hotspots.append({"year": year, "papers": top})
    hotspots.sort(key=lambda x: x["year"])

    # burst detection: year-over-year surge per sub-field
    per_sub: Dict[str, Dict[int, int]] = defaultdict(dict)
    for year, sub_counts in by_year_subfield.items():
        for sub, cnt in sub_counts.items():
            per_sub[sub][year] = cnt

    bursts: List[Dict] = []
    for sub, year_counts in per_sub.items():
        years = sorted(year_counts.keys())
        if len(years) < 2:
            continue
        prev = None
        for y in years:
            curr = year_counts[y]
            if prev is None:
                prev = curr
                continue
            growth = curr - prev
            growth_ratio = (growth / prev) if prev > 0 else float(curr)
            if curr >= 3 and growth >= 2 and growth_ratio >= 0.5:
                bursts.append(
                    {
                        "sub_field": sub,
                        "year": y,
                        "count": curr,
                        "growth": growth,
                        "growth_ratio": round(growth_ratio, 3),
                    }
                )
            prev = curr

    bursts.sort(key=lambda x: (x["growth_ratio"], x["count"]), reverse=True)
    return {"trends": trends, "hotspots": hotspots, "bursts": bursts[:20]}


def build_progress_snapshot(papers: List[Dict]) -> Dict:
    total = len(papers)
    read = len([p for p in papers if (p.get("read_status") or 0) == 1])
    unread = max(0, total - read)
    uploaded = len([p for p in papers if p.get("file_path")])
    return {"total": total, "read": read, "unread": unread, "uploaded": uploaded}
