from __future__ import annotations

from collections import defaultdict
import json
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

    def topic_of(p: Dict) -> str:
        sub = p.get("sub_field")
        if sub:
            return str(sub)
        tags = p.get("dynamic_tags")
        if isinstance(tags, str) and tags.strip():
            parsed = None
            try:
                parsed = json.loads(tags)
            except Exception:
                parsed = None
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
            if "," in tags:
                first = tags.split(",")[0].strip()
                if first:
                    return first
        keywords = p.get("keywords")
        if isinstance(keywords, str) and keywords.strip():
            first = keywords.split(",")[0].strip()
            if first:
                return first
        return "Unknown"

    for p in papers:
        year = p.get("year")
        if not year:
            continue
        sub = topic_of(p)
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
    all_years = sorted(by_year_subfield.keys())
    all_subfields = sorted(
        {
            topic_of(p)
            for p in papers
            if p.get("year")
        }
    )
    river = []
    for year in all_years:
        total = max(1, year_totals.get(year, 0))
        for sub in all_subfields:
            count = by_year_subfield[year].get(sub, 0)
            if count <= 0:
                continue
            river.append(
                {
                    "year": year,
                    "sub_field": sub,
                    "count": count,
                    "ratio": round(count / total, 4),
                }
            )

    return {
        "trends": trends,
        "hotspots": hotspots,
        "bursts": bursts[:20],
        "river": river,
        "years": all_years,
        "sub_fields": all_subfields,
    }


def build_progress_snapshot(papers: List[Dict]) -> Dict:
    total = len(papers)
    read = len([p for p in papers if (p.get("read_status") or 0) == 1])
    unread = max(0, total - read)
    uploaded = len([p for p in papers if p.get("file_path")])
    return {"total": total, "read": read, "unread": unread, "uploaded": uploaded}
