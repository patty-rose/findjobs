"""Fast keyword-based job scoring — no LLM required.

Scores jobs 1-10 based on:
- Primary skill matches in the job description (worth more)
- Secondary skill matches in the job description (worth less)
- Location bonus for Portland/Oregon roles

Skills are loaded from the user's profile.json. Score breakdown:
  Primary skills:   up to 6 points (1.5 pts each, capped)
  Secondary skills: up to 2 points (0.5 pts each, capped)
  Location bonus:   +2 for Portland/Oregon, +1 for Remote
  Floor:            1, ceiling: 10
"""

import logging
import time
from datetime import datetime, timezone

from applypilot.config import load_profile
from applypilot.database import get_connection, get_jobs_by_stage

log = logging.getLogger(__name__)


def _load_skill_lists(profile: dict) -> tuple[list[str], list[str]]:
    """Split profile skills into primary and secondary keyword lists."""
    skills = profile.get("skills_boundary", {})
    langs = skills.get("programming_languages", [])
    frameworks = skills.get("frameworks", [])
    tools = skills.get("tools", [])

    # Primary: languages + core frameworks most likely to appear in job postings
    primary = langs + frameworks

    # Secondary: tools and platforms
    secondary = tools

    # Normalize to lowercase for matching
    primary = [s.lower() for s in primary if s]
    secondary = [s.lower() for s in secondary if s]

    return primary, secondary


def _location_bonus(location: str | None, boost_terms: list[str]) -> int:
    """Return location bonus points.

    Remote jobs get +1. Jobs matching user-configured boost locations get +2.
    Boost terms are loaded from searches.yaml (location_boost or location_accept
    non-remote entries) so no personal info is hardcoded here.
    """
    if not location:
        return 0
    loc = location.lower()
    if any(t in loc for t in ("remote", "anywhere", "distributed", "work from home")):
        return 1
    if boost_terms and any(t.lower() in loc for t in boost_terms):
        return 2
    return 0


def _score_job(desc: str, location: str | None,
               primary: list[str], secondary: list[str],
               boost_terms: list[str] | None = None) -> tuple[int, str]:
    """Score a single job. Returns (score, reasoning)."""
    if not desc:
        return 1, "No description available"

    text = desc.lower()

    primary_hits = [kw for kw in primary if kw in text]
    secondary_hits = [kw for kw in secondary if kw in text]

    primary_pts = min(6.0, len(primary_hits) * 1.5)
    secondary_pts = min(2.0, len(secondary_hits) * 0.5)
    loc_bonus = _location_bonus(location, boost_terms or [])

    raw = primary_pts + secondary_pts + loc_bonus
    score = max(1, min(10, round(raw)))

    parts = []
    if primary_hits:
        parts.append(f"primary: {', '.join(primary_hits)}")
    if secondary_hits:
        parts.append(f"secondary: {', '.join(secondary_hits)}")
    if loc_bonus:
        parts.append(f"location +{loc_bonus}")
    reasoning = "; ".join(parts) if parts else "no skill matches"

    return score, reasoning


def run_keyword_scoring(rescore: bool = False) -> dict:
    """Score jobs using keyword matching against profile skills.

    Fast alternative to LLM scoring — runs in seconds, no API calls.

    Args:
        rescore: If True, re-score all jobs. Otherwise only unscored ones.
    """
    profile = load_profile()
    if not profile:
        log.error("No profile found. Run 'applypilot init' first.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0}

    primary, secondary = _load_skill_lists(profile)
    log.info("Primary skills (%d): %s", len(primary), ", ".join(primary))
    log.info("Secondary skills (%d): %s", len(secondary), ", ".join(secondary))

    from applypilot.config import load_search_config
    search_cfg = load_search_config() or {}
    # Boost terms: explicit location_boost list, or fall back to non-remote location_accept entries
    boost_terms = search_cfg.get("location_boost") or [
        t for t in search_cfg.get("location_accept", [])
        if t not in ("remote", "anywhere", "distributed")
    ]

    conn = get_connection()

    if rescore:
        rows = conn.execute("SELECT * FROM jobs").fetchall()
    else:
        rows = get_jobs_by_stage(conn=conn, stage="pending_score")

    if not rows:
        log.info("No unscored jobs found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0}

    if rows and not isinstance(rows[0], dict):
        columns = rows[0].keys()
        rows = [dict(zip(columns, row)) for row in rows]

    log.info("Keyword-scoring %d jobs...", len(rows))
    t0 = time.time()
    now = datetime.now(timezone.utc).isoformat()

    dist = {}
    for job in rows:
        desc = job.get("full_description") or ""
        location = job.get("location") or ""
        score, reasoning = _score_job(desc, location, primary, secondary, boost_terms)

        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (score, reasoning, now, job["url"]),
        )
        dist[score] = dist.get(score, 0) + 1

    conn.commit()
    elapsed = time.time() - t0

    log.info("Done: %d jobs scored in %.1fs", len(rows), elapsed)
    for score in sorted(dist.keys(), reverse=True):
        log.info("  score %d: %d jobs", score, dist[score])

    return {
        "scored": len(rows),
        "errors": 0,
        "elapsed": elapsed,
        "distribution": sorted(dist.items(), reverse=True),
    }
