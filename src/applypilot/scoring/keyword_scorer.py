"""Fast keyword-based job scoring — no LLM required.

Scores jobs 1-10 based on tiered skill keyword matches, location, and
years-of-experience signals found in the job description.

Skill tiers and location config are loaded from searches.yaml so all
personal preferences stay in local config, never in the codebase.

Score breakdown:
  Tier 1 skills:  2.0 pts each (core stack)
  Tier 2 skills:  0.75 pts each (secondary stack)
  Tier 3 skills:  0.25 pts each (tools / weak skills)
  Frequency boost: sublinear — 1 mention=1.0x, 2=1.2x, 3+=1.35x per keyword
  Length norm:     skill_pts divided by log2(word_count+2) so long JDs
                   don't unfairly dominate; normalized to a 9-word baseline
  Location bonus: +2 local, +1 US remote
  Experience:     +1 if ≤3 yrs required, 0 if 4-5, -1 if 6+
  Culture:        ratio of matched/total terms, capped at 1.0
  Floor: 1, ceiling: 10
"""

import logging
import math
import re
import time
from datetime import datetime, timezone

from applypilot.config import load_profile, load_search_config
from applypilot.database import get_connection, get_jobs_by_stage

log = logging.getLogger(__name__)


# ── Skill loading ────────────────────────────────────────────────────────────

def _load_tiers(profile: dict, search_cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Load skill tiers from searches.yaml, falling back to profile categories.

    searches.yaml can define:
      skill_tiers:
        tier1: [python, django, ...]
        tier2: [javascript, node.js, ...]
        tier3: [docker, aws, ...]
    """
    tiers_cfg = search_cfg.get("skill_tiers", {})

    if tiers_cfg:
        tier1 = [s.lower() for s in tiers_cfg.get("tier1", [])]
        tier2 = [s.lower() for s in tiers_cfg.get("tier2", [])]
        tier3 = [s.lower() for s in tiers_cfg.get("tier3", [])]
    else:
        # Fallback: use profile categories as tiers
        skills = profile.get("skills_boundary", {})
        tier1 = [s.lower() for s in skills.get("programming_languages", [])]
        tier2 = [s.lower() for s in skills.get("frameworks", [])]
        tier3 = [s.lower() for s in skills.get("tools", [])]

    return tier1, tier2, tier3


# ── Location scoring ─────────────────────────────────────────────────────────

def _location_score(location: str | None, boost_terms: list[str],
                    reject_always: list[str]) -> int:
    """Return location bonus. Returns -99 to signal the job should be skipped."""
    if not location:
        return 1  # unknown — treat as remote-ish

    loc = location.lower()

    # Hard reject: international locations even if "remote" appears
    if reject_always and any(t.lower() in loc for t in reject_always):
        return -99  # signal to skip

    is_remote = any(t in loc for t in ("remote", "anywhere", "distributed", "work from home"))

    if is_remote:
        # Boost if it's a local remote (e.g. "Portland, OR (Remote)")
        if boost_terms and any(t.lower() in loc for t in boost_terms):
            return 2
        return 1  # generic US remote

    # In-person: boost if it's a preferred location
    if boost_terms and any(t.lower() in loc for t in boost_terms):
        return 2

    return 0


# ── Seniority scoring ────────────────────────────────────────────────────────

_INTERN_TERMS = ("intern", "internship", "co-op", "coop")
_JUNIOR_TERMS = ("junior", "entry level", "entry-level", "jr.", " i ", "level 1",
                 "l1", "swe 1", "swe i", "engineer i ", "engineer 1", " ii ", "level 2",
                 "l2", "swe 2", "swe ii", "mid level", "mid-level", "engineer ii")
_SENIOR_TERMS = ("senior", "sr ", "sr.", "lead ", "staff", "principal")

def _seniority_bonus(title: str, desc: str) -> float:
    """Penalty for intern/senior, slight penalty for junior (still worth applying)."""
    t = title.lower()
    if any(term in t for term in _INTERN_TERMS):
        return -1.5
    if any(term in t for term in _JUNIOR_TERMS):
        return -0.5
    if any(term in t for term in _SENIOR_TERMS):
        return -0.5
    return 0.0


# ── Remote-first bonus ───────────────────────────────────────────────────────

_REMOTE_FIRST_TERMS = ("remote-first", "remote first", "fully remote", "fully distributed",
                       "async-first", "async first", "distributed team", "remote only")

def _remote_first_bonus(desc: str) -> float:
    text = desc.lower()
    return 0.5 if any(t in text for t in _REMOTE_FIRST_TERMS) else 0.0


# ── Full stack bonus ─────────────────────────────────────────────────────────

def _fullstack_bonus(desc: str, tier1: list[str]) -> float:
    """Bonus if job mentions both backend and frontend tier1 skills."""
    text = desc.lower()
    backend = {"python", "django", "postgresql", "node.js", "express", "fast api", "fastapi"}
    frontend = {"react", "typescript", "javascript", "next.js"}
    has_backend = any(s in text for s in backend if s in tier1)
    has_frontend = any(s in text for s in frontend if s in tier1)
    return 1.0 if (has_backend and has_frontend) else 0.0


# ── Culture scoring ──────────────────────────────────────────────────────────

_DEFAULT_CULTURE_TERMS = (
    "collaborative", "collaboration", "mentorship", "mentoring", "mentor",
    "growth", "learning", "development opportunities", "career growth",
    "fun", "kind", "kindness", "empathy", "empathetic", "inclusive",
    "supportive", "psychological safety", "work-life balance", "work life balance",
    "team player", "people first", "human", "humble", "curiosity", "curious",
    "trust", "transparent", "transparency", "belonging",
)

def _culture_bonus(desc: str, culture_terms: list[str]) -> float:
    """Bonus for positive culture signals in the job description.

    Uses ratio of matched/total terms so long JDs don't get an unfair
    advantage over shorter ones. Capped at 1.0.
    """
    if not culture_terms:
        return 0.0
    text = desc.lower()
    hits = sum(1 for t in culture_terms if t in text)
    ratio = hits / len(culture_terms)
    # Scale: ratio=0.25 → ~1.0 pts (most jobs that mention culture hit ~25% of terms)
    return round(min(ratio * 4.0, 1.0), 2)


# ── Penalty signals ──────────────────────────────────────────────────────────

_CONTRACT_TERMS = ("contract role", "contract position", "1099", "freelance", "contract-to-hire",
                   "contract to hire", "independent contractor")
_CLEARANCE_TERMS = ("security clearance", "clearance required", "secret clearance",
                    "top secret", "ts/sci", "dod clearance", "active clearance")

def _penalty_bonus(title: str, desc: str) -> float:
    text = (title + " " + desc).lower()
    penalty = 0.0
    if any(t in text for t in _CLEARANCE_TERMS):
        penalty -= 2.0
    if any(t in text for t in _CONTRACT_TERMS):
        penalty -= 1.0
    return penalty


# ── Experience scoring ───────────────────────────────────────────────────────

_EXP_PATTERNS = [
    re.compile(r'(\d+)\+?\s*(?:to\s*\d+)?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience', re.I),
    re.compile(r'minimum\s+(?:of\s+)?(\d+)\s+years?', re.I),
    re.compile(r'at\s+least\s+(\d+)\s+years?', re.I),
    re.compile(r'(\d+)-\d+\s+years?\s+(?:of\s+)?experience', re.I),
]

def _experience_bonus(desc: str) -> tuple[int, int | None]:
    """Scan description for years-of-experience requirements.

    Returns (bonus, min_years_required). min_years is None if not mentioned.
    """
    years_found = []
    for pattern in _EXP_PATTERNS:
        for match in pattern.finditer(desc):
            try:
                years_found.append(int(match.group(1)))
            except (ValueError, IndexError):
                pass

    if not years_found:
        return 0, None

    min_years = min(years_found)
    if min_years <= 3:
        return 1, min_years
    elif min_years <= 5:
        return 0, min_years
    else:
        return -1, min_years


# ── Main scorer ──────────────────────────────────────────────────────────────

def _score_job(desc: str, location: str | None, title: str,
               tier1: list[str], tier2: list[str], tier3: list[str],
               boost_terms: list[str], reject_always: list[str],
               culture_terms: list[str] | None = None) -> tuple[int, str, int | None] | None:
    """Score a single job. Returns (score, reasoning, years_required) or None to skip."""
    loc_pts = _location_score(location, boost_terms, reject_always)
    if loc_pts == -99:
        return None  # skip this job

    # Hard filter: senior/lead/QA/test/intern titles are not a fit
    _TITLE_REJECT = (
        "senior", "sr ", "sr.", "lead ", "staff ", "principal", "director", "head of",
        "vp ", "vice president", "manager",
        "qa ", " qa", "qe ", " qe", "sdet", "quality assurance", "quality engineer",
        "test engineer", "software test", "testing engineer",
        "intern", "internship",
        "machine learning", "ml engineer", "data scientist", "data engineer",
        "android", "ios ", "mobile engineer", "mobile developer",
        "devops", "site reliability", " sre",
    )
    if any(t in title.lower() for t in _TITLE_REJECT):
        return 1, "filtered title", None

    if not desc:
        score = max(1, loc_pts)
        return score, "no description", None

    text = desc.lower()

    def _freq_weight(kw: str) -> float:
        """Sublinear frequency boost: 1 mention=1.0x, 2=1.2x, 3+=1.35x."""
        count = text.count(kw)
        if count == 0:
            return 0.0
        if count == 1:
            return 1.0
        if count == 2:
            return 1.2
        return 1.35

    t1_hits = [kw for kw in tier1 if kw in text]
    t2_hits = [kw for kw in tier2 if kw in text]
    t3_hits = [kw for kw in tier3 if kw in text]

    t1_pts = sum(_freq_weight(kw) * 2.0 for kw in t1_hits)
    t2_pts = sum(_freq_weight(kw) * 0.75 for kw in t2_hits)
    t3_pts = sum(_freq_weight(kw) * 0.25 for kw in t3_hits)
    raw_skill_pts = t1_pts + t2_pts + t3_pts

    # Length normalization: divide by log2(word_count+2), anchored to a
    # 300-word baseline so a typical JD scores roughly the same as before.
    word_count = len(text.split())
    length_norm = math.log2(300 + 2) / math.log2(word_count + 2) if word_count > 0 else 1.0
    skill_pts = raw_skill_pts * length_norm

    exp_pts, years_required = _experience_bonus(text)
    seniority_pts = _seniority_bonus(title, text)
    remote_pts = _remote_first_bonus(text)
    fullstack_pts = _fullstack_bonus(text, tier1)
    penalty_pts = _penalty_bonus(title, text)
    culture_pts = _culture_bonus(text, culture_terms or list(_DEFAULT_CULTURE_TERMS))

    raw = skill_pts + loc_pts + exp_pts + seniority_pts + remote_pts + fullstack_pts + penalty_pts + culture_pts
    score = max(1, min(10, round(raw)))

    parts = []
    if t1_hits:
        parts.append(f"tier1: {', '.join(t1_hits)}")
    if t2_hits:
        parts.append(f"tier2: {', '.join(t2_hits)}")
    if t3_hits:
        parts.append(f"tier3: {', '.join(t3_hits)}")
    if loc_pts:
        parts.append(f"loc +{loc_pts}")
    if exp_pts:
        parts.append(f"exp {exp_pts:+.1f}")
    if seniority_pts:
        parts.append(f"seniority {seniority_pts:+.1f}")
    if remote_pts:
        parts.append(f"remote-first +{remote_pts}")
    if fullstack_pts:
        parts.append(f"fullstack +{fullstack_pts}")
    if penalty_pts:
        parts.append(f"penalty {penalty_pts:+.1f}")
    if culture_pts:
        parts.append(f"culture +{culture_pts}")
    reasoning = "; ".join(parts) if parts else "no matches"

    return score, reasoning, years_required


# ── Entry point ──────────────────────────────────────────────────────────────

def run_keyword_scoring(rescore: bool = False) -> dict:
    """Score jobs using tiered keyword matching. Fast, no API calls."""
    profile = load_profile()
    if not profile:
        log.error("No profile found. Run 'applypilot init' first.")
        return {"scored": 0, "skipped": 0, "errors": 0, "elapsed": 0.0}

    search_cfg = load_search_config() or {}
    tier1, tier2, tier3 = _load_tiers(profile, search_cfg)

    boost_terms = search_cfg.get("location_boost") or [
        t for t in search_cfg.get("location_accept", [])
        if t not in ("remote", "anywhere", "distributed")
    ]
    reject_always = search_cfg.get("location_reject_always", [])
    culture_terms = search_cfg.get("culture_keywords") or list(_DEFAULT_CULTURE_TERMS)

    log.info("Tier 1 (%d): %s", len(tier1), ", ".join(tier1))
    log.info("Tier 2 (%d): %s", len(tier2), ", ".join(tier2))
    log.info("Tier 3 (%d): %s", len(tier3), ", ".join(tier3))

    conn = get_connection()

    if rescore:
        rows = conn.execute("SELECT * FROM jobs").fetchall()
    else:
        rows = get_jobs_by_stage(conn=conn, stage="pending_score", limit=0)

    if not rows:
        log.info("No unscored jobs found.")
        return {"scored": 0, "skipped": 0, "errors": 0, "elapsed": 0.0}

    if rows and not isinstance(rows[0], dict):
        columns = rows[0].keys()
        rows = [dict(zip(columns, row)) for row in rows]

    log.info("Keyword-scoring %d jobs...", len(rows))
    t0 = time.time()
    now = datetime.now(timezone.utc).isoformat()

    dist: dict[int, int] = {}
    skipped = 0

    for job in rows:
        desc = job.get("full_description") or ""
        location = job.get("location") or ""

        title = job.get("title") or ""
        result = _score_job(desc, location, title, tier1, tier2, tier3, boost_terms, reject_always, culture_terms)

        if result is None:
            # International remote — score 1, don't delete
            result = (1, "international location", None)
            skipped += 1

        score, reasoning, years_required = result
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, years_required = ? WHERE url = ?",
            (score, reasoning, now, years_required, job["url"]),
        )
        dist[score] = dist.get(score, 0) + 1

    conn.commit()
    elapsed = time.time() - t0

    log.info("Done: %d scored, %d removed (international) in %.1fs", len(rows) - skipped, skipped, elapsed)
    for score in sorted(dist.keys(), reverse=True):
        log.info("  score %d: %d jobs", score, dist[score])

    return {
        "scored": len(rows) - skipped,
        "skipped": skipped,
        "errors": 0,
        "elapsed": elapsed,
        "distribution": sorted(dist.items(), reverse=True),
    }
