"""Batch LLM re-scoring using the claude CLI.

After keyword scoring (fastscore), this stage sends all keyword-scored jobs to
Claude in one batch subprocess call. Claude adjusts scores based on contextual
understanding that keyword matching misses: title nuance, YOE buried in prose,
role discipline context, company signals.

Uses `claude -p --output-format json --model haiku` — one subprocess call per
batch of 100 jobs. No separate API key required; uses the existing claude CLI
session.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone

from rich.console import Console

from findjobs.config import load_profile, load_search_config
from findjobs.database import get_connection

log = logging.getLogger(__name__)
console = Console()

_MODEL = "haiku"
_MAX_JOBS_PER_BATCH = 100  # keep prompt comfortably under context limits


def _build_system_prompt() -> str:
    """Build the Claude scoring prompt from profile.json and searches.yaml."""
    try:
        profile = load_profile()
    except FileNotFoundError:
        profile = {}
    search_cfg = load_search_config() or {}

    exp = profile.get("experience", {})
    yoe = exp.get("years_of_experience_total", "?")
    current_title = exp.get("current_title", "software engineer")
    target_role = exp.get("target_role", "Software Engineer")

    tiers = search_cfg.get("skill_tiers", {})
    tier1 = tiers.get("tier1", [])
    tier2 = tiers.get("tier2", [])
    stack = ", ".join(tier1[:8] + tier2[:4]) if (tier1 or tier2) else "Python, TypeScript, React"

    loc_accept = search_cfg.get("location_accept", [])
    local_locs = [t for t in loc_accept if t.lower() not in ("remote", "anywhere", "us")]
    loc_desc = f"remote-US or {' / '.join(t.title() for t in local_locs)} on-site" if local_locs else "remote-US"

    reject_always = search_cfg.get("location_reject_always", [])
    reject_loc_str = f" Not interested in international locations: {', '.join(reject_always[:5])}." if reject_always else ""

    return f"""You are a job fit evaluator for a software engineer job search.

CANDIDATE:
- ~{yoe} years experience, current title: {current_title}, target: {target_role}
- Wants: fullstack-leaning-backend or pure backend roles
- NOT interested in: frontend-only, DevOps/SRE, QA/SDET, mobile, senior/staff/lead/principal roles
- Stack: {stack}
- Location: {loc_desc}{reject_loc_str}
- Not interested in: PHP, Java, .NET, C#, Golang, Scala, hardware/embedded, ML/AI engineering

Each job line is: id={{id}} | {{title}} @ {{company}} | {{location}} | keyword_score={{N}} | desc: {{first 300 chars}}

Adjust the keyword_score up or down based on what you can infer:

Green flags (raise score by 1-3):
- "mid-level", "midlevel", "2-4 years", "2-3 years", "3-5 years" in description
- Title is clean and generic: "Software Engineer", "Full Stack Engineer", "Backend Engineer"
- Primary stack matches candidate's tier1 skills
- Role is clearly backend or fullstack with backend emphasis

Red flags (lower score by 1-3):
- Title has seniority hidden in description ("5+ years required", "senior-level responsibilities")
- Role turns out to be infrastructure/platform engineering (often DevOps despite "engineer" title)
- Description reveals frontend-only work despite "fullstack" in title
- Java, PHP, .NET, C# mentioned as the primary stack with no Python/TS
- 4+ years required buried in description prose
- Role is at a non-US location that snuck through

Return ONLY a JSON array, no other text:
[{{"id": <int>, "score": <1-10>, "reason": "<one short phrase>"}}]
"""


def _build_summary(job: dict) -> str:
    desc = (job.get("full_description") or job.get("description") or "")[:300]
    desc = desc.replace("\n", " ").strip()
    company = job.get("company") or job.get("site") or "?"
    return (
        f'id={job["id"]} | {job.get("title", "?")} @ {company} | '
        f'{job.get("location", "?")} | keyword_score={job.get("fit_score", "?")} | '
        f'desc: {desc}'
    )


def _call_claude(jobs: list[dict]) -> list[dict]:
    """Send a batch of job summaries to claude CLI and return parsed adjustments."""
    system_prompt = _build_system_prompt()
    summaries = "\n".join(_build_summary(j) for j in jobs)
    prompt = (
        f"{system_prompt}\n\n"
        f"Review these {len(jobs)} jobs and return the JSON array:\n\n"
        f"{summaries}"
    )

    t0 = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--model", _MODEL],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. Make sure Claude Code is installed and `claude` is in PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude CLI timed out after 120s")

    elapsed = time.time() - t0

    if proc.returncode != 0:
        stderr = proc.stderr[:300] if proc.stderr else "(no stderr)"
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {stderr}")

    # Parse the outer JSON envelope from --output-format json
    try:
        envelope = json.loads(proc.stdout)
        raw_text = envelope.get("result", "")
    except (json.JSONDecodeError, AttributeError):
        raw_text = proc.stdout

    # Extract the JSON array from Claude's response text
    match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
    if not match:
        log.error("Claude returned no JSON array (%.1fs): %s", elapsed, raw_text[:300])
        return []

    try:
        adjustments = json.loads(match.group())
        log.info("Claude batch: %d jobs -> %d adjustments (%.1fs)", len(jobs), len(adjustments), elapsed)
        return adjustments
    except json.JSONDecodeError as e:
        log.error("Failed to parse Claude JSON: %s — %s", e, raw_text[:300])
        return []


def run_llm_scoring(rescore: bool = False) -> dict:
    """Batch re-score keyword-scored jobs with Claude.

    Fetches jobs that have fit_score (from fastscore) but haven't been LLM-scored yet,
    sends them to Claude in batches, and overwrites fit_score with Claude's adjusted score.
    """
    conn = get_connection()

    if rescore:
        query = (
            "SELECT rowid as id, * FROM jobs WHERE fit_score IS NOT NULL "
            "AND full_description IS NOT NULL ORDER BY fit_score DESC"
        )
    else:
        query = (
            "SELECT rowid as id, * FROM jobs WHERE fit_score IS NOT NULL "
            "AND llm_scored_at IS NULL "
            "AND full_description IS NOT NULL ORDER BY fit_score DESC"
        )

    rows = conn.execute(query).fetchall()
    if not rows:
        console.print("  [dim]No jobs pending LLM scoring.[/dim]")
        return {"scored": 0, "elapsed": 0.0}

    cols = rows[0].keys()
    jobs = [dict(zip(cols, r)) for r in rows]

    console.print(f"  [cyan]Sending {len(jobs)} jobs to Claude ({_MODEL}) for batch scoring...[/cyan]")

    t_total = time.time()
    all_adjustments: list[dict] = []

    for i in range(0, len(jobs), _MAX_JOBS_PER_BATCH):
        batch = jobs[i : i + _MAX_JOBS_PER_BATCH]
        batch_num = i // _MAX_JOBS_PER_BATCH + 1
        total_batches = (len(jobs) + _MAX_JOBS_PER_BATCH - 1) // _MAX_JOBS_PER_BATCH
        console.print(f"  Batch {batch_num}/{total_batches}: {len(batch)} jobs...")
        try:
            adjustments = _call_claude(batch)
            all_adjustments.extend(adjustments)
        except Exception as e:
            log.error("Batch %d failed: %s", batch_num, e)
            console.print(f"  [red]Batch {batch_num} error:[/red] {e}")

    if not all_adjustments:
        console.print("  [yellow]No adjustments returned from Claude.[/yellow]")
        return {"scored": 0, "elapsed": time.time() - t_total}

    # Apply adjustments — overwrite fit_score with Claude's score
    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    for adj in all_adjustments:
        job_id = adj.get("id")
        new_score = adj.get("score")
        reason = adj.get("reason", "")
        if job_id is None or new_score is None:
            continue
        new_score = max(1, min(10, int(new_score)))
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, llm_scored_at = ? WHERE rowid = ?",
            (new_score, reason, now, job_id),
        )
        updated += 1

    conn.commit()
    elapsed = time.time() - t_total
    console.print(f"  [green]Updated {updated} scores in {elapsed:.1f}s[/green]")

    # Score distribution after LLM pass
    dist = conn.execute(
        "SELECT fit_score, COUNT(*) FROM jobs WHERE llm_scored_at IS NOT NULL "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()
    for score, count in dist:
        bar = "█" * min(count, 40)
        console.print(f"  {score:2d}: {bar} ({count})")

    return {"scored": updated, "elapsed": elapsed}
