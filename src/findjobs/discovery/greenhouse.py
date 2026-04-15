"""Greenhouse ATS direct API scraper.

Fetches jobs from companies' public Greenhouse job boards using the
boards-api.greenhouse.io JSON API. No authentication required.

Employer registry is loaded from config/greenhouse_employers.yaml.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from findjobs import config
from findjobs.database import get_connection, init_db, make_canonical_id
from findjobs.discovery.jobspy import _title_ok, _location_ok

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"
API_BASE = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"


def load_employers() -> dict:
    """Load Greenhouse employer registry from config/greenhouse_employers.yaml."""
    path = CONFIG_DIR / "greenhouse_employers.yaml"
    if not path.exists():
        log.warning("greenhouse_employers.yaml not found at %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("employers", {})


def _fetch_jobs(slug: str) -> list[dict]:
    """Fetch all jobs for a company from the Greenhouse API."""
    url = API_BASE.format(slug=slug)
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        return resp.json().get("jobs", [])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            log.warning("%s: Greenhouse board not found (404) — check the slug", slug)
        else:
            log.error("%s: HTTP error %s", slug, e.response.status_code)
        return []
    except Exception as e:
        log.error("%s: fetch error: %s", slug, e)
        return []


def _store_jobs(conn, jobs: list[dict], employer_name: str) -> tuple[int, int]:
    """Store Greenhouse jobs in the DB. Returns (new, existing)."""
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("absolute_url", "")
        if not url:
            continue

        # Check duplicate
        row = conn.execute("SELECT 1 FROM jobs WHERE url = ?", (url,)).fetchone()
        if row:
            existing += 1
            continue

        title = job.get("title", "")
        location = job.get("location", {}).get("name", "")

        # Strip HTML from content
        content = job.get("content", "") or ""
        if content:
            import re
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

        canonical_id = make_canonical_id(title, employer_name)

        existing_url = conn.execute(
            "SELECT url, also_found_on FROM jobs WHERE canonical_id = ? AND url != ?",
            (canonical_id, url),
        ).fetchone()

        conn.execute("""
            INSERT INTO jobs (url, title, site, location, full_description, strategy, discovered_at,
                             canonical_id, discovery_query)
            VALUES (?, ?, ?, ?, ?, 'greenhouse', ?, ?, ?)
        """, (url, title, employer_name, location, content or None, now, canonical_id, employer_name))
        new += 1

        if existing_url:
            other_url, other_also = existing_url
            new_also = f"{other_also},{employer_name}" if other_also else employer_name
            conn.execute("UPDATE jobs SET also_found_on = ? WHERE url = ?", (new_also, other_url))

    conn.commit()
    return new, existing


def run_greenhouse_discovery() -> dict:
    """Main entry point for Greenhouse job discovery."""
    employers = load_employers()
    if not employers:
        log.warning("No Greenhouse employers configured.")
        return {"found": 0, "new": 0, "existing": 0}

    search_cfg = config.load_search_config()
    accept_locs = search_cfg.get("location_accept", [])
    reject_locs = search_cfg.get("location_reject_non_remote", [])
    reject_always = search_cfg.get("location_reject_always", [])
    accept_titles = search_cfg.get("title_accept", [])
    reject_titles = search_cfg.get("title_reject", [])

    init_db()
    conn = get_connection()

    grand_new = 0
    grand_existing = 0
    grand_found = 0
    t0 = time.time()

    for i, (key, emp) in enumerate(employers.items(), 1):
        name = emp["name"]
        slug = emp["slug"]
        log.info("[%d/%d] %s: fetching...", i, len(employers), name)

        raw_jobs = _fetch_jobs(slug)
        if not raw_jobs:
            continue

        # Apply title + location filters
        filtered = []
        for job in raw_jobs:
            title = job.get("title", "")
            location = job.get("location", {}).get("name", "")

            if not _title_ok(title, accept_titles, reject_titles):
                continue
            if not _location_ok(location, accept_locs, reject_locs, reject_always):
                continue
            filtered.append(job)

        grand_found += len(filtered)
        new, existing = _store_jobs(conn, filtered, name)
        grand_new += new
        grand_existing += existing

        log.info("%s: %d total -> %d matched -> %d new, %d dupes",
                 name, len(raw_jobs), len(filtered), new, existing)

    elapsed = time.time() - t0
    log.info("Greenhouse crawl done: %d found, %d new, %d existing in %.0fs",
             grand_found, grand_new, grand_existing, elapsed)

    return {"found": grand_found, "new": grand_new, "existing": grand_existing}
