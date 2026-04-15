"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from findjobs import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="findjobs",
    help="Find and score jobs from across the web. Discover, rank, review.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "fastscore", "llmscore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from findjobs.config import load_env, ensure_dirs
    from findjobs.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]findjobs[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from findjobs.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help="Pipeline stages to run: discover, enrich, fastscore, llmscore. Defaults to all four.",
    ),
    rescore: bool = typer.Option(False, "--rescore", help="Re-score all jobs (fastscore only)."),
    smart_extract: bool = typer.Option(False, "--smart-extract", help="Enable AI-powered smart extract scraper."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    # hidden options kept for backward compat
    min_score: int = typer.Option(7, "--min-score", hidden=True),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", hidden=True),
    stream: bool = typer.Option(False, "--stream", hidden=True),
    validation: str = typer.Option("normal", "--validation", hidden=True),
) -> None:
    """Scrape and score jobs (discover → enrich → fastscore → llmscore)."""
    _bootstrap()

    from findjobs.pipeline import run_pipeline

    stage_list = stages if stages else ["discover", "enrich", "fastscore", "llmscore"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from findjobs.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        limit=limit,
        rescore=rescore,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
        smart_extract=smart_extract,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command(hidden=True)
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from findjobs.config import check_tier, PROFILE_PATH as _profile_path
    from findjobs.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from findjobs.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from findjobs.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from findjobs.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    # Check 1: Tier 3 required (Claude Code CLI + Chrome)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]findjobs init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Scored jobs exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? AND applied_at IS NULL",
            (min_score,),
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No scored jobs ready to apply to.[/red]\n"
                "Run [bold]findjobs run fastscore[/bold] first."
            )
            raise typer.Exit(code=1)

    if gen:
        from findjobs.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from findjobs.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
    )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from findjobs.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from findjobs.view import open_dashboard

    open_dashboard()


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from findjobs.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'findjobs init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'findjobs init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'findjobs init'"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM API key", fail_mark,
                        "Set GEMINI_API_KEY in ~/.findjobs/.env (run 'findjobs init')"))

    # --- Tier 3 checks ---
    # Claude Code CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", fail_mark,
                        "Install from https://claude.ai/code (needed for auto-apply)"))

    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    # Node.js / npx (for Playwright MCP)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for auto-apply)"))

    # CapSolver (optional)
    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyPilot Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from findjobs.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")

    console.print()


@app.command()
def local(
    min_score: int = typer.Option(1, "--min-score", help="Minimum fit score to include."),
) -> None:
    """Show on-site/local jobs based on your location_accept config, ordered by score."""
    from findjobs.config import load_search_config
    from findjobs.database import get_connection, init_db

    _bootstrap()
    init_db()

    search_cfg = load_search_config() or {}
    location_accept = search_cfg.get("location_accept", [])
    local_terms = [t for t in location_accept if t.lower() not in ("remote", "anywhere", "us")]

    if not local_terms:
        console.print("[yellow]No local locations configured in searches.yaml (location_accept).[/yellow]")
        raise typer.Exit()

    where_clauses = " OR ".join(f"location LIKE '%{term}%'" for term in local_terms)

    conn = get_connection()
    rows = conn.execute(f"""
        SELECT title, location, fit_score, score_reasoning, url
        FROM jobs
        WHERE ({where_clauses})
          AND (applied_at IS NULL OR apply_status = 'manual')
          AND fit_score >= ?
        ORDER BY fit_score DESC
    """, (min_score,)).fetchall()

    label = " / ".join(t.title() for t in local_terms)
    if not rows:
        console.print(f"[yellow]No local jobs found for: {label}[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold]Local jobs — {label} ({len(rows)} found):[/bold]\n")
    for r in rows:
        score_color = "green" if r["fit_score"] >= 7 else "yellow" if r["fit_score"] >= 4 else "dim"
        console.print(f"  [[{score_color}]{r['fit_score']}[/{score_color}]] {r['title']}")
        console.print(f"        [dim]{r['location']}[/dim]")
        console.print(f"        [dim]{r['score_reasoning']}[/dim]")
        console.print(f"        [dim]{r['url'][:80]}[/dim]")
        console.print()


@app.command()
def search(
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment."),
) -> None:
    """Discover jobs and score them (discover + enrich + fastscore + llmscore).

    One command for the full scrape-and-score cycle.
    """
    from findjobs.pipeline import run_pipeline

    _bootstrap()
    run_pipeline(
        stages=["discover", "enrich", "fastscore", "llmscore"],
        workers=workers,
    )


@app.command()
def browse(
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score to include."),
    limit: int = typer.Option(20, "--limit", "-l", help="Max jobs to open."),
) -> None:
    """Open manual-apply jobs (Greenhouse, Lever, etc.) in your browser, ordered by score.

    Jobs are marked as 'manual' so they are never auto-applied.
    Already-opened jobs are skipped.
    """
    import webbrowser
    from datetime import datetime, timezone
    from findjobs.config import load_search_config
    from findjobs.database import get_connection, init_db

    _bootstrap()
    init_db()

    search_cfg = load_search_config() or {}
    manual_sites = search_cfg.get("manual_apply_sites", ["greenhouse.io", "lever.co", "ashby.io"])

    if not manual_sites:
        console.print("[yellow]No manual_apply_sites configured in searches.yaml.[/yellow]")
        raise typer.Exit()

    # Build WHERE clause matching any manual site
    site_clauses = " OR ".join(f"url LIKE '%{site}%'" for site in manual_sites)

    conn = get_connection()
    rows = conn.execute(f"""
        SELECT url, title, fit_score, score_reasoning
        FROM jobs
        WHERE fit_score >= ?
          AND applied_at IS NULL
          AND (apply_status IS NULL OR apply_status NOT IN ('manual', 'applied'))
          AND ({site_clauses})
        ORDER BY fit_score DESC
        LIMIT ?
    """, (min_score, limit)).fetchall()

    if not rows:
        console.print(f"[yellow]No manual-apply jobs found scoring >= {min_score}.[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold]Opening {len(rows)} jobs in browser (score >= {min_score}):[/bold]\n")

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        url, title, score, reasoning = row["url"], row["title"], row["fit_score"], row["score_reasoning"]
        console.print(f"  [{score}] {title}")
        console.print(f"        [dim]{url}[/dim]")
        webbrowser.open(url)
        conn.execute(
            "UPDATE jobs SET apply_status = 'manual', applied_at = ?, apply_error = 'manual apply' WHERE url = ?",
            (now, url),
        )

    conn.commit()
    console.print(f"\n[green]Opened {len(rows)} jobs and marked as manual.[/green]")
    console.print("[dim]These will not appear in future browse or auto-apply runs.[/dim]\n")


@app.command()
def review(
    score: int = typer.Option(None, "--score", "-s", help="Open jobs with exactly this score."),
    min_score: int = typer.Option(None, "--min-score", help="Open jobs with at least this score."),
    limit: int = typer.Option(50, "--limit", "-l", help="Max jobs to open."),
) -> None:
    """Open job URLs in your browser by score tier for manual review.

    Without arguments, shows the score distribution and opens the top score tier.
    Use --score 9 to open all 9s, --min-score 7 to open everything >= 7.
    Jobs are not marked as applied — they stay visible for future runs.
    """
    import webbrowser
    from findjobs.database import get_connection, init_db

    _bootstrap()
    init_db()

    conn = get_connection()

    # Show score distribution first
    dist = conn.execute(
        "SELECT fit_score, COUNT(*) as n FROM jobs "
        "WHERE fit_score IS NOT NULL AND (applied_at IS NULL OR apply_status NOT IN ('applied')) "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()

    if not dist:
        console.print("[yellow]No scored jobs found.[/yellow]")
        raise typer.Exit()

    console.print("\n[bold]Score distribution:[/bold]")
    for row in dist:
        bar = "█" * min(row["n"], 40)
        score_color = "green" if row["fit_score"] >= 7 else "yellow" if row["fit_score"] >= 4 else "dim"
        console.print(f"  [{score_color}]{row['fit_score']:2d}[/{score_color}]: {bar} ({row['n']})")
    console.print()

    # Determine which score(s) to open
    if score is not None:
        where = "fit_score = ?"
        params = (score, limit)
        label = f"score = {score}"
    elif min_score is not None:
        where = "fit_score >= ?"
        params = (min_score, limit)
        label = f"score >= {min_score}"
    else:
        top_score = dist[0]["fit_score"]
        where = "fit_score = ?"
        params = (top_score, limit)
        label = f"score = {top_score} (top tier)"

    rows = conn.execute(f"""
        SELECT url, title, site, fit_score, score_reasoning
        FROM jobs
        WHERE {where}
          AND url IS NOT NULL
          AND (applied_at IS NULL OR apply_status NOT IN ('applied'))
        ORDER BY fit_score DESC, title
        LIMIT ?
    """, params).fetchall()

    if not rows:
        console.print(f"[yellow]No jobs found for {label}.[/yellow]")
        raise typer.Exit()

    console.print(f"[bold]Opening {len(rows)} jobs ({label}):[/bold]\n")
    for row in rows:
        score_color = "green" if row["fit_score"] >= 7 else "yellow" if row["fit_score"] >= 4 else "dim"
        site = row["site"] or ""
        console.print(f"  [[{score_color}]{row['fit_score']}[/{score_color}]] {row['title']}" + (f" @ {site}" if site else ""))
        if row["score_reasoning"]:
            console.print(f"        [dim]{row['score_reasoning']}[/dim]")
        console.print(f"        [dim]{row['url'][:80]}[/dim]")
        webbrowser.open(row["url"])

    console.print(f"\n[green]Opened {len(rows)} jobs.[/green] [dim](not marked — will appear in future runs)[/dim]\n")


if __name__ == "__main__":
    app()
