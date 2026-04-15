# ApplyPilot — Project Context for Claude

## Purpose

This is a personal fork of ApplyPilot, repurposed as a **job discovery and scoring tool** — not an auto-apply bot. The goal is to surface the best-matching software engineering jobs from across the web, score them with both keyword heuristics and Claude, and let the user review and apply manually.

## Intended Workflow

```
findjobs run discover enrich fastscore llmscore
findjobs review          # open top-scoring jobs in browser
findjobs review --score 8
findjobs portland        # filter for Portland/Oregon area jobs
findjobs browse          # open Greenhouse/Lever jobs (manual-apply sites)
```

## What We Use

- `discover` — JobSpy (Indeed, LinkedIn), Workday employer portals, Greenhouse API
- `enrich` — fetches full job descriptions
- `fastscore` — keyword-based scoring (fast, no LLM)
- `llmscore` — Claude Haiku re-scores based on title nuance, YOE, stack fit
- `review` — opens jobs by score tier in browser for manual review

## What We Don't Use

The `tailor`, `cover`, `pdf`, and `apply` stages exist in the codebase but are not part of this workflow. Do not suggest using or improving them unless asked. Auto-apply is not a goal.

## Scoring Profile

Candidate is a ~3 YOE Python/Django/TypeScript/React fullstack-leaning-backend engineer based in Portland OR. Target roles: remote-US or Portland on-site, mid-level, backend or fullstack. Not interested in: frontend-only, DevOps/SRE, QA, mobile, senior/staff/lead roles, PHP/Java/.NET/Go.
