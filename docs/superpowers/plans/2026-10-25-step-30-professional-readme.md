# Step 30 — Professional README

## Goal

Turn the repository's 2,398-line chronological build log into a front page that
lets a new reader understand, run, evaluate, and inspect the project in minutes.
The detailed experimental record remains in the roadmap, step plans, committed
benchmark histories, and git history.

## Verified starting point

- Steps 02–29 are complete on `main` at `bfc2bf2`.
- Ruff, Ruff format, and strict mypy pass.
- With the repository Redis service running, 653 non-paid tests pass and branch
  coverage is 95.79%, above the 95% gate.
- The first CI run after step 28 passed both the quality and pinned-corpus
  retrieval-regression jobs.
- The offline dashboard contains 63 labelled views built from 64 committed rows.

## Contract

- Keep the public README in French and make the first screen explain the product,
  its evidence, and its current status.
- Add the verified CI badge; do not add badges for services that do not exist.
- Include architecture, Docker quick start, local development, API examples,
  measured results, deliberate defaults, limitations, and documentation links.
- Report only numbers already present in committed benchmark histories or verified
  by the step-30 gate. Do not rerun paid benchmarks for a documentation change.
- Keep operational details in `.env.example` and research detail in
  `docs/roadmap.md`; the README should route to them instead of duplicating them.

## Execution

1. Replace the chronological README with a task-oriented front page.
2. Check every command, link, endpoint, default, and headline number against the
   current repository.
3. Run Markdown/link sanity checks plus the repository quality gates.
4. Mark step 30 complete in the global roadmap.

## Definition of done

- A new reader can identify the use case, architecture, and strongest measured
  result without scrolling through implementation history.
- Docker setup reaches a grounded query from a fresh clone with explicit required
  credentials.
- The README distinguishes shipped defaults from measured-but-rejected options.
- CI status, dashboard, roadmap, plans, configuration, and license are linked.
- No source code, dependency, benchmark history, or runtime default changes.
