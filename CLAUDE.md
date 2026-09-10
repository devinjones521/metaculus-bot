# metaculus-bot — devinjones-bot

Split out of `../free money` on 2026-09-10. Every design receipt: `docs/METHOD.md`. The private
research record (window sweep, kill criteria K1–K4) stays in `../free money/findings/WINDOWS.md`.
**This repo is PUBLIC** (open-source bots get ~2x Metaculus credit): nothing personal goes in it.

## Commands
- `uv run python -m bot.verify` — ruff, format, mypy, pytest, 3 invariants. One exit code.
- `uv run python -m bot.health` — is the poller on the box alive and sweeping? Run FIRST.
- `uv run python -m bot.forecast --tournament bot-testing-area` — rehearsal (resubmit allowed).
- `uv run python -m bot.poll --tournament <slug> --until <ISO>` — the unattended loop.
- Predictions about this bot go in the free-money ledger: from `../free money`,
  `uv run python -m bot.calibrate --predict "..." --p 0.x --resolves YYYY-MM-DD`.

## Rules that override convenience
1. **Nothing secret in any tracked file, ever.** `verify` scans every publishable file for key
   shapes AND for the literal `.env` values. Keys live in `.env` (gitignored) only.
2. **One forecast per question in real tournaments.** Resubmission and rehearsal happen only in
   `bot-testing-area` (32977). Never preview or rerun on live tournament questions.
3. **Comments are required for prizes** and must reflect the bot's actual reasoning.
4. **All network access lives in `bot/venues/`.** `verify` enforces it.
5. **Every number carries N and a standard error.** +23.9/q was n=10 from one 3-hour window.
6. **A quiet log is not a healthy one.** A poll that sees nothing looks exactly like a broken one;
   `polls.jsonl` has a row per sweep, including failed ones, and `bot.health` reads its age.
7. **Never weaken, skip or delete a test to go green.**

## Funding
The OpenRouter key is **donated by Metaculus** (emailed 2026-09-10): $100 for FutureEval
and MiniBench, raised automatically on above-average MiniBench performance, and ~2x for open-source
bots. Usable = balance − ~$8.50 (the `max_tokens=8000` reservation 402s below that). Measured cost
is ~$1.15/question, of which research is ~64%.
