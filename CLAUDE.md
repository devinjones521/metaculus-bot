# metaculus-bot — devinjones-bot

**Start with `HANDOVER.local.md` if it exists.** It is gitignored and holds the local status,
to-dos and private context that must not be published.

Split out of `../free money` on 2026-09-10. Every design receipt: `docs/METHOD.md`. The private
research record (window sweep, kill criteria K1–K4) stays in `../free money/findings/WINDOWS.md`.
**This repo is PUBLIC** (open-source bots get ~2x Metaculus credit): nothing personal goes in it.

## Commands
- `uv run python -m bot.verify` — ruff, format, mypy, pytest, 4 invariants. One exit code.
- `uv run python -m bot.health` — is the poller on the box alive and sweeping? Run FIRST.
- `uv run python -m bot.alerts --status` — both keys' balances and alert levels. `--test` mails one.
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

## Known defects
1. **`llm_cost_usd` logs $0.00 while the balance falls** (the first forecast cost $0.93). The
   donated key returns no `usage.cost`. The budget guard reads the balance, so it is unaffected;
   the log is wrong. Derive per-sweep cost from the balance delta.
2. **Every MiniBench comment this bot posts is private, and nothing has confirmed that scores.**
   Measured 2026-09-21 on the warmup: all 21 comments read `is_private: true`, which is what the
   code asks for — the 08-24 comments that read `is_private: false` came from the older rig. List
   them with `?author=307009&is_private=true`; the plain listing hides private comments, and
   `?on_post=` returns 403. What is still unknown is whether a private comment satisfies the
   prize rule that asks for reasoning. Worth asking Metaculus before the Fall questions open.

Fixed 2026-09-21: a dropped connection (`RemoteProtocolError`, which carries no status code)
killed 4 of 9 MiniBench questions in the opening hour, because both venues retried on statuses
only. Writes now resolve a drop — `submit` asks the server what landed before resending, `comment`
resends — and a comment that dies after its forecast landed is logged and mailed as
`comment_failed`, not as a failed question. q45942's lost comment was posted by hand; the design
receipts are in `docs/METHOD.md`.

Fixed 2026-09-10: Gemini 3.1 Pro's HTTP 429 (replaced by `gemini-3.8-flash`, evidence in
`docs/METHOD.md`), and Grok is back as the fourth family, on the personal key.

## Funding: two keys
- `OPENROUTER_API_KEY` is **donated by Metaculus** (emailed 2026-09-10): $100 for FutureEval and
  MiniBench, raised automatically on above-average MiniBench performance, ~2x for open-source bots.
  Its account allows only openai, anthropic and google-ai-studio. Usable = balance − ~$8.50 (the
  `max_tokens=8000` reservation 402s below that). Measured cost ~$1.15/question, ~64% research.
- `OPENROUTER_PERSONAL_KEY` is the owner's own. It pays for `x-ai/*` (Grok, ~$0.025/question),
  and for Gemini's second try whenever the donated route fails (~$0.01/run). On the donated key,
  Gemini is BYOK on Metaculus's Google account: $0 to us, and its quota is shared with every bot.
- `bot.alerts` mails the owner on every change of level (ok → low → empty → funded again) for each
  key, from the pollers themselves. Gmail SMTP on port 587: the box blocks 25 and 465.
- `bot.notify` mails the owner when a tournament opens (the first new questions after a quiet day)
  and, after every sweep that submitted or failed anything, what was answered. Each tournament's
  memory lives in `data/metaculus/notify-<slug>.json` on the box.
- `bot.results` mails the owner each resolution: the outcome, the bot's answer and its spot peer
  score, plus the cycle's running total with n and standard error. It checks hourly on every tick,
  parked or not, and costs no model credit. Its memory is `data/metaculus/results-<slug>.json`.
  Checked on the 08-24 cycle on 2026-09-27: +23.9/q, standard error 7.3, n=10, 50 missed.
