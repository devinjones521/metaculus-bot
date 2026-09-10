# metaculus-bot — `devinjones-bot`

A forecasting bot for Metaculus's bots-only tournaments:
[FutureEval](https://www.metaculus.com/futureeval/) (three $50k seasons a year) and the
bi-weekly $1k [MiniBench](https://www.metaculus.com/aib/minibench/).

## What it does

For each open question: research it, have five models forecast it independently, take the
median, and submit one forecast with a comment explaining the reasoning. A poller repeats this
every 20 minutes for the whole forecasting window, because MiniBench questions are only open for
about three hours each.

```
research ──► 5 independent forecasts ──► median ──► tail caps ──► CDF built in code ──► submit + comment
(web search)   (Claude ×2, GPT, Gemini,   (one bad    (no 0.1% on    (numeric/discrete)
               Grok: four families)       run can't   a live
                                          move it)    question)
```

Every design choice is backed by an effect size measured in Metaculus's own published analyses
or by open-sourced winners. The table is in [`docs/METHOD.md`](docs/METHOD.md):
research breadth (r=0.42), median-of-5 ensembling (+1,799 pts), tail caps (r=+0.48), base-rate
prompting (r=+0.38), CDFs built in code rather than by the model (bad CDFs are a documented 30%
failure mode), and a guard against forecasting questions that have already resolved.

## Results so far, with the error bars

| cycle | questions | per-question score | rank | prize |
|---|---|---|---|---|
| MiniBench 2026-08-24 | 10 of 58 scored | **+23.9** spot-peer (n=10, se 7.3, 95% CI +9.7 to +38.2) | 27 / 182 | $0 |

That per-question rate was the highest on the board (the winner averaged +23.0 across 49
questions). But n=10 is small, and the ten questions all came from one three-hour window, not a
random sample. What this supports is **"not obviously worse than the top"**, not "best".
The bot placed 27th on **coverage**: it ran twice on the first morning and missed 46
questions. `bot/poll.py` exists to fix that.

## Running it

```
uv sync
cp .env.example .env         # METACULUS_TOKEN, OPENROUTER_API_KEY; the rest optional
uv run python -m bot.verify                                   # lint, types, tests, invariants
uv run python -m bot.forecast --tournament bot-testing-area   # rehearse (resubmission allowed)
uv run python -m bot.poll --tournament minibench --until 2026-10-05T00:00Z
uv run python -m bot.alerts --status                          # both keys' balances and levels
```

`bot-testing-area` is the only place to rehearse. Tournaments expect one forecast per question,
and the code refuses `--resubmit` anywhere else.

### Two keys

Metaculus's donated OpenRouter keys allow only OpenAI, Anthropic and Google models. The Grok
run therefore goes to `OPENROUTER_PERSONAL_KEY`, your own key; everything else, research
included, stays on the donated one. Without a personal key, Grok is dropped from the roster at
startup and the ensemble runs on four models.

### Funding alerts

With `ALERT_SMTP_USER` and `ALERT_SMTP_PASSWORD` (a Gmail app password) set, the poller emails
you when either key runs low, runs out, or is topped up: once per change, not once per tick.
The donated key's "empty" is exactly the point at which the poller parks and stops forecasting.
`uv run python -m bot.alerts --test` sends one test email.

## Invariants (`bot.verify`, one exit code)

1. All network access lives in `bot/venues/`, so there is one place to audit.
2. No skipped tests.
3. No credential in any file git would publish. The check covers both the shape of known key
   formats and the literal values in the local `.env`.
4. Every deployed poller has a liveness contract in `bot.health`, and vice versa.

## License

MIT.
