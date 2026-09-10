# Method: why the bot is built the way it is

Every design choice below was made **before** the bot had a score. Each maps to an effect
someone had already measured. The sources are Metaculus's own analyses (the Fall 2025 bot-maker
survey, bot-maker advice, a three-season search-provider study, a calibration-adjustment study,
and the "What 11 Analyses Say" synthesis). Two open bots were also studied: the Q2 2025 winner
(Panshul42, AGPL — **ideas only, no code taken**) and nostreambot (MIT). So were the template
bot's API mechanics and the AIA Forecaster / silicon-crowd-diversity papers.

## The design, and the evidence for each part

| design choice | evidence |
|---|---|
| 2 research sources (model-native web search ×2 rounds + AskNews), breadth over brand | number of sources ↔ score r=0.42, the strongest predictor; no single provider wins |
| 5-run ensemble across ≥2 model families, median taken in code | aggregators +1,799 pts (CI +1,017..+2,582); 86% of winners aggregate; asking an LLM to merge distributions is a named trap |
| tail caps [0.02, 0.98], tighter [0.10, 0.90] when flagged | capping is the strongest differentiator among winners, r=+0.48 |
| explicit base-rate step in every prompt | r=+0.38 among winners (40% of the top 15 vs 7% of the bottom) |
| already-resolved guard in the prompt, plus a second-look check on any aggregate ≥97% | the known catastrophic mode: 99% on a question that is still open |
| percentiles → CDF entirely in code, ramp-blended so the minimum step 0.01/(n-1) holds by construction | one maker lost ~30% of numeric submissions to "strictly increasing" rejections |
| reliability before cleverness: per-question containment, submit on one surviving run, JSONL logs | the synthesis names "scaffolding before reliability" as an anti-pattern (10 missed questions = −150 pts) |
| ~8–12 LLM calls per question (the winners' median is 28) | a deliberate floor for the first cycles; tune on MiniBench scores, not in advance |
| no "Bayesian" framing in prompts | measured to underperform in both prompt-optimisation studies |
| a poller that sweeps every 20 minutes for the whole window | MiniBench questions are open ~3 hours each and arrive in a rolling stream. One morning run covered 10 of 58 questions (see README) |

The ensemble default samples the Anthropic and OpenAI models twice each and Gemini once.
That is five runs over three families. It is deliberate, and it gives the doubled models
double weight in the median.

## Deliberately not built yet

Each waits until MiniBench scores say it matters:

- prediction-market priors (Polymarket/Kalshi snapshots; the rules allow it, and nostreambot does it)
- lookup of similar resolved questions on Metaculus (34% of winners vs 0% of non-winners)
- a fetcher for the page each question resolves against
- Platt-scaling recalibration (needs ~500 logged forecasts; the log schema already records them)
- AskNews DeepNews (used by the top research-only bot; costs 5 calls per query)
- a strict-cutoff backtester

## How prizes are paid, and why coverage matters so much

From the scores FAQ:

- rank = Σ spot-peer scores
- take = max(Σ, 0)²
- prize share = take / Σ all takes
- prizes under $50 are cancelled and redistributed upward

Because the take is squared and small prizes are cut, **coverage compounds**. Half the questions
earns roughly none of the money, not half of it. On a $1k MiniBench across ~180 bots, only about
the top 5–10 are paid in a cycle. So a $0 cycle is the most likely outcome even for a good bot,
and it is not a verdict on its own.

Operating costs, as measured: about **$1.15 per question** on OpenRouter, of which research is
about 64%. OpenRouter reserves `max_tokens × output price` against the balance, so below about
$8.50 every ensemble call fails with HTTP 402 before generating anything. `bot.poll` stops
spending before that point.

## Sources

- FutureEval hub and resources: <https://www.metaculus.com/futureeval/> ·
  <https://www.metaculus.com/notebooks/38928/futureeval-resources-page/>
- Spring 2026 analyses: <https://www.metaculus.com/notebooks/45373/spring-2026-futureeval-analysis/> ·
  <https://www.metaculus.com/notebooks/45382/bot-survey-spring-2026/> ·
  <https://www.metaculus.com/notebooks/45336/bot-maker-advice-spring-2026/>
- Q2 2025 winners: <https://www.metaculus.com/notebooks/39140/winners-of-q2-2025-ai-benchmark-tournament/>
- Q2 2025 analysis (Pros beat bots, p=0.00001):
  <https://www.lesswrong.com/posts/Surnjh8A4WjgtQTkZ/q2-ai-benchmark-results-pros-maintain-clear-lead>
- Template bot: <https://github.com/Metaculus/metac-bot-template> · nostreambot:
  <https://github.com/No-Stream/metaculus-bot>
