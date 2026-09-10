"""Aggregate independent forecast runs into one submission. Pure math, no LLM.

The two strongest post-forecast moves in the evidence base live here:

  - **Median aggregation across runs.** Botmakers who aggregated scored +1,799
    points over those who did not (Q2 2025, 95% CI +1,017 to +2,582); 86% of
    Fall 2025 winners aggregate. Median, not mean — one deranged run must not
    drag the submission (this repo's rule 4, now with prize money attached).
  - **Tail caps.** Capping final outputs was the strongest within-winners
    differentiator in Fall 2025 (r = +0.48). One confidently-wrong extreme
    under a log score erases a season; the bounded upside of an extreme right
    answer never pays for that. Tighter caps apply when the extremeness check
    flagged the forecast as SUSPECT.

An LLM is never asked to merge distributions — a Fall winner's words: it
"kept driving me insane". Runs come in, numbers come out, deterministically.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from bot.elicit import NUMERIC_PERCENTILES

BINARY_CAP = (0.02, 0.98)
BINARY_CAP_SUSPECT = (0.10, 0.90)
# Every multiple-choice option keeps at least this much mass: a zero on the
# realised option is unboundedly bad, and 0.5% costs nearly nothing when right.
MC_FLOOR = 0.005


def aggregate_binary(runs: Sequence[float], *, suspect: bool = False) -> float:
    if not runs:
        raise ValueError("no runs to aggregate")
    lo, hi = BINARY_CAP_SUSPECT if suspect else BINARY_CAP
    return min(hi, max(lo, statistics.median(runs)))


def aggregate_multiple_choice(
    runs: Sequence[Mapping[str, float]], options: Sequence[str]
) -> dict[str, float]:
    """Per-option median, floored, then renormalised to sum to exactly 1."""
    if not runs:
        raise ValueError("no runs to aggregate")
    medians = {option: statistics.median(run[option] for run in runs) for option in options}
    floored = {k: max(v, MC_FLOOR) for k, v in medians.items()}
    total = sum(floored.values())
    normalised = {k: v / total for k, v in floored.items()}
    # Float dust: make the sum exactly 1.0 on the largest option, where a
    # 1e-12 nudge is invisible.
    drift = 1.0 - sum(normalised.values())
    largest = max(normalised, key=lambda k: normalised[k])
    normalised[largest] += drift
    return normalised


def aggregate_percentiles(
    runs: Sequence[Mapping[float, float]],
) -> dict[float, float]:
    """Per-percentile median across runs.

    Aggregation happens in percentile space, before the CDF is built, so the
    CDF constraints are enforced exactly once, on exactly one distribution.
    Medians of individually non-decreasing sequences are non-decreasing, so the
    output is a valid percentile set by construction.
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    return {p: statistics.median(run[p] for run in runs) for p in NUMERIC_PERCENTILES}


def is_extreme(probability: float, *, threshold: float = 0.97) -> bool:
    """Does a binary aggregate sit in the tail where the known catastrophic
    failure modes (already-resolved, misread criteria) concentrate?"""
    return probability >= threshold or probability <= 1.0 - threshold
