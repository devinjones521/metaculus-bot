"""Tests for run aggregation: median, caps, floors — the measured post-forecast moves."""

from __future__ import annotations

import pytest

from bot.elicit import NUMERIC_PERCENTILES
from bot.ensemble import (
    BINARY_CAP,
    BINARY_CAP_SUSPECT,
    aggregate_binary,
    aggregate_multiple_choice,
    aggregate_percentiles,
    is_extreme,
)


def test_median_ignores_one_deranged_run() -> None:
    """Rule 4 with prize money attached: one 99% hallucination among sane runs
    must not drag the submission."""
    assert aggregate_binary([0.30, 0.32, 0.99, 0.28, 0.31]) == pytest.approx(0.31)


def test_tail_caps_apply() -> None:
    lo, hi = BINARY_CAP
    assert aggregate_binary([0.999, 0.999, 0.999]) == hi
    assert aggregate_binary([0.001, 0.001, 0.001]) == lo


def test_suspect_forecasts_get_pulled_further_from_the_cliff() -> None:
    lo, hi = BINARY_CAP_SUSPECT
    assert aggregate_binary([0.99, 0.99, 0.99], suspect=True) == hi
    assert aggregate_binary([0.01, 0.01, 0.01], suspect=True) == lo


def test_mc_floor_and_exact_normalisation() -> None:
    """Zero on the realised option is unboundedly bad under a log score, so
    every option keeps a floor — and the distribution still sums to exactly 1."""
    options = ("A", "B", "C")
    runs = [{"A": 0.98, "B": 0.02, "C": 0.0}, {"A": 0.96, "B": 0.04, "C": 0.0}]
    out = aggregate_multiple_choice(runs, options)
    assert out["C"] > 0.0
    assert sum(out.values()) == 1.0


def test_percentile_median_stays_monotone() -> None:
    runs = [
        dict(zip(NUMERIC_PERCENTILES, [10, 20, 30, 40, 50, 60, 70], strict=True)),
        dict(zip(NUMERIC_PERCENTILES, [5, 25, 28, 45, 55, 58, 90], strict=True)),
        dict(zip(NUMERIC_PERCENTILES, [12, 18, 33, 39, 52, 65, 68], strict=True)),
    ]
    out = aggregate_percentiles(runs)
    values = [out[p] for p in NUMERIC_PERCENTILES]
    assert values == sorted(values)
    assert out[0.50] == 40


def test_empty_runs_raise_rather_than_fabricate() -> None:
    with pytest.raises(ValueError):
        aggregate_binary([])
    with pytest.raises(ValueError):
        aggregate_multiple_choice([], ("A",))
    with pytest.raises(ValueError):
        aggregate_percentiles([])


def test_extremeness_trigger() -> None:
    assert is_extreme(0.98)
    assert is_extreme(0.02)
    assert not is_extreme(0.90)
