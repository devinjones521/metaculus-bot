"""Tests for percentiles → CDF. Named for the failures they forbid.

The stakes, from the field: a ~30% numeric submission-failure rate for one Q2
2025 maker ("values must be in strictly increasing order"), and a ~$500 bug in
a rare format for a Fall winner. Whatever the elicited percentiles look like,
the output must satisfy the full server contract — so several tests here are
property checks over ugly inputs rather than examples.
"""

from __future__ import annotations

import pytest

from bot.cdf import DEFAULT_POINTS, build_cdf, scaled_location, validate_cdf

BRENT = {
    0.05: 62.0,
    0.10: 68.0,
    0.25: 78.0,
    0.50: 86.0,
    0.75: 93.0,
    0.90: 101.0,
    0.95: 108.0,
}


def _assert_valid(cdf: list[float], *, lower_open: bool, upper_open: bool, n: int) -> None:
    validate_cdf(cdf, open_lower_bound=lower_open, open_upper_bound=upper_open, n_points=n)


def test_typical_numeric_question_yields_valid_cdf() -> None:
    cdf = build_cdf(BRENT, range_min=55.0, range_max=130.0)
    _assert_valid(cdf, lower_open=True, upper_open=True, n=DEFAULT_POINTS)
    # The median should land near 86 on the value axis: location (86-55)/75 ≈ 0.41.
    median_index = next(i for i, v in enumerate(cdf) if v >= 0.5)
    assert 70 <= median_index <= 95


def test_closed_bounds_pin_ends_exactly() -> None:
    cdf = build_cdf(
        BRENT,
        range_min=55.0,
        range_max=130.0,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    _assert_valid(cdf, lower_open=False, upper_open=False, n=DEFAULT_POINTS)


def test_open_bounds_reserve_tail_mass() -> None:
    """A resolution outside the range must never be scored as impossible."""
    cdf = build_cdf(BRENT, range_min=55.0, range_max=130.0)
    assert cdf[0] > 0.0
    assert cdf[-1] < 1.0


def test_clustered_percentiles_still_clear_min_step() -> None:
    """The spike case: a model 95% sure of one exact value. Without the ramp
    blend this produces flat runs that the server rejects wholesale."""
    spike = {
        0.05: 99.9,
        0.10: 100.0,
        0.25: 100.0,
        0.50: 100.0,
        0.75: 100.0,
        0.90: 100.0,
        0.95: 100.1,
    }
    cdf = build_cdf(spike, range_min=0.0, range_max=1000.0)
    _assert_valid(cdf, lower_open=True, upper_open=True, n=DEFAULT_POINTS)


def test_values_beyond_range_are_survivable() -> None:
    """Models sometimes put p95 above the question's cap. Mass must pile at the
    bound, not crash the pipeline or invert the CDF."""
    wide = {0.05: -50.0, 0.10: 10.0, 0.25: 40.0, 0.50: 80.0, 0.75: 120.0, 0.90: 500.0, 0.95: 900.0}
    cdf = build_cdf(wide, range_min=0.0, range_max=130.0)
    _assert_valid(cdf, lower_open=True, upper_open=True, n=DEFAULT_POINTS)


def test_discrete_grid_size() -> None:
    cdf = build_cdf(BRENT, range_min=55.0, range_max=130.0, n_points=31)
    _assert_valid(cdf, lower_open=True, upper_open=True, n=31)


def test_log_scaled_location_is_not_linear() -> None:
    """zero_point questions: using the linear map is the silent-magnitude bug."""
    linear = scaled_location(1000.0, 100.0, 100000.0, None)
    logged = scaled_location(1000.0, 100.0, 100000.0, 0.0)
    assert abs(logged - 1 / 3) < 1e-9  # log10(1000/100) / log10(100000/100)
    assert abs(linear - logged) > 0.2


def test_log_scaled_cdf_valid() -> None:
    counts = {
        0.05: 200.0,
        0.10: 400.0,
        0.25: 900.0,
        0.50: 2000.0,
        0.75: 5000.0,
        0.90: 20000.0,
        0.95: 60000.0,
    }
    cdf = build_cdf(counts, range_min=100.0, range_max=100000.0, zero_point=0.0)
    _assert_valid(cdf, lower_open=True, upper_open=True, n=DEFAULT_POINTS)


def test_decreasing_values_raise() -> None:
    """Percentile values that fall as percentiles rise are not a distribution;
    silently sorting them would submit a forecast nobody made."""
    with pytest.raises(ValueError):
        build_cdf({0.10: 50.0, 0.50: 40.0, 0.90: 60.0}, range_min=0.0, range_max=100.0)


def test_no_step_exceeds_spike_cap() -> None:
    cdf = build_cdf(BRENT, range_min=55.0, range_max=130.0)
    steps = [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]
    assert max(steps) <= 0.15 + 1e-9
