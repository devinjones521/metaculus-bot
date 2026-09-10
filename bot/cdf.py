"""Percentiles → a Metaculus-valid CDF. The most dangerous 200 numbers in the bot.

WHY THIS MODULE EXISTS AS PURE MATH
-----------------------------------

Continuous questions are where other bots bleed: one Q2 2025 maker reported a
~30% submission-failure rate from "values must be in strictly increasing order",
and a Fall 2025 winner lost ~$500 of prize money to one bug in a rare question
format. The LLM is never asked to emit 201 numbers — it is asked for a handful
of percentile values, and everything from there to the wire is deterministic
code with property tests. (The advice post is blunt: using LLMs to combine
distributions "kept driving me insane".)

THE CONTRACT THE OUTPUT SATISFIES, ALWAYS
-----------------------------------------

For a grid of `n` points (201 for numeric, outcome_count+1 for discrete):
  - every value in [0, 1], strictly increasing;
  - consecutive steps >= 0.01/(n-1) (the server's 5e-05 at n=201), guaranteed
    by blending in a linear ramp rather than by post-hoc fixups;
  - closed bounds pin the ends to exactly 0 / 1; open bounds leave mass
    outside (>= ~0.1%) so a resolution beyond the range is never scored as
    impossible;
  - no single step exceeds MAX_STEP, because a spike distribution is an
    overconfidence cliff under a log score.

HOW IT COULD LIE
----------------

- **Log-scaled questions.** `zero_point` questions map value→location through a
  log, and using the linear map instead produces a plausible-looking CDF that
  is confidently wrong about magnitudes. The mapping is explicit and tested in
  both directions.
- **Percentiles the model got backwards.** Non-monotone (percentile, value)
  input raises rather than being silently sorted into a distribution nobody
  forecast.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

DEFAULT_POINTS = 201
# Blend weight for the uniform ramp. Guarantees min step 0.011*(hi-lo)/(n-1),
# which clears the server's 0.01/(n-1) with the open-bound span of 0.998.
RAMP_WEIGHT = 0.011
# Mass held outside an open bound. The template bot uses the same order.
OPEN_BOUND_TAIL = 0.001
# Cap on any single step of probability mass (spike guard).
MAX_STEP = 0.15


def scaled_location(
    value: float, range_min: float, range_max: float, zero_point: float | None
) -> float:
    """Map a real-world value onto the question's [0, 1] axis.

    Linear questions: simple affine. Log-scaled questions (zero_point set) use
    Metaculus's formula: log((v-zp)/(min-zp)) / log((max-zp)/(min-zp)).
    """
    if range_max <= range_min:
        raise ValueError(f"degenerate range [{range_min}, {range_max}]")
    if zero_point is None:
        return (value - range_min) / (range_max - range_min)
    if zero_point >= range_min:
        raise ValueError(f"zero_point {zero_point} not below range_min {range_min}")
    numerator = (value - zero_point) / (range_min - zero_point)
    denominator = (range_max - zero_point) / (range_min - zero_point)
    if numerator <= 0:
        return -1.0  # far below the range; caller clips
    return math.log(numerator) / math.log(denominator)


def build_cdf(
    percentile_values: Mapping[float, float],
    *,
    range_min: float,
    range_max: float,
    zero_point: float | None = None,
    open_lower_bound: bool = True,
    open_upper_bound: bool = True,
    n_points: int = DEFAULT_POINTS,
) -> list[float]:
    """Build the CDF from elicited percentiles ({0.05: value, ..., 0.95: value}).

    Raises on input that does not describe a distribution (fewer than two
    distinct points, values that decrease as percentiles increase). Everything
    else — values outside the range, clustered values — is handled and still
    yields a valid CDF.
    """
    if n_points < 2:
        raise ValueError(f"n_points {n_points} too small")
    items = sorted(percentile_values.items())
    if len(items) < 2:
        raise ValueError("need at least two percentile points")
    percentiles = [p for p, _ in items]
    values = [v for _, v in items]
    if any(not (0.0 < p < 1.0) for p in percentiles):
        raise ValueError(f"percentiles must be in (0,1): {percentiles}")
    if any(later < earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ValueError(f"values decrease as percentiles increase: {values}")

    # To the scaled axis, clipped into [0, 1].
    locations = [
        min(1.0, max(0.0, scaled_location(v, range_min, range_max, zero_point))) for v in values
    ]

    # Piecewise-linear F(location) through the elicited points, extended flat
    # beyond the first/last elicited location. Duplicate locations (values
    # clustered at one grid cell) keep the highest percentile at that location.
    anchor: dict[float, float] = {}
    for loc, pct in zip(locations, percentiles, strict=False):
        anchor[loc] = max(anchor.get(loc, 0.0), pct)
    xs = sorted(anchor)
    ys = [anchor[x] for x in xs]

    grid = [i / (n_points - 1) for i in range(n_points)]
    raw = [_interpolate(x, xs, ys) for x in grid]

    # Force monotone (interpolation already is; this is belt for float dust).
    for i in range(1, n_points):
        raw[i] = max(raw[i], raw[i - 1])

    # Rescale to span exactly [0, 1] inside the range...
    lo_raw, hi_raw = raw[0], raw[-1]
    span = hi_raw - lo_raw
    # span 0 = total ignorance; fall back to uniform rather than divide by it.
    base = list(grid) if span <= 0 else [(f - lo_raw) / span for f in raw]

    # ...then place that span between the bound offsets,
    lo = OPEN_BOUND_TAIL if open_lower_bound else 0.0
    hi = 1.0 - OPEN_BOUND_TAIL if open_upper_bound else 1.0
    placed = [lo + (hi - lo) * f for f in base]

    # ...blend in a ramp so every step clears the server minimum by construction,
    ramp = [lo + (hi - lo) * x for x in grid]
    blended = [(1.0 - RAMP_WEIGHT) * f + RAMP_WEIGHT * r for f, r in zip(placed, ramp, strict=True)]

    # ...and cap spikes, redistributing the excess pro-rata to uncapped steps.
    steps = [blended[i + 1] - blended[i] for i in range(n_points - 1)]
    steps = _cap_steps(steps, MAX_STEP)
    out = [blended[0]]
    for step in steps:
        out.append(out[-1] + step)
    out[-1] = hi  # exact endpoint, no float drift
    return out


def _interpolate(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            width = xs[i + 1] - xs[i]
            if width == 0:
                return ys[i + 1]
            t = (x - xs[i]) / width
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]  # pragma: no cover - unreachable


def _cap_steps(steps: list[float], cap: float) -> list[float]:
    """Cap each step at `cap`, redistributing excess pro-rata to uncapped steps.

    Total mass is preserved. Iterates because redistribution can push another
    step over the cap; bounded loop, then a final hard clip as belt-and-braces
    (only reachable if nearly all mass sits in capped steps, which the ramp
    blend prevents).
    """
    total = sum(steps)
    for _ in range(10):
        over = [i for i, s in enumerate(steps) if s > cap]
        if not over:
            break
        excess = sum(steps[i] - cap for i in over)
        for i in over:
            steps[i] = cap
        under = [i for i, s in enumerate(steps) if s < cap]
        if not under:
            break
        weight = sum(steps[i] for i in under)
        for i in under:
            share = steps[i] / weight if weight > 0 else 1.0 / len(under)
            steps[i] += excess * share
    # Preserve the total exactly (float drift from redistribution).
    drift = total - sum(steps)
    if steps:
        steps[-1] += drift
    return steps


def validate_cdf(
    cdf: list[float], *, open_lower_bound: bool, open_upper_bound: bool, n_points: int
) -> None:
    """Assert the full contract. Used by tests and by the orchestrator before
    any submission — a rejected CDF must be impossible, not merely unlikely."""
    if len(cdf) != n_points:
        raise ValueError(f"{len(cdf)} points, expected {n_points}")
    if any(not (0.0 <= v <= 1.0) for v in cdf):
        raise ValueError("cdf values outside [0, 1]")
    min_step = 0.01 / (n_points - 1)
    for i in range(1, n_points):
        if cdf[i] - cdf[i - 1] < min_step * 0.98:
            raise ValueError(f"step {i} below server minimum: {cdf[i] - cdf[i - 1]:.2e}")
    if not open_lower_bound and cdf[0] != 0.0:
        raise ValueError(f"closed lower bound but cdf[0]={cdf[0]}")
    if not open_upper_bound and cdf[-1] != 1.0:
        raise ValueError(f"closed upper bound but cdf[-1]={cdf[-1]}")
    if open_lower_bound and cdf[0] < OPEN_BOUND_TAIL * 0.5:
        raise ValueError(f"open lower bound but no tail mass: {cdf[0]}")
    if open_upper_bound and cdf[-1] > 1.0 - OPEN_BOUND_TAIL * 0.5:
        raise ValueError(f"open upper bound but no tail mass: {cdf[-1]}")
