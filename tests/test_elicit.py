"""Tests for prompts and parsers.

The prompt tests pin the presence of the guards that carry measured evidence —
if someone edits a prompt and drops the already-resolved guard or the units
line, a test names exactly what was lost and why it was there.
"""

from __future__ import annotations

import pytest

from bot.elicit import (
    NUMERIC_PERCENTILES,
    ParseError,
    binary_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    parse_binary,
    parse_extreme_check,
    parse_multiple_choice,
    parse_numeric,
)
from bot.venues.metaculus import Question


def _question(qtype: str = "binary", **overrides: object) -> Question:
    base: dict = {
        "post_id": 45163,
        "question_id": 44120,
        "title": "Will X happen by 2026-09-01?",
        "qtype": qtype,
        "description": "Background text.",
        "resolution_criteria": "Resolves YES if X is reported by the source.",
        "fine_print": "Fine print text.",
        "options": (),
        "open_time": "2026-08-10T14:00:00Z",
        "close_time": "2026-08-22T14:00:00Z",
        "unit": "",
        "range_min": None,
        "range_max": None,
        "open_lower_bound": None,
        "open_upper_bound": None,
        "zero_point": None,
    }
    base.update(overrides)
    return Question(**base)


TODAY = "2026-08-18"


def test_binary_prompt_carries_the_guards() -> None:
    prompt = binary_prompt(_question(), "research text", TODAY)
    assert "has NOT yet resolved" in prompt  # the catastrophic 99% trap
    assert "status quo" in prompt.lower()  # status-quo weighting
    assert "base rate" in prompt.lower()  # r=+0.38 within winners
    assert TODAY in prompt  # assumed start date confusion
    assert "Resolves YES if X" in prompt  # resolution criteria verbatim
    assert "Bayesian" not in prompt  # measured underperformer


def test_numeric_prompt_states_unit_and_bounds() -> None:
    question = _question(
        qtype="numeric",
        unit="$/bbl",
        range_min=55.0,
        range_max=130.0,
        open_lower_bound=False,
        open_upper_bound=True,
    )
    prompt = numeric_prompt(question, "research", TODAY)
    assert "$/bbl" in prompt
    assert "cannot be below 55.0" in prompt
    assert "above 130.0 are possible" in prompt
    assert "TOO NARROW" in prompt  # the tails lesson


def test_mc_prompt_lists_options_in_order() -> None:
    question = _question(qtype="multiple_choice", options=("Texas", "Arizona", "Other"))
    prompt = multiple_choice_prompt(question, "research", TODAY)
    assert prompt.index("- Texas") < prompt.index("- Arizona") < prompt.index("- Other")


def test_parse_binary_takes_the_final_answer_line() -> None:
    text = "Base rate is about 30%. The trend suggests 45%.\n\nProbability: 37.5%"
    assert parse_binary(text) == pytest.approx(0.375)


def test_parse_binary_tolerates_markdown_bold() -> None:
    assert parse_binary("**Probability: 12%**") == pytest.approx(0.12)


def test_parse_binary_with_no_percentage_raises() -> None:
    with pytest.raises(ParseError):
        parse_binary("I cannot answer this question.")


def test_parse_mc_requires_every_option() -> None:
    """A missing option means the model answered a different question —
    normalising around the hole would hide that."""
    options = ("Texas", "Arizona", "Other")
    text = "Texas: 20%\nArizona: 50%\nOther: 30%"
    parsed = parse_multiple_choice(text, options)
    assert parsed["Arizona"] == pytest.approx(0.5)
    assert sum(parsed.values()) == pytest.approx(1.0)
    with pytest.raises(ParseError):
        parse_multiple_choice("Texas: 60%\nArizona: 40%", options)


def test_parse_mc_normalises_sloppy_sums() -> None:
    options = ("A", "B")
    parsed = parse_multiple_choice("A: 60%\nB: 60%", options)
    assert sum(parsed.values()) == pytest.approx(1.0)
    assert parsed["A"] == pytest.approx(0.5)


def test_parse_numeric_strips_commas_and_requires_all_percentiles() -> None:
    lines = "\n".join(
        f"Percentile {int(p * 100)}: {1_000 * (i + 1):,}" for i, p in enumerate(NUMERIC_PERCENTILES)
    )
    parsed = parse_numeric(lines)
    assert parsed[0.05] == 1000.0
    assert parsed[0.95] == 7000.0
    with pytest.raises(ParseError):
        parse_numeric("Percentile 5: 100\nPercentile 50: 200")


def test_parse_numeric_rejects_decreasing_values() -> None:
    lines = "\n".join(
        f"Percentile {int(p * 100)}: {v}"
        for p, v in zip(NUMERIC_PERCENTILES, [10, 20, 30, 25, 40, 50, 60], strict=True)
    )
    with pytest.raises(ParseError):
        parse_numeric(lines)


def test_extreme_check_verdicts() -> None:
    assert parse_extreme_check("SUSPECT\nThe question may already have resolved...") is True
    assert parse_extreme_check("SOUND\nCriteria read correctly; near-mechanical.") is False
    assert parse_extreme_check("") is False
