"""Tests for the Metaculus venue client.

Each test is named for the specific failure it forbids. The theme, as everywhere
in this repo, is the failure that looks like success: a forecast that silently
went to the wrong question, got clamped to a probability nobody chose, or never
went anywhere at all while the run reported clean.
"""

from __future__ import annotations

import pytest

from bot.venues.metaculus import (
    CDF_POINTS,
    binary_payload,
    multiple_choice_payload,
    next_page_url,
    numeric_payload,
    only_open,
    parse_posts,
)

BINARY_POST = {
    "id": 45163,
    "title": "Will the 2026 Atlantic hurricane season have at least 3 named storms?",
    "question": {
        "id": 44120,
        "type": "binary",
        "title": "Will the 2026 Atlantic hurricane season have at least 3 named storms?",
        "description": "Some background.",
        "resolution_criteria": "Resolves YES if NHC names a third storm.",
        "fine_print": "Small print.",
        "open_time": "2026-08-10T14:00:00Z",
        "scheduled_close_time": "2026-08-22T14:00:00Z",
    },
}

MC_POST = {
    "id": 45200,
    "title": "What state will have the hottest temperature?",
    "question": {
        "id": 44160,
        "type": "multiple_choice",
        "options": ["Texas", "Arizona", "Nevada", "Other"],
        "scheduled_close_time": "2026-08-31T14:00:00Z",
    },
}

NUMERIC_POST = {
    "id": 45210,
    "title": "What will Brent close at?",
    "question": {
        "id": 44170,
        "type": "numeric",
        "unit": "$/bbl",
        "open_upper_bound": True,
        "open_lower_bound": False,
        "scaling": {"range_min": 55.0, "range_max": 130.0, "zero_point": None},
    },
}

GROUP_POST = {
    "id": 45220,
    "title": "Group: named storms by month",
    "group_of_questions": {
        "questions": [
            {"id": 44180, "type": "binary", "title": "By September?"},
            {"id": 44181, "type": "binary", "title": "By October?"},
        ]
    },
}

NOTEBOOK_POST = {"id": 45230, "title": "Tournament announcement", "notebook": {"id": 9}}


def test_parse_flattens_single_group_and_skips_notebooks() -> None:
    """A notebook parsed as a question would get a forecast POSTed at nothing;
    a group post parsed as one question would drop its siblings silently."""
    payload = {"results": [BINARY_POST, MC_POST, NUMERIC_POST, GROUP_POST, NOTEBOOK_POST]}
    questions = parse_posts(payload)
    assert [q.question_id for q in questions] == [44120, 44160, 44170, 44180, 44181]
    assert all(q.post_id for q in questions)


def test_parse_keeps_server_type_string_verbatim() -> None:
    """An unknown type must arrive unmodified so the orchestrator can refuse it,
    rather than being coerced into a family it does not belong to."""
    post = {"id": 1, "question": {"id": 2, "type": "date"}}
    (question,) = parse_posts({"results": [post]})
    assert question.qtype == "date"


def test_parse_reads_numeric_scaling() -> None:
    (question,) = parse_posts({"results": [NUMERIC_POST]})
    assert question.range_min == 55.0
    assert question.range_max == 130.0
    assert question.open_upper_bound is True
    assert question.open_lower_bound is False
    assert question.unit == "$/bbl"


def test_binary_payload_shape() -> None:
    payload = binary_payload(44120, 0.58)
    assert payload == {
        "question": 44120,
        "probability_yes": 0.58,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }


@pytest.mark.parametrize("p", [0.0, 0.005, 0.995, 1.0, -0.2, 1.7])
def test_binary_payload_refuses_tails_rather_than_clamping(p: float) -> None:
    """A clamped probability is a forecast nobody made. Refusing is loud;
    clamping would quietly rewrite the bot's judgment at the wire."""
    with pytest.raises(ValueError):
        binary_payload(44120, p)


def test_mc_payload_requires_exact_option_match() -> None:
    """A missing or extra key means the caller answered a different question —
    the server may even accept it, which is exactly why it must not leave here."""
    options = ["Texas", "Arizona", "Nevada", "Other"]
    good = {"Texas": 0.2, "Arizona": 0.4, "Nevada": 0.15, "Other": 0.25}
    assert multiple_choice_payload(44160, good, options)["probability_yes_per_category"] == {
        "Texas": 0.2,
        "Arizona": 0.4,
        "Nevada": 0.15,
        "Other": 0.25,
    }
    with pytest.raises(ValueError):
        multiple_choice_payload(44160, {k: v for k, v in good.items() if k != "Other"}, options)
    with pytest.raises(ValueError):
        multiple_choice_payload(44160, {**good, "Utah": 0.0}, options)


def test_mc_payload_requires_probabilities_that_sum_to_one() -> None:
    options = ["A", "B"]
    with pytest.raises(ValueError):
        multiple_choice_payload(1, {"A": 0.5, "B": 0.4}, options)


def test_numeric_payload_validates_cdf() -> None:
    rising = [i / (CDF_POINTS - 1) for i in range(CDF_POINTS)]
    assert numeric_payload(44170, rising)["continuous_cdf"] == rising
    with pytest.raises(ValueError):
        numeric_payload(44170, rising[:-1])  # wrong length
    broken = list(rising)
    broken[100], broken[101] = broken[101], broken[100]
    with pytest.raises(ValueError):
        numeric_payload(44170, broken)  # decreasing step
    with pytest.raises(ValueError):
        numeric_payload(44170, [v * 1.5 for v in rising])  # exceeds 1


def test_resolved_members_of_an_open_group_are_filtered() -> None:
    """Observed live 2026-08-18: post 43325 was open while two of its four
    sub-questions were resolved; forecasting those returned HTTP 405. The
    post-level statuses=open filter cannot see this — the question's own
    status decides."""
    group = {
        "id": 43325,
        "title": "Net worth thresholds",
        "group_of_questions": {
            "questions": [
                {"id": 43330, "type": "binary", "status": "open"},
                {"id": 43327, "type": "binary", "status": "resolved"},
                {"id": 43329, "type": "binary", "status": "open"},
                {"id": 43328, "type": "binary", "status": "resolved"},
            ]
        },
    }
    questions = only_open(parse_posts({"results": [group]}))
    assert [q.question_id for q in questions] == [43330, 43329]


def test_empty_page_ends_pagination_even_when_next_is_set() -> None:
    """Observed live 2026-08-18: an empty `statuses=open` page still carries a
    `next` URL, at every offset, forever. Following it walked to offset 800 and
    into the rate limiter. Empty results end the walk, whatever `next` says."""
    assert next_page_url({"results": [], "next": "https://x/api/posts/?offset=3"}) is None


def test_populated_page_paginates_and_last_page_stops() -> None:
    assert next_page_url({"results": [{"id": 1}], "next": "https://x/next"}) == "https://x/next"
    assert next_page_url({"results": [{"id": 1}], "next": None}) is None


def test_empty_page_parses_to_no_questions() -> None:
    """Prove the null can fire: an empty tournament reads as zero questions,
    not as a crash and not as fabricated rows."""
    assert parse_posts({"results": []}) == []
