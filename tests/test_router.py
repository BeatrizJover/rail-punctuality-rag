"""Tests for question routing."""

from __future__ import annotations

from rail_rag.rag.exceptions import ProviderError
from rail_rag.rag.providers.fake import FakeGenerator
from rail_rag.rag.router import DEFAULT_ROUTE, Route, classify, classify_lexically


class FailingGenerator:
    """A provider that is down, to prove routing survives it."""

    def generate(self, *, system: str, prompt: str) -> str:
        raise ProviderError("unreachable")


def test_a_counting_question_is_data() -> None:
    assert classify_lexically("How many trains ran last month?") is Route.DATA


def test_a_superlative_question_is_data() -> None:
    assert classify_lexically("Which station had the worst punctuality?") is Route.DATA


def test_a_bare_year_is_enough_to_suggest_a_period() -> None:
    assert classify_lexically("Punctuality per station in 2025") is Route.DATA


def test_a_year_alone_routes_to_data() -> None:
    # No other marker here: the period is the whole signal.
    assert classify_lexically("Punctuality at Gent in 2025") is Route.DATA


def test_a_definition_question_is_conceptual() -> None:
    assert classify_lexically("How do you define punctuality here?") is Route.CONCEPTUAL


def test_a_why_question_is_conceptual() -> None:
    assert classify_lexically("Why is ptcar_no null for most stations?") is Route.CONCEPTUAL


def test_a_question_carrying_both_kinds_of_marker_is_undecided() -> None:
    question = "Explain how many stations there are"
    assert classify_lexically(question) is None


def test_a_question_carrying_no_marker_is_undecided() -> None:
    assert classify_lexically("Tell me about Gent-Sint-Pieters") is None


def test_an_undecided_question_reaches_the_model() -> None:
    generator = FakeGenerator(["CONCEPTUAL"])
    assert classify("Tell me about Gent-Sint-Pieters", generator) is Route.CONCEPTUAL
    assert len(generator.calls) == 1


def test_a_decided_question_never_reaches_the_model() -> None:
    # The whole point of the lexical pass is that it costs nothing.
    generator = FakeGenerator(["CONCEPTUAL"])
    assert classify("How many trains ran in 2025?", generator) is Route.DATA
    assert generator.calls == []


def test_a_verbose_model_reply_is_still_read() -> None:
    generator = FakeGenerator(["DATA — it asks for a figure."])
    assert classify("Tell me about Gent", generator) is Route.DATA


def test_an_unusable_model_reply_falls_back_to_the_default() -> None:
    generator = FakeGenerator(["I am not sure, possibly both."])
    assert classify("Tell me about Gent", generator) is DEFAULT_ROUTE


def test_a_failing_provider_does_not_abort_the_answer() -> None:
    # Routing is an optimisation; losing it must not cost the user their answer.
    assert classify("Tell me about Gent", FailingGenerator()) is DEFAULT_ROUTE


def test_no_generator_means_no_call_and_the_default() -> None:
    assert classify("Tell me about Gent") is DEFAULT_ROUTE


def test_the_default_is_the_recoverable_path() -> None:
    # DATA can refuse with the sentinel and fall back; CONCEPTUAL cannot discover
    # halfway through that it needed a number.
    assert DEFAULT_ROUTE is Route.DATA
