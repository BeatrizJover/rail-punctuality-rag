"""Tests for prompt assembly and reply parsing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rail_rag.rag.exceptions import AnswerError
from rail_rag.rag.prompts import (
    MAX_QUOTED_ROWS,
    NO_SQL,
    build_answer_prompt,
    build_conceptual_prompt,
    build_repair_prompt,
    build_sql_prompt,
    build_sql_system,
    extract_sql,
    render_passages,
)


@dataclass(frozen=True)
class StubPassage:
    """Matches the structural ``Passage`` protocol without importing the store."""

    doc_id: str
    heading: str | None
    content: str


PASSAGE = StubPassage("03-punctuality", "The threshold", "Under 360 seconds.")


def test_a_fenced_block_yields_only_the_query() -> None:
    reply = "Here you go:\n```sql\nSELECT 1 FROM gold.dim_date\n```\nHope that helps."
    assert extract_sql(reply) == "SELECT 1 FROM gold.dim_date"


def test_an_unfenced_query_is_still_accepted() -> None:
    assert extract_sql("SELECT count(*) FROM gold.fact_stop_event") == (
        "SELECT count(*) FROM gold.fact_stop_event"
    )


def test_a_leading_cte_counts_as_a_query() -> None:
    reply = "```sql\nWITH x AS (SELECT 1) SELECT * FROM x\n```"
    assert extract_sql(reply) is not None


def test_the_trailing_semicolon_is_stripped() -> None:
    # The guard treats a second statement as fatal, so the terminator is removed
    # here rather than relying on sqlglot forgiving it.
    assert extract_sql("SELECT 1;") == "SELECT 1"


def test_the_sentinel_means_no_query_rather_than_a_failure() -> None:
    assert extract_sql(NO_SQL) is None


def test_a_fenced_sentinel_is_still_a_refusal() -> None:
    assert extract_sql(f"```\n{NO_SQL}\n```") is None


def test_a_sentence_merely_mentioning_the_sentinel_is_malformed() -> None:
    # Otherwise a model narrating its reasoning would be read as a refusal.
    with pytest.raises(AnswerError):
        extract_sql(f"I think the answer here is {NO_SQL} because it is conceptual.")


def test_prose_without_a_query_is_rejected() -> None:
    with pytest.raises(AnswerError):
        extract_sql("Punctuality is defined as arriving under six minutes late.")


def test_an_empty_reply_is_rejected() -> None:
    with pytest.raises(AnswerError):
        extract_sql("   \n  ")


def test_a_write_statement_is_not_silently_accepted_as_a_query() -> None:
    # The guard would reject it anyway; failing here keeps the error close to
    # the contract that was broken.
    with pytest.raises(AnswerError):
        extract_sql("```sql\nDELETE FROM gold.fact_stop_event\n```")


def test_the_system_prompt_carries_the_context_and_the_sentinel() -> None:
    system = build_sql_system("SCHEMA\nTABLE gold.dim_date")
    assert "TABLE gold.dim_date" in system
    assert NO_SQL in system


def test_passages_are_numbered_and_titled() -> None:
    rendered = render_passages([PASSAGE])
    assert "[1] 03-punctuality / The threshold" in rendered
    assert "Under 360 seconds." in rendered


def test_a_passage_without_a_heading_renders_its_document() -> None:
    rendered = render_passages([StubPassage("01-data-source", None, "Infrabel.")])
    assert "[1] 01-data-source" in rendered


def test_no_passages_says_so_rather_than_rendering_nothing() -> None:
    # An empty block reads as a formatting bug to the model; a stated absence
    # does not.
    assert render_passages([]) != ""


def test_the_sql_prompt_contains_the_question_and_the_passages() -> None:
    prompt = build_sql_prompt("Which station was worst?", [PASSAGE])
    assert "Which station was worst?" in prompt
    assert "The threshold" in prompt


def test_the_repair_prompt_quotes_the_rejection_verbatim() -> None:
    prompt = build_repair_prompt("q", "SELECT 1", "Table(s) not allowed: gold.stg_fact_stop_event")
    assert "Table(s) not allowed: gold.stg_fact_stop_event" in prompt
    assert "SELECT 1" in prompt


def test_the_answer_prompt_renders_rows_and_marks_a_truncated_result() -> None:
    prompt = build_answer_prompt(
        "q", ["station", "rate"], [("Gent", 0.91)], truncated=True, passages=[]
    )
    assert "station | rate" in prompt
    assert "Gent | 0.91" in prompt
    assert "capped" in prompt


def test_a_null_cell_renders_as_empty_rather_than_as_the_word_none() -> None:
    prompt = build_answer_prompt("q", ["ptcar_no"], [(None,)], truncated=False)
    assert "None" not in prompt


def test_long_results_are_capped_in_the_prompt_but_the_total_is_stated() -> None:
    rows = [(index,) for index in range(MAX_QUOTED_ROWS + 5)]
    prompt = build_answer_prompt("q", ["n"], rows, truncated=False)
    assert f"{MAX_QUOTED_ROWS + 5} rows in total" in prompt
    assert str(MAX_QUOTED_ROWS + 4) not in prompt


def test_the_conceptual_prompt_carries_the_passages() -> None:
    prompt = build_conceptual_prompt("Why is ptcar_no null?", [PASSAGE])
    assert "Why is ptcar_no null?" in prompt
    assert "03-punctuality" in prompt
