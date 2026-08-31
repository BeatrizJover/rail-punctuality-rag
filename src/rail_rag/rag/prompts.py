"""Prompt text and the parsing of what comes back.

This module is deliberately free of engines, providers and configuration: it
turns values into strings and strings into values. Keeping it that way is what
lets the wording be tested without a database, a model or a network call, which
matters because prompt text is the part of the system most likely to be edited
casually.

Two conventions are load-bearing and repeated in every instruction below:

* SQL is returned inside a fenced ``sql`` block, so the answer can carry
  reasoning without the reasoning being executed;
* a question that no query can answer is refused with the ``NO_SQL`` sentinel
  rather than with an invented query. The router upstream is a heuristic and
  will sometimes send a conceptual question down the data path; the sentinel is
  what turns that mistake into a fallback instead of a fabricated result.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from rail_rag.rag.exceptions import AnswerError

#: What the generator must return when the question needs no SQL.
NO_SQL = "NO_SQL"

#: Longest passage body included in a prompt; the corpus chunks are smaller than
#: this, so it only ever bites if the chunker's limits are raised later.
_MAX_PASSAGE_CHARS = 2000

#: Rows quoted back to the model when it writes the final answer. The full result
#: is returned separately: the caller renders the table, the model only narrates.
MAX_QUOTED_ROWS = 30

_FENCED_SQL = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_STARTS_A_QUERY = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


class Passage(Protocol):
    """The shape of a retrieved chunk, as prompts need it.

    Structural rather than an import of ``RetrievedChunk``: this module has no
    business depending on the storage layer to format a string.
    """

    @property
    def doc_id(self) -> str: ...

    @property
    def heading(self) -> str | None: ...

    @property
    def content(self) -> str: ...


SQL_INSTRUCTIONS = f"""\
You write PostgreSQL SELECT queries against the schema described above, for a \
system that answers questions about Belgian railway punctuality.

RULES
- Return exactly one SELECT statement, inside a fenced block: ```sql ... ```
- Read only. No INSERT, UPDATE, DELETE, DDL, CTE that writes, or SELECT INTO.
- Query only the tables listed in the schema above. Staging and operational \
tables do not exist as far as you are concerned.
- Resolve every surrogate key to its human-readable name by joining the \
dimension. A hash in a result set is not an answer.
- Obey the measure rules above exactly, in particular the punctuality ratio.
- Do not explain the query. The fenced block is the whole reply.

WHEN NOT TO WRITE SQL
- If the question is about how the system works, how a term is defined, or why \
a modelling decision was made, no query answers it. Reply with exactly \
{NO_SQL} and nothing else.
- If the question asks for a period the coverage section above says has no \
data, still write the query: the empty result is handled by the caller.
"""

ANSWER_INSTRUCTIONS = """\
You explain the result of a query to the person who asked the question.

RULES
- Answer the question directly, in the language the question was asked in.
- Use only the numbers given below. Never estimate, extrapolate or round a \
figure into a different one.
- Two or three sentences. The full result table is shown to the user \
separately, so do not reproduce it row by row.
- If the rows are marked as truncated, say the answer is based on a capped \
number of rows.
"""

CONCEPTUAL_INSTRUCTIONS = """\
You answer questions about how this system and its data are built, using only \
the documentation passages given below.

RULES
- Answer in the language the question was asked in.
- Use only the passages. If they do not contain the answer, say so plainly \
rather than filling the gap from general knowledge.
- Refer to the source documents by their names when it helps the reader find \
more detail.
- Be brief. A paragraph is usually enough.
"""

ROUTER_INSTRUCTIONS = """\
Classify a question about a Belgian railway punctuality database.

Answer with exactly one word:
- DATA if answering requires counting, aggregating or listing rows: figures, \
rankings, comparisons, trends, "how many", "which station", "when".
- CONCEPTUAL if it asks how the system works, how a term is defined, why \
something was modelled a certain way, or what a column means.

One word. No punctuation, no explanation.
"""


def render_passages(passages: Sequence[Passage]) -> str:
    """Format retrieved chunks for a prompt, numbered so they can be referred to."""
    if not passages:
        return "(no relevant documentation was retrieved)"
    blocks = []
    for index, passage in enumerate(passages, start=1):
        title = f"{passage.doc_id} / {passage.heading}" if passage.heading else passage.doc_id
        body = passage.content[:_MAX_PASSAGE_CHARS]
        blocks.append(f"[{index}] {title}\n{body}")
    return "\n\n".join(blocks)


def build_sql_system(context: str) -> str:
    """The system instruction for SQL generation: ground truth, then the rules."""
    return f"{context}\n\n{SQL_INSTRUCTIONS}"


def build_sql_prompt(question: str, passages: Sequence[Passage]) -> str:
    """The user turn for SQL generation: the question plus best-effort context."""
    documentation = render_passages(passages)
    return f"DOCUMENTATION THAT MAY BE RELEVANT\n{documentation}\n\nQUESTION\n{question}"


def build_repair_prompt(question: str, sql: str, error: str) -> str:
    """The single follow-up turn after the guard rejects a query.

    The rejection reason is quoted verbatim: it names the rule that was broken,
    which is more useful to the model than a paraphrase of it.
    """
    return (
        "The query you wrote was rejected before it reached the database.\n\n"
        f"QUERY\n{sql}\n\n"
        f"REJECTION\n{error}\n\n"
        "Rewrite it so it satisfies the rules. Same output format: one fenced "
        f"sql block, or {NO_SQL} if no query can answer the question.\n\n"
        f"QUESTION\n{question}"
    )


def build_answer_prompt(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    truncated: bool,
    passages: Sequence[Passage] = (),
) -> str:
    """The user turn that turns a result set into prose."""
    quoted = rows[:MAX_QUOTED_ROWS]
    lines = [" | ".join(columns)]
    lines.extend(" | ".join("" if value is None else str(value) for value in row) for row in quoted)
    note = ""
    if truncated:
        note = "\n(the result was capped by a row limit)"
    elif len(rows) > len(quoted):
        note = f"\n({len(rows)} rows in total, {len(quoted)} shown here)"
    parts = [f"QUESTION\n{question}", "RESULT\n" + "\n".join(lines) + note]
    if passages:
        parts.append(f"BACKGROUND\n{render_passages(passages)}")
    return "\n\n".join(parts)


def build_conceptual_prompt(question: str, passages: Sequence[Passage]) -> str:
    """The user turn for a question answered from documentation alone."""
    return f"PASSAGES\n{render_passages(passages)}\n\nQUESTION\n{question}"


def extract_sql(reply: str) -> str | None:
    """Pull the query out of a generator reply.

    Returns:
        The SQL, or ``None`` when the model declined with the sentinel.

    Raises:
        AnswerError: if the reply contains neither a query nor the sentinel,
            which means the model ignored the output contract.
    """
    stripped = reply.strip()
    if not stripped:
        raise AnswerError("The model returned an empty reply")

    match = _FENCED_SQL.search(stripped)
    candidate = match.group(1).strip() if match else stripped

    # Checked after unfencing: a model that wraps the sentinel in a block is
    # still declining, and the sentinel must not be mistaken for a query.
    if _is_refusal(candidate):
        return None
    if not _STARTS_A_QUERY.match(candidate):
        raise AnswerError("The model returned neither a SELECT query nor a refusal")
    return candidate.rstrip().rstrip(";")


def _is_refusal(candidate: str) -> bool:
    """True when the reply is the sentinel rather than a query.

    Bounded on purpose: an unfenced sentence that merely mentions ``NO_SQL``
    somewhere is a malformed reply, not a refusal, and should be caught as one.
    """
    return candidate.strip().strip(".").upper() == NO_SQL
