"""Deciding whether a question needs a query or a passage.

The routing bar is deliberately low. A misrouted question degrades the answer
but cannot poison it: a conceptual question sent down the data path is refused
by the generator with the ``NO_SQL`` sentinel and falls back, and a data
question sent down the conceptual path is answered from documentation that says
what the measure means rather than what its value is. Neither produces a wrong
number.

Because of that, the cheap path runs first: lexical markers decide most
questions with no model call at all. The generator is consulted only when the
markers disagree or find nothing, and its failure is not fatal — an unreachable
provider falls through to the default rather than aborting the answer.

The markers are English because the corpus and the schema are English. A
question in another language usually falls through to the model call, which
handles it; this is a known limitation rather than an oversight.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from rail_rag.rag.exceptions import ProviderError
from rail_rag.rag.prompts import ROUTER_INSTRUCTIONS
from rail_rag.rag.providers.base import TextGenerator

logger = logging.getLogger(__name__)


class Route(Enum):
    """Which of the two answering paths a question takes."""

    DATA = "data"
    CONCEPTUAL = "conceptual"


#: Chosen when nothing else decides. The data path is the safer default because
#: it can refuse and fall back to the conceptual one; the conceptual path cannot
#: discover halfway through that it needed a number.
DEFAULT_ROUTE = Route.DATA

#: Phrases that only make sense about rows: counts, rankings, periods, extremes.
_DATA_MARKERS: tuple[str, ...] = (
    "how many",
    "how much",
    "which station",
    "which stations",
    "which train",
    "which relation",
    "which operator",
    "worst",
    "best",
    "highest",
    "lowest",
    "most delayed",
    "least",
    "top ",
    "rank",
    "average delay",
    "total",
    "count",
    "compare",
    "trend",
    "per day",
    "per hour",
    "per station",
    "by station",
    "by hour",
    "by month",
    "last week",
    "last month",
    "list ",
    "show me",
)

#: Phrases about the system rather than its contents.
_CONCEPTUAL_MARKERS: tuple[str, ...] = (
    "why is",
    "why are",
    "why does",
    "why do",
    "what does it mean",
    "what is meant",
    "how is punctuality",
    "how do you define",
    "how is it defined",
    "what does",
    "how does the",
    "how was",
    "where does the data",
    "who publishes",
    "explain",
    "definition",
    "architecture",
    "limitation",
    "known issue",
    "modelled",
    "modeled",
    "design",
)

#: A bare year or an ISO date is a strong signal that a period is being asked about.
_PERIOD = re.compile(r"\b(19|20)\d{2}\b")


def classify(question: str, generator: TextGenerator | None = None) -> Route:
    """Route a question, consulting the model only when the markers cannot.

    A provider failure is logged and swallowed: routing is an optimisation, and
    refusing to answer because a classification call timed out would trade a
    slightly worse answer for no answer at all.
    """
    lexical = classify_lexically(question)
    if lexical is not None:
        logger.debug("routed %r lexically to %s", question, lexical.value)
        return lexical
    if generator is None:
        return DEFAULT_ROUTE
    try:
        decided = _classify_with_model(question, generator)
    except ProviderError:
        logger.warning("router provider failed; falling back to %s", DEFAULT_ROUTE.value)
        return DEFAULT_ROUTE
    return decided if decided is not None else DEFAULT_ROUTE


def classify_lexically(question: str) -> Route | None:
    """Return a route when the wording is unambiguous, else ``None``.

    Ambiguity is real rather than a shrug: "what does punctual mean for the
    trains in August" carries markers of both kinds, and guessing between them
    is worse than paying for one short model call.
    """
    lowered = question.lower()
    data = sum(marker in lowered for marker in _DATA_MARKERS)
    if _PERIOD.search(lowered):
        data += 1
    conceptual = sum(marker in lowered for marker in _CONCEPTUAL_MARKERS)
    if data and not conceptual:
        return Route.DATA
    if conceptual and not data:
        return Route.CONCEPTUAL
    return None


def _classify_with_model(question: str, generator: TextGenerator) -> Route | None:
    """Ask for one word and accept nothing else."""
    reply = generator.generate(system=ROUTER_INSTRUCTIONS, prompt=question)
    # Only the first word is read: a model that adds a justification has still
    # answered, and a model that answers something else entirely has not.
    head = reply.strip().split(maxsplit=1)[0].strip(".:,").upper() if reply.strip() else ""
    if head == "DATA":
        return Route.DATA
    if head == "CONCEPTUAL":
        return Route.CONCEPTUAL
    logger.warning("router returned an unusable reply: %r", reply[:80])
    return None
