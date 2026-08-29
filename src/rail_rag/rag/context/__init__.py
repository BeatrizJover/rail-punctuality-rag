"""Deterministic context assembled for the SQL generator.

Three parts, in the order the model reads them: what the tables are, how their
measures must be combined, and what the data actually contains. None of it
involves a language model, so the same question always starts from the same
ground truth.
"""

from sqlalchemy import Engine

from rail_rag.rag.context.profile import DataProfile, load_profile
from rail_rag.rag.context.rules import SEMANTIC_RULES
from rail_rag.rag.context.schema import render_schema
from rail_rag.rag.sql.policy import SqlPolicy

__all__ = [
    "SEMANTIC_RULES",
    "DataProfile",
    "build_context",
    "load_profile",
    "render_context",
    "render_schema",
]


def render_context(policy: SqlPolicy, profile: DataProfile) -> str:
    """Assemble the full context block from parts already computed."""
    return "\n\n".join(
        (
            "SCHEMA",
            render_schema(policy),
            SEMANTIC_RULES.strip(),
            profile.render(),
        )
    )


def build_context(engine: Engine, policy: SqlPolicy) -> str:
    """Measure the database and return the complete context block.

    Raises:
        DatabaseError: if the database cannot be profiled.
        RagError: if the policy and the schema have diverged.
    """
    return render_context(policy, load_profile(engine))
