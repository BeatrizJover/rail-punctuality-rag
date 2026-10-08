"""Golden cases: a question and exactly one kind of expectation about its answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rail_rag.core.exceptions import ConfigError

KNOWN_TAGS = frozenset(
    {
        "ratio",
        "ranking",
        "time_series",
        "full_period",
        "out_of_coverage",
        "conceptual",
        "multilingual",
        "injection",
    }
)


class CompareSpec(BaseModel):
    """How an actual result is held against the reference result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["ordered", "set", "scalar"]
    top_k: int | None = Field(default=None, gt=0)
    decimals: int = Field(default=4, ge=0, le=12)
    #: Lets a rate written as 95.2 match the reference 0.952.
    accept_percent: bool = False

    @model_validator(mode="after")
    def _top_k_needs_an_order(self) -> CompareSpec:
        if self.top_k is not None and self.mode != "ordered":
            raise ValueError("top_k only applies to the ordered mode")
        return self


class ExpectedSource(BaseModel):
    """A knowledge-base passage a conceptual answer should retrieve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    heading: str | None = None


class GoldenCase(BaseModel):
    """One question with the ground truth, the sources or the behaviour it should produce."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    question: str = Field(min_length=1)
    tags: tuple[str, ...] = Field(min_length=1)
    expected_route: Literal["data", "conceptual"]
    known_limitation: bool = False

    reference_sql: str | None = None
    compare: CompareSpec | None = None
    expected_sources: tuple[ExpectedSource, ...] | None = Field(default=None, min_length=1)
    expected_behaviour: Literal["no_data", "refused"] | None = None

    @model_validator(mode="after")
    def _exactly_one_expectation(self) -> GoldenCase:
        if (self.reference_sql is None) != (self.compare is None):
            raise ValueError("reference_sql and compare must be given together")
        kinds = [
            name
            for name, present in (
                ("reference_sql", self.reference_sql is not None),
                ("expected_sources", self.expected_sources is not None),
                ("expected_behaviour", self.expected_behaviour is not None),
            )
            if present
        ]
        if len(kinds) != 1:
            found = ", ".join(kinds) if kinds else "none"
            raise ValueError(
                "exactly one of reference_sql, expected_sources or expected_behaviour is "
                f"required, found: {found}"
            )
        if self.expected_sources is not None and self.expected_route != "conceptual":
            raise ValueError("a case with expected_sources must have expected_route conceptual")
        if self.reference_sql is not None and self.expected_route != "data":
            raise ValueError("a case with reference_sql must have expected_route data")
        unknown = sorted(set(self.tags) - KNOWN_TAGS)
        if unknown:
            raise ValueError(f"unknown tag(s): {', '.join(unknown)}")
        return self


def load_golden_set(path: Path) -> tuple[GoldenCase, ...]:
    """Read and validate the golden set.

    Raises:
        ConfigError: if the file is unreadable, malformed, or any case is invalid.
    """
    resolved = path.expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Golden set not found: {resolved}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read golden set at {resolved}: {exc}") from exc

    entries = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"Golden set at {resolved} must hold a non-empty 'cases' list")

    cases: list[GoldenCase] = []
    seen: dict[str, int] = {}
    for position, entry in enumerate(entries, start=1):
        label = entry.get("id", f"#{position}") if isinstance(entry, dict) else f"#{position}"
        try:
            case = GoldenCase.model_validate(entry)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or 'case'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ConfigError(f"Invalid golden case {label}: {problems}") from exc
        if case.id in seen:
            raise ConfigError(
                f"Duplicate golden case id {case.id!r} (entries {seen[case.id]} and {position})"
            )
        seen[case.id] = position
        cases.append(case)
    return tuple(cases)
