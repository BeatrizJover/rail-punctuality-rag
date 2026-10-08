"""Settings for an evaluation run, kept in a committed YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rail_rag.core.exceptions import ConfigError

DEFAULT_EVAL_CONFIG = Path("config/eval_config.yaml")


class EvalConfig(BaseModel):
    """Root of ``config/eval_config.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Below the provider's limit, because a retry also counts as a request.
    generation_rpm: int = Field(default=4, gt=0)
    daily_generation_budget: int = Field(default=15, gt=0)
    cache: bool = True
    golden_set: Path = Path("eval/golden_set.yaml")
    cache_dir: Path = Path(".eval_cache")

    @property
    def min_interval_s(self) -> float:
        return 60.0 / self.generation_rpm

    @property
    def expected_dir(self) -> Path:
        return self.cache_dir / "expected"

    @property
    def generations_dir(self) -> Path:
        return self.cache_dir / "generations"

    @property
    def usage_dir(self) -> Path:
        return self.cache_dir / "usage"


def load_eval_config(path: Path = DEFAULT_EVAL_CONFIG) -> EvalConfig:
    """Read and validate the evaluation configuration.

    Raises:
        ConfigError: if the file is missing, unreadable, not a mapping, or invalid.
    """
    resolved = path.expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Evaluation configuration not found: {resolved}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read evaluation configuration at {resolved}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Evaluation configuration at {resolved} must be a mapping")

    try:
        return EvalConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid evaluation configuration at {resolved}: {exc}") from exc
