"""Evaluation of the answering pipeline against a golden set."""

from rail_rag.eval.budget import BudgetedGenerator, BudgetExhausted, DailyBudget, RateLimiter
from rail_rag.eval.cache import CachingGenerator, GeneratorFingerprint
from rail_rag.eval.cases import (
    CompareSpec,
    ExpectedSource,
    GoldenCase,
    load_golden_set,
)
from rail_rag.eval.compare import compare_results, hit_at_k, percentile, reciprocal_rank
from rail_rag.eval.config import EvalConfig, load_eval_config
from rail_rag.eval.expected import ExpectedResult, compute_expected, fact_window
from rail_rag.eval.runner import (
    CaseResult,
    RunReport,
    completed_ids,
    estimate_calls,
    run_cases,
)

__all__ = [
    "BudgetExhausted",
    "BudgetedGenerator",
    "CachingGenerator",
    "CaseResult",
    "CompareSpec",
    "DailyBudget",
    "EvalConfig",
    "ExpectedResult",
    "ExpectedSource",
    "GeneratorFingerprint",
    "GoldenCase",
    "RateLimiter",
    "RunReport",
    "compare_results",
    "completed_ids",
    "compute_expected",
    "estimate_calls",
    "fact_window",
    "hit_at_k",
    "load_eval_config",
    "load_golden_set",
    "percentile",
    "reciprocal_rank",
    "run_cases",
]
