"""Evaluation of the answering pipeline against a golden set."""

from rail_rag.eval.cases import (
    CompareSpec,
    ExpectedSource,
    GoldenCase,
    load_golden_set,
)
from rail_rag.eval.compare import compare_results, hit_at_k, percentile, reciprocal_rank
from rail_rag.eval.expected import ExpectedResult, compute_expected, fact_window

__all__ = [
    "CompareSpec",
    "ExpectedResult",
    "ExpectedSource",
    "GoldenCase",
    "compare_results",
    "compute_expected",
    "fact_window",
    "hit_at_k",
    "load_golden_set",
    "percentile",
    "reciprocal_rank",
]
