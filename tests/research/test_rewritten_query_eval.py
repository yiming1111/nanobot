from __future__ import annotations

import pytest

from benchmarks.paper_research.run_rewritten_query_eval import (
    build_metrics,
    clean_rewritten_query,
)


def test_clean_rewritten_query_accepts_one_plain_english_line() -> None:
    assert clean_rewritten_query("Query: vehicular MEC offloading probability") == (
        "vehicular MEC offloading probability"
    )
    assert clean_rewritten_query("```text\nNash equilibrium convergence\n```") == (
        "Nash equilibrium convergence"
    )


def test_clean_rewritten_query_rejects_explanation_or_non_english() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        clean_rewritten_query("English query\nAdditional explanation")
    with pytest.raises(ValueError, match="English text"):
        clean_rewritten_query("车辆卸载概率")


def test_build_metrics_scores_only_labelled_cases() -> None:
    records = [
        {
            "recall_at_k": 0.5,
            "precision_at_k": 0.25,
            "reciprocal_rank": 1.0,
            "rewrite_ms": 100,
            "retrieval_ms": 1000,
            "rewrite_error": None,
            "retrieval_error": None,
        },
        {
            "recall_at_k": None,
            "precision_at_k": None,
            "reciprocal_rank": None,
            "rewrite_ms": 200,
            "retrieval_ms": 3000,
            "rewrite_error": None,
            "retrieval_error": None,
        },
    ]

    metrics = build_metrics(records, timeout_ms=2000)

    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 0.25
    assert metrics["mrr"] == 1.0
    assert metrics["mean_rewrite_ms"] == 150
    assert metrics["timeout_rate"] == 0.5
