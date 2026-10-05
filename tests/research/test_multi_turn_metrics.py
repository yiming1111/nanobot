from benchmarks.paper_research.build_multi_turn_report import _retrieval_metrics
from benchmarks.paper_research.run_multi_turn_consistency_eval import _summarize


def test_multi_turn_retrieval_metrics_use_first_eight_ranked_chunks() -> None:
    ranked = [f"P1-C{index}" for index in range(1, 11)]

    recall, precision, reciprocal_rank = _retrieval_metrics(
        ranked,
        ["P1-C2", "P1-C9"],
        top_k=8,
    )

    assert recall == 0.5
    assert precision == 0.125
    assert reciprocal_rank == 0.5


def test_multi_turn_summary_separates_semantic_and_abstain_cases() -> None:
    report = _summarize(
        [
            {
                "case_id": "answer",
                "expected_behavior": "answer",
                "answer_correct": True,
                "faithfulness": 0.8,
                "answer_relevancy": 0.9,
                "context_relevancy": 0.5,
                "context_recall": 0.75,
                "preserved": [{"text": "paper", "preserved": True}],
                "memory_probe": True,
                "compaction_before_turn": False,
                "error": None,
            },
            {
                "case_id": "abstain",
                "expected_behavior": "abstain",
                "answer_correct": False,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_relevancy": None,
                "context_recall": None,
                "preserved": [],
                "memory_probe": False,
                "compaction_before_turn": False,
                "error": None,
            },
        ]
    )

    assert report["answer_correctness"] == 0.5
    assert report["semantic_case_count"] == 1
    assert report["faithfulness"] == 0.8
    assert report["answer_relevancy"] == 0.9
    assert report["context_relevancy"] == 0.5
    assert report["context_recall"] == 0.75
    assert report["memory_probe_consistency_rate"] == 1.0
