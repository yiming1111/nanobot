from pathlib import Path

import pytest

from nanobot.research.evaluation import (
    AnswerBehavior,
    SingleTurnEvalCase,
    SingleTurnPrediction,
    evaluate_single_turn_retrieval,
    load_single_turn_cases,
    score_single_turn_outputs,
)
from nanobot.research.models import EvidenceSearchResult


def _evidence(chunk_id: str) -> EvidenceSearchResult:
    return EvidenceSearchResult(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("-")[0],
        title="Paper",
        section="Results",
        page_start=2,
        page_end=2,
        text="Evidence text",
        fused_score=0.1,
    )


class _FakeRetrieval:
    last_query: str | None = None

    def retrieve_evidence(
        self, query: str, **kwargs: object
    ) -> list[EvidenceSearchResult]:
        self.last_query = query
        if query == "failure":
            raise RuntimeError("temporary failure")
        if query == "irrelevant":
            return [_evidence("P9-C1")]
        return [_evidence("P9-C1"), _evidence("P1-C2")]


def _case(
    case_id: str = "case-1",
    query: str = "relevant query",
    behavior: AnswerBehavior = AnswerBehavior.ANSWER,
) -> SingleTurnEvalCase:
    return SingleTurnEvalCase(
        case_id=case_id,
        query=query,
        ground_truth="Supported answer",
        relevant_chunk_ids=[] if behavior == AnswerBehavior.ABSTAIN else ["P1-C2"],
        expected_behavior=behavior,
        unsupported_requirements=["missing evidence"]
        if behavior != AnswerBehavior.ANSWER
        else [],
    )


def test_evaluate_single_turn_retrieval_computes_metrics() -> None:
    service = _FakeRetrieval()
    case = _case()
    case.retrieval_query = "English retrieval query"
    report = evaluate_single_turn_retrieval(
        service,
        [case],
        top_k=4,
    )

    assert report.scored_case_count == 1
    assert report.metrics["recall_at_k"] == 1.0
    assert report.metrics["precision_at_k"] == 0.25
    assert report.metrics["mrr"] == 0.5
    assert report.metrics["error_rate"] == 0.0
    assert service.last_query == "English retrieval query"


def test_abstain_is_not_scored_as_retrieval_miss_and_failure_does_not_abort() -> None:
    report = evaluate_single_turn_retrieval(
        _FakeRetrieval(),
        [
            _case("abstain", "irrelevant", AnswerBehavior.ABSTAIN),
            _case("failure", "failure"),
            _case("success"),
        ],
        top_k=4,
    )

    assert report.case_count == 3
    assert report.scored_case_count == 1
    assert report.cases[0].recall_at_k is None
    assert report.cases[1].error == "RuntimeError: temporary failure"
    assert report.metrics["error_rate"] == pytest.approx(1 / 3)


def test_load_single_turn_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    row = _case().model_dump_json()
    path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_single_turn_cases(path)


def test_score_single_turn_outputs_counts_missing_predictions_as_failures() -> None:
    report = score_single_turn_outputs(
        [_case("answered"), _case("missing")],
        [
            SingleTurnPrediction(
                case_id="answered",
                answer_text="Answer",
                actual_behavior=AnswerBehavior.ANSWER,
                cited_chunk_ids=["P1-C2"],
                citation_locators_valid=True,
            )
        ],
    )

    assert report.metrics["completion_rate"] == 0.5
    assert report.metrics["decision_accuracy"] == 0.5
    assert report.metrics["citation_accuracy"] == 0.5
