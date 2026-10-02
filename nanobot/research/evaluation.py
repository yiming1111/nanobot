"""Single-turn retrieval evaluation for the labelled local-paper benchmark."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Protocol

from pydantic import Field, model_validator

from nanobot.research.models import EvidenceSearchResult, ResearchModel, utc_now


class RetrievalService(Protocol):
    def retrieve_evidence(
        self, query: str, **kwargs: Any
    ) -> list[EvidenceSearchResult]: ...


class AnswerBehavior(StrEnum):
    ANSWER = "answer"
    PARTIAL = "partial"
    ABSTAIN = "abstain"


class SingleTurnEvalCase(ResearchModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_behavior: AnswerBehavior
    unsupported_requirements: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_labels(self) -> "SingleTurnEvalCase":
        if self.expected_behavior == AnswerBehavior.ANSWER:
            if not self.relevant_chunk_ids:
                raise ValueError("answer cases require at least one relevant chunk")
            if self.unsupported_requirements:
                raise ValueError("answer cases cannot contain unsupported requirements")
        elif self.expected_behavior == AnswerBehavior.PARTIAL:
            if not self.relevant_chunk_ids:
                raise ValueError("partial cases require at least one relevant chunk")
            if not self.unsupported_requirements:
                raise ValueError("partial cases require an explicit evidence gap")
        elif self.relevant_chunk_ids:
            raise ValueError("abstain cases cannot contain relevant chunks")
        elif not self.unsupported_requirements:
            raise ValueError("abstain cases require an explicit evidence gap")
        return self


class SingleTurnRetrievalResult(ResearchModel):
    case_id: str
    expected_behavior: AnswerBehavior
    chunk_ids: list[str] = Field(default_factory=list)
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    precision_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_ms: int = Field(ge=0)
    timed_out: bool = False
    error: str | None = None


class SingleTurnRetrievalReport(ResearchModel):
    created_at: datetime = Field(default_factory=utc_now)
    case_count: int = Field(ge=0)
    scored_case_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    timeout_ms: int | None = Field(default=None, ge=1)
    metrics: dict[str, float | int | None]
    cases: list[SingleTurnRetrievalResult]


class SingleTurnPrediction(ResearchModel):
    """Serializable output captured from one full Agent run."""

    case_id: str = Field(min_length=1)
    answer_text: str = ""
    actual_behavior: AnswerBehavior | None = None
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    citation_locators_valid: bool | None = None
    task_id: str | None = None
    total_ms: int | None = Field(default=None, ge=0)
    error: str | None = None


class SingleTurnOutputReport(ResearchModel):
    case_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    metrics: dict[str, float | int | None]


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        rows.append((line_number, payload))
    if not rows:
        raise ValueError("file contains no records")
    return rows


def _ensure_unique_ids(records: list[tuple[int, ResearchModel]]) -> None:
    seen: dict[str, int] = {}
    for line_number, record in records:
        record_id = str(getattr(record, "case_id"))
        if record_id in seen:
            raise ValueError(
                f"duplicate case_id {record_id!r} at lines {seen[record_id]} and {line_number}"
            )
        seen[record_id] = line_number


def load_single_turn_cases(path: Path) -> list[SingleTurnEvalCase]:
    parsed: list[tuple[int, SingleTurnEvalCase]] = []
    for line_number, payload in _read_jsonl(path):
        try:
            parsed.append((line_number, SingleTurnEvalCase.model_validate(payload)))
        except ValueError as exc:
            raise ValueError(f"invalid single-turn case at line {line_number}: {exc}") from exc
    _ensure_unique_ids(parsed)
    return [record for _, record in parsed]


def load_single_turn_predictions(path: Path) -> list[SingleTurnPrediction]:
    parsed: list[tuple[int, SingleTurnPrediction]] = []
    for line_number, payload in _read_jsonl(path):
        try:
            parsed.append((line_number, SingleTurnPrediction.model_validate(payload)))
        except ValueError as exc:
            raise ValueError(f"invalid prediction at line {line_number}: {exc}") from exc
    _ensure_unique_ids(parsed)
    return [record for _, record in parsed]


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def evaluate_single_turn_retrieval(
    service: RetrievalService,
    cases: list[SingleTurnEvalCase],
    *,
    top_k: int = 4,
    timeout_ms: int | None = 240_000,
    on_case_complete: Callable[[int, int, SingleTurnRetrievalResult], None]
    | None = None,
) -> SingleTurnRetrievalReport:
    """Run full-corpus chunk retrieval and continue after individual failures."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    results: list[SingleTurnRetrievalResult] = []
    for index, case in enumerate(cases, start=1):
        started = perf_counter()
        evidence: list[EvidenceSearchResult] = []
        error: str | None = None
        try:
            evidence = service.retrieve_evidence(case.query, top_k=top_k)
        except Exception as exc:  # A failed case must not abort the remaining benchmark.
            error = f"{type(exc).__name__}: {exc}"
        retrieval_ms = round((perf_counter() - started) * 1000)
        chunk_ids = [item.chunk_id for item in evidence]
        relevant_chunks = set(case.relevant_chunk_ids)
        matched_count = len(relevant_chunks.intersection(chunk_ids))
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(chunk_ids, start=1)
                if chunk_id in relevant_chunks
            ),
            None,
        )
        is_scored = bool(relevant_chunks) and error is None
        result = SingleTurnRetrievalResult(
            case_id=case.case_id,
            expected_behavior=case.expected_behavior,
            chunk_ids=chunk_ids,
            recall_at_k=(matched_count / len(relevant_chunks)) if is_scored else None,
            precision_at_k=(matched_count / top_k) if is_scored else None,
            reciprocal_rank=(1.0 / first_rank if first_rank else 0.0)
            if is_scored
            else None,
            retrieval_ms=retrieval_ms,
            timed_out=timeout_ms is not None and retrieval_ms > timeout_ms,
            error=error,
        )
        results.append(result)
        if on_case_complete is not None:
            on_case_complete(index, len(cases), result)

    def average(field: str) -> float | None:
        values = [
            value
            for item in results
            if (value := getattr(item, field)) is not None
        ]
        return mean(values) if values else None

    latencies = [item.retrieval_ms for item in results]
    scored_case_count = sum(item.recall_at_k is not None for item in results)
    denominator = len(results)
    return SingleTurnRetrievalReport(
        case_count=denominator,
        scored_case_count=scored_case_count,
        top_k=top_k,
        timeout_ms=timeout_ms,
        metrics={
            "recall_at_k": average("recall_at_k"),
            "precision_at_k": average("precision_at_k"),
            "mrr": average("reciprocal_rank"),
            "mean_retrieval_ms": mean(latencies) if latencies else None,
            "p95_retrieval_ms": _percentile(latencies, 0.95),
            "timeout_rate": (
                sum(item.timed_out for item in results) / denominator
                if denominator
                else None
            ),
            "error_rate": (
                sum(item.error is not None for item in results) / denominator
                if denominator
                else None
            ),
        },
        cases=results,
    )


def score_single_turn_outputs(
    cases: list[SingleTurnEvalCase],
    predictions: list[SingleTurnPrediction],
) -> SingleTurnOutputReport:
    """Score deterministic output fields; semantic RAG metrics are computed separately."""

    predictions_by_id = {item.case_id: item for item in predictions}
    known_ids = {case.case_id for case in cases}
    unknown_ids = sorted(set(predictions_by_id).difference(known_ids))
    if unknown_ids:
        raise ValueError(f"predictions contain unknown case IDs: {', '.join(unknown_ids)}")

    decision_scores: list[bool] = []
    citation_scores: list[bool] = []
    completed = 0
    for case in cases:
        prediction = predictions_by_id.get(case.case_id)
        if prediction is None or prediction.error is not None:
            decision_scores.append(False)
            if case.expected_behavior != AnswerBehavior.ABSTAIN:
                citation_scores.append(False)
            continue
        completed += 1
        decision_scores.append(prediction.actual_behavior == case.expected_behavior)
        if case.expected_behavior != AnswerBehavior.ABSTAIN:
            citation_scores.append(prediction.citation_locators_valid is True)

    case_count = len(cases)
    return SingleTurnOutputReport(
        case_count=case_count,
        prediction_count=len(predictions),
        metrics={
            "completion_rate": completed / case_count if case_count else None,
            "decision_accuracy": mean(decision_scores) if decision_scores else None,
            "citation_accuracy": mean(citation_scores) if citation_scores else None,
        },
    )


__all__ = [
    "AnswerBehavior",
    "SingleTurnEvalCase",
    "SingleTurnOutputReport",
    "SingleTurnPrediction",
    "SingleTurnRetrievalReport",
    "SingleTurnRetrievalResult",
    "evaluate_single_turn_retrieval",
    "load_single_turn_cases",
    "load_single_turn_predictions",
    "score_single_turn_outputs",
]
