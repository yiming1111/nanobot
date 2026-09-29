"""Deterministic retrieval evaluation for a labelled local-paper benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Protocol

from pydantic import Field

from nanobot.research.models import (
    EvidenceSearchResult,
    PaperSearchResult,
    ResearchModel,
    utc_now,
)


class RetrievalService(Protocol):
    def search_papers(self, query: str, **kwargs: Any) -> list[PaperSearchResult]: ...

    def retrieve_evidence(
        self, query: str, **kwargs: Any
    ) -> list[EvidenceSearchResult]: ...


class RetrievalEvalCase(ResearchModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_paper_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RetrievalCaseResult(ResearchModel):
    case_id: str
    paper_ids: list[str]
    chunk_ids: list[str]
    paper_recall_at_k: float | None = None
    paper_reciprocal_rank: float | None = None
    evidence_recall_at_k: float | None = None
    evidence_paper_recall_at_k: float | None = None
    paper_search_ms: int = Field(ge=0)
    evidence_search_ms: int = Field(ge=0)


class RetrievalEvalReport(ResearchModel):
    created_at: datetime = Field(default_factory=utc_now)
    case_count: int = Field(ge=0)
    paper_top_k: int = Field(ge=1)
    evidence_top_k: int = Field(ge=1)
    metrics: dict[str, float | int | None]
    cases: list[RetrievalCaseResult]


def load_retrieval_cases(path: Path) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            cases.append(RetrievalEvalCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark case at line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("benchmark contains no cases")
    return cases


def evaluate_retrieval(
    service: RetrievalService,
    cases: list[RetrievalEvalCase],
    *,
    paper_top_k: int = 10,
    evidence_top_k: int = 8,
) -> RetrievalEvalReport:
    results: list[RetrievalCaseResult] = []
    for case in cases:
        started = perf_counter()
        papers = service.search_papers(case.query, top_k=paper_top_k)
        paper_ms = round((perf_counter() - started) * 1000)

        started = perf_counter()
        evidence = service.retrieve_evidence(case.query, top_k=evidence_top_k)
        evidence_ms = round((perf_counter() - started) * 1000)

        paper_ids = [item.paper_id for item in papers]
        chunk_ids = [item.chunk_id for item in evidence]
        evidence_paper_ids = {item.paper_id for item in evidence}
        relevant_papers = set(case.relevant_paper_ids)
        relevant_chunks = set(case.relevant_chunk_ids)
        first_rank = next(
            (
                index
                for index, paper_id in enumerate(paper_ids, start=1)
                if paper_id in relevant_papers
            ),
            None,
        )
        results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                paper_ids=paper_ids,
                chunk_ids=chunk_ids,
                paper_recall_at_k=(
                    len(relevant_papers.intersection(paper_ids)) / len(relevant_papers)
                    if relevant_papers
                    else None
                ),
                paper_reciprocal_rank=(
                    1.0 / first_rank if first_rank is not None else 0.0
                )
                if relevant_papers
                else None,
                evidence_recall_at_k=(
                    len(relevant_chunks.intersection(chunk_ids)) / len(relevant_chunks)
                    if relevant_chunks
                    else None
                ),
                evidence_paper_recall_at_k=(
                    len(relevant_papers.intersection(evidence_paper_ids))
                    / len(relevant_papers)
                    if relevant_papers
                    else None
                ),
                paper_search_ms=paper_ms,
                evidence_search_ms=evidence_ms,
            )
        )

    def average(field: str) -> float | None:
        values = [
            value
            for item in results
            if (value := getattr(item, field)) is not None
        ]
        return mean(values) if values else None

    paper_latencies = [item.paper_search_ms for item in results]
    evidence_latencies = [item.evidence_search_ms for item in results]
    return RetrievalEvalReport(
        case_count=len(results),
        paper_top_k=paper_top_k,
        evidence_top_k=evidence_top_k,
        metrics={
            "paper_recall_at_k": average("paper_recall_at_k"),
            "paper_mrr": average("paper_reciprocal_rank"),
            "evidence_recall_at_k": average("evidence_recall_at_k"),
            "evidence_paper_recall_at_k": average("evidence_paper_recall_at_k"),
            "mean_paper_search_ms": mean(paper_latencies) if paper_latencies else None,
            "mean_evidence_search_ms": (
                mean(evidence_latencies) if evidence_latencies else None
            ),
        },
        cases=results,
    )
