"""FastMCP surface for nanobot's scientific-paper Agentic RAG tools."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

from mcp.server.fastmcp import FastMCP

from nanobot.research.config import ResearchConfig
from nanobot.research.models import (
    CitationLocatorCheck,
    ConstraintValue,
    EvidenceAssessment,
    EvidenceItem,
    EvidenceRelation,
    EvidenceSearchResult,
    ResearchPlan,
    ResearchState,
    ResearchTaskStatus,
    RetrievalScope,
    SubQuestion,
    SubQuestionStatus,
)
from nanobot.research.observability.trace import PipelineTraceStore
from nanobot.research.retrieval.service import HybridRetrievalService
from nanobot.research.workflow.state_store import ResearchStateStore

mcp = FastMCP("nanobot-paper-research", json_response=True)
logger = logging.getLogger(__name__)


class ResearchTools:
    """Tool implementation kept separate from MCP decorators for testing."""

    def __init__(
        self,
        config: ResearchConfig,
        *,
        retrieval: HybridRetrievalService | None = None,
    ) -> None:
        config.ensure_directories()
        self.config = config
        self.states = ResearchStateStore(config.states_dir)
        self.traces = PipelineTraceStore(config.traces_dir)
        self._retrieval = retrieval

    def _record_trace(
        self,
        task_id: str,
        *,
        stage: str,
        operation: str,
        elapsed_ms: int = 0,
        sub_question_id: str | None = None,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.traces.append(
                task_id,
                stage=stage,
                operation=operation,
                elapsed_ms=elapsed_ms,
                sub_question_id=sub_question_id,
                input_summary=input_summary,
                output_summary=output_summary,
            )
        except Exception:
            logger.exception("failed to record research trace task=%s", task_id)

    @property
    def retrieval(self) -> HybridRetrievalService:
        if self._retrieval is None:
            logger.info("retrieval service construction started")
            started_at = perf_counter()
            self._retrieval = HybridRetrievalService(self.config)
            logger.info(
                "retrieval service construction completed elapsed_ms=%d",
                round((perf_counter() - started_at) * 1000),
            )
        return self._retrieval

    def start(
        self,
        *,
        original_question: str,
        normalized_question: str,
        sub_questions: list[str],
        task_type: str = "paper_research",
        constraints: dict[str, ConstraintValue] | None = None,
        session_key: str | None = None,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        questions = [value.strip() for value in sub_questions if value.strip()]
        if not questions:
            questions = [normalized_question.strip()]
        plan = ResearchPlan(
            original_question=original_question,
            normalized_question=normalized_question,
            task_type=task_type,
            constraints=constraints or {},
            requires_decomposition=len(questions) > 1,
            sub_questions=[
                SubQuestion(sub_question_id=f"SQ{index}", question=question)
                for index, question in enumerate(questions, start=1)
            ],
        )
        state = ResearchState(
            session_key=session_key,
            status=ResearchTaskStatus.RETRIEVING,
            plan=plan,
            max_retrieval_rounds=self.config.max_retrieval_rounds,
        )
        self.states.create(state)
        payload = state.model_dump(mode="json")
        self._record_trace(
            state.task_id,
            stage="plan",
            operation="research_start",
            elapsed_ms=round((perf_counter() - started_at) * 1000),
            input_summary={
                "task_type": task_type,
                "sub_question_count": len(questions),
                "constraint_keys": sorted((constraints or {}).keys()),
            },
            output_summary={
                "status": state.status.value,
                "sub_question_ids": [
                    item.sub_question_id for item in plan.sub_questions
                ],
            },
        )
        return payload

    def _evidence_items(
        self,
        *,
        task_id: str,
        sub_question_id: str,
        results: list[EvidenceSearchResult],
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for result in results:
            digest = hashlib.sha256(
                f"{task_id}\0{sub_question_id}\0{result.chunk_id}".encode("utf-8")
            ).hexdigest()[:16]
            evidence.append(
                EvidenceItem(
                    evidence_id=f"E-{digest}",
                    sub_question_id=sub_question_id,
                    paper_id=result.paper_id,
                    chunk_id=result.chunk_id,
                    title=result.title,
                    section=result.section,
                    page_start=result.page_start,
                    page_end=result.page_end,
                    text=result.text,
                    relation=EvidenceRelation.UNCLASSIFIED,
                    dense_score=result.dense_score,
                    sparse_score=result.sparse_score,
                    fused_score=result.fused_score,
                    rerank_score=result.rerank_score,
                )
            )
        return evidence

    def _retrieve_full_corpus(
        self,
        *,
        query: str,
        top_k: int | None,
        year_from: int | None,
        year_to: int | None,
        sections: list[str] | None,
    ) -> tuple[list[EvidenceSearchResult], dict[str, Any]]:
        started_at = perf_counter()
        results = self.retrieval.retrieve_evidence(
            query,
            top_k=top_k,
            paper_ids=None,
            year_from=year_from,
            year_to=year_to,
            sections=sections,
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000)
        return results, {
            "scope": RetrievalScope.FULL_CORPUS.value,
            "chunk_ids": [item.chunk_id for item in results],
            "elapsed_ms": elapsed_ms,
        }

    def research_retrieve(
        self,
        *,
        task_id: str,
        sub_question_id: str,
        query: str,
        top_k: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        sections: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run one semantic search over every eligible chunk in the corpus."""
        total_started_at = perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        state = self.states.load(task_id)
        sub_question = next(
            (
                item
                for item in state.plan.sub_questions
                if item.sub_question_id == sub_question_id
            ),
            None,
        )
        if sub_question is None:
            raise ValueError(f"unknown sub-question: {sub_question_id}")
        if sub_question.status == SubQuestionStatus.SUFFICIENT:
            raise RuntimeError(f"evidence is already sufficient for {sub_question_id}")
        if len(sub_question.retrieval_attempts) >= state.max_retrieval_rounds:
            raise RuntimeError(
                f"maximum semantic retrieval rounds reached for {sub_question_id}"
            )
        if sub_question.retrieval_attempts and sub_question.next_query is None:
            raise RuntimeError(
                f"run research_reflect before retrying {sub_question_id}"
            )
        if (
            sub_question.retrieval_attempts
            and sub_question.next_query is not None
            and query != sub_question.next_query.strip()
        ):
            raise ValueError(
                f"retry query must match Reflect next_query: {sub_question.next_query}"
            )

        semantic_round = len(sub_question.retrieval_attempts) + 1
        logger.info(
            "research retrieval started task=%s sub_question=%s semantic_round=%d",
            task_id,
            sub_question_id,
            semantic_round,
        )
        results, retrieval_attempt = self._retrieve_full_corpus(
            query=query,
            top_k=top_k,
            year_from=year_from,
            year_to=year_to,
            sections=sections,
        )

        evidence = self._evidence_items(
            task_id=task_id,
            sub_question_id=sub_question_id,
            results=results,
        )
        total_elapsed_ms = round((perf_counter() - total_started_at) * 1000)
        coverage_reason = (
            "full-corpus candidate passages retrieved; semantic sufficiency awaits "
            "research_reflect"
            if evidence
            else "no candidate passages were found in the eligible corpus chunks"
        )
        updated = self.states.add_evidence(
            task_id,
            sub_question_id,
            evidence,
            query=query,
            paper_ids=None,
            scope=RetrievalScope.FULL_CORPUS,
            elapsed_ms=total_elapsed_ms,
            coverage_reason=coverage_reason,
        )
        updated_sub_question = next(
            item
            for item in updated.plan.sub_questions
            if item.sub_question_id == sub_question_id
        )
        payload = {
            "task_id": task_id,
            "sub_question_id": sub_question_id,
            "semantic_round": semantic_round,
            "max_semantic_rounds": updated.max_retrieval_rounds,
            "retrieval_scope": RetrievalScope.FULL_CORPUS.value,
            "retrieval_attempt": retrieval_attempt,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "candidate_passages_found": bool(evidence),
            "semantic_sufficiency": "pending_reflect",
            "task_status": updated.status.value,
            "sub_question_status": updated_sub_question.status.value,
            "next_step": "run research_reflect after all pending sub-questions are retrieved",
        }
        self._record_trace(
            task_id,
            stage="retrieve",
            operation="research_retrieve",
            elapsed_ms=total_elapsed_ms,
            sub_question_id=sub_question_id,
            input_summary={
                "query": query,
                "semantic_round": semantic_round,
                "year_from": year_from,
                "year_to": year_to,
                "sections": sections or [],
            },
            output_summary={
                "retrieval_scope": RetrievalScope.FULL_CORPUS.value,
                "retrieval_attempt": retrieval_attempt,
                "evidence_ids": [item.evidence_id for item in evidence],
                "candidate_passages_found": bool(evidence),
                "task_status": updated.status.value,
            },
        )
        return payload

    def reflect(
        self,
        *,
        task_id: str,
        assessments: list[EvidenceAssessment],
    ) -> dict[str, Any]:
        """Decide whether retrieved evidence covers each outstanding evidence need."""
        started_at = perf_counter()
        if not assessments:
            raise ValueError("at least one evidence assessment is required")
        updated = self.states.record_reflections(task_id, assessments)
        retry_sub_questions = [
            {
                "sub_question_id": item.sub_question_id,
                "next_query": item.next_query,
                "remaining_semantic_rounds": (
                    updated.max_retrieval_rounds - len(item.retrieval_attempts)
                ),
            }
            for item in updated.plan.sub_questions
            if item.status == SubQuestionStatus.INSUFFICIENT
            and item.next_query is not None
        ]
        sufficient_ids = [
            item.sub_question_id
            for item in updated.plan.sub_questions
            if item.status == SubQuestionStatus.SUFFICIENT
        ]
        unresolved_ids = [
            item.sub_question_id
            for item in updated.plan.sub_questions
            if item.status == SubQuestionStatus.INSUFFICIENT
            and item.next_query is None
        ]
        payload = {
            "task_id": task_id,
            "status": updated.status.value,
            "sufficient_sub_question_ids": sufficient_ids,
            "retry_sub_questions": retry_sub_questions,
            "unresolved_sub_question_ids": unresolved_ids,
            "can_generate": updated.status == ResearchTaskStatus.READY_TO_SYNTHESIZE,
            "must_abstain": updated.status == ResearchTaskStatus.REFUSED,
            "next_step": (
                "run one focused retrieval for each retry_sub_question"
                if retry_sub_questions
                else "generate only from sufficient evidence"
                if updated.status == ResearchTaskStatus.READY_TO_SYNTHESIZE
                else "abstain because the corpus lacks sufficient evidence"
            ),
        }
        self._record_trace(
            task_id,
            stage="reflect",
            operation="research_reflect",
            elapsed_ms=round((perf_counter() - started_at) * 1000),
            input_summary={
                "assessments": [item.model_dump(mode="json") for item in assessments]
            },
            output_summary={
                "status": updated.status.value,
                "sufficient_sub_question_ids": sufficient_ids,
                "retry_sub_question_ids": [
                    item["sub_question_id"] for item in retry_sub_questions
                ],
                "unresolved_sub_question_ids": unresolved_ids,
            },
        )
        return payload

    def neighbors(
        self,
        chunk_id: str,
        *,
        window: int = 1,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        started_at = perf_counter()
        payload = [
            item.model_dump(mode="json")
            for item in self.retrieval.get_neighbors(chunk_id, window=window)
        ]
        if task_id is not None:
            self._record_trace(
                task_id,
                stage="retrieve",
                operation="get_neighbor_evidence",
                elapsed_ms=round((perf_counter() - started_at) * 1000),
                input_summary={"chunk_id": chunk_id, "window": window},
                output_summary={
                    "chunk_ids": [item["chunk_id"] for item in payload]
                },
            )
        return payload

    def finalize(
        self,
        *,
        task_id: str,
        answered_sub_question_ids: list[str],
        cited_evidence_ids: list[str],
    ) -> dict[str, Any]:
        """Validate citation locators after generating an evidence-grounded answer."""
        started_at = perf_counter()
        if not answered_sub_question_ids:
            raise ValueError("at least one answered sub-question is required")
        if len(answered_sub_question_ids) != len(set(answered_sub_question_ids)):
            raise ValueError("answered_sub_question_ids must be unique")
        if len(cited_evidence_ids) != len(set(cited_evidence_ids)):
            raise ValueError("cited_evidence_ids must be unique")

        state = self.states.load(task_id)
        if state.status != ResearchTaskStatus.READY_TO_SYNTHESIZE:
            raise RuntimeError("research_finalize requires reflected sufficient evidence")
        sub_questions = {
            item.sub_question_id: item for item in state.plan.sub_questions
        }
        unknown_sub_questions = set(answered_sub_question_ids) - set(sub_questions)
        if unknown_sub_questions:
            raise ValueError(
                f"unknown answered sub-questions: {sorted(unknown_sub_questions)}"
            )
        insufficient = [
            sub_question_id
            for sub_question_id in answered_sub_question_ids
            if sub_questions[sub_question_id].status != SubQuestionStatus.SUFFICIENT
        ]
        if insufficient:
            raise ValueError(
                f"cannot answer sub-questions with insufficient evidence: {insufficient}"
            )

        checks: list[CitationLocatorCheck] = []
        errors: list[str] = []
        cited_by_sub_question: dict[str, int] = {
            value: 0 for value in answered_sub_question_ids
        }
        for evidence_id in cited_evidence_ids:
            evidence = state.evidence.get(evidence_id)
            if evidence is None:
                checks.append(
                    CitationLocatorCheck(
                        evidence_id=evidence_id,
                        reason="evidence ID is not present in this research task",
                    )
                )
                errors.append(f"unknown evidence ID: {evidence_id}")
                continue
            locatable = bool(
                evidence.title
                and evidence.chunk_id
                and evidence.page_start >= 1
                and evidence.page_end >= evidence.page_start
            )
            checks.append(
                CitationLocatorCheck(
                    evidence_id=evidence_id,
                    sub_question_id=evidence.sub_question_id,
                    locatable=locatable,
                    reason=(
                        "citation locator is complete"
                        if locatable
                        else "citation locator is incomplete"
                    ),
                    title=evidence.title,
                    page_start=evidence.page_start,
                    page_end=evidence.page_end,
                    chunk_id=evidence.chunk_id,
                )
            )
            if evidence.sub_question_id not in cited_by_sub_question:
                errors.append(
                    "citation belongs to an unanswered sub-question: "
                    f"{evidence.sub_question_id}"
                )
            else:
                cited_by_sub_question[evidence.sub_question_id] += 1
            if not locatable:
                errors.append(f"citation locator is incomplete: {evidence_id}")
        for sub_question_id, citation_count in cited_by_sub_question.items():
            if citation_count == 0:
                errors.append(f"answered sub-question has no citation: {sub_question_id}")
        if errors:
            raise ValueError("; ".join(errors))

        updated = self.states.finalize(
            task_id,
            answered_sub_question_ids=answered_sub_question_ids,
            cited_evidence_ids=cited_evidence_ids,
            checks=checks,
        )
        unresolved = list(
            updated.metadata.get("finalization", {}).get(
                "unresolved_sub_question_ids", []
            )
        )
        payload = {
            "task_id": task_id,
            "status": updated.status.value,
            "citations_valid": True,
            "citation_checks": [item.model_dump(mode="json") for item in checks],
            "unresolved_sub_question_ids": unresolved,
            "next_step": (
                "return the grounded answer and state unresolved evidence gaps"
                if unresolved
                else "return the grounded answer"
            ),
        }
        self._record_trace(
            task_id,
            stage="finalize",
            operation="research_finalize",
            elapsed_ms=round((perf_counter() - started_at) * 1000),
            input_summary={
                "answered_sub_question_ids": answered_sub_question_ids,
                "cited_evidence_ids": cited_evidence_ids,
            },
            output_summary={
                "status": updated.status.value,
                "citations_valid": True,
                "unresolved_sub_question_ids": unresolved,
            },
        )
        return payload

    def status(
        self,
        task_id: str,
        *,
        include_evidence_text: bool = False,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        payload = self.states.load(task_id).model_dump(mode="json")
        if not include_evidence_text:
            for evidence in payload["evidence"].values():
                evidence.pop("text", None)
        self._record_trace(
            task_id,
            stage="inspect",
            operation="research_status",
            elapsed_ms=round((perf_counter() - started_at) * 1000),
            input_summary={"include_evidence_text": include_evidence_text},
            output_summary={
                "status": payload["status"],
                "evidence_count": len(payload["evidence"]),
            },
        )
        return payload


@lru_cache(maxsize=1)
def _tools() -> ResearchTools:
    return ResearchTools(ResearchConfig())


@mcp.tool()
def research_start(
    original_question: str,
    normalized_question: str,
    sub_questions: list[str],
    task_type: str = "paper_research",
    constraints: dict[str, ConstraintValue] | None = None,
    session_key: str | None = None,
) -> dict[str, Any]:
    """Create an evidence-needs plan before searching the local paper corpus."""
    return _tools().start(
        original_question=original_question,
        normalized_question=normalized_question,
        sub_questions=sub_questions,
        task_type=task_type,
        constraints=constraints,
        session_key=session_key,
    )


@mcp.tool()
def research_retrieve(
    task_id: str,
    sub_question_id: str,
    query: str,
    top_k: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Run one semantic retrieval round with automatic empty-result fallback."""
    return _tools().research_retrieve(
        task_id=task_id,
        sub_question_id=sub_question_id,
        query=query,
        top_k=top_k,
        year_from=year_from,
        year_to=year_to,
        sections=sections,
    )


@mcp.tool()
def research_reflect(
    task_id: str,
    assessments: list[EvidenceAssessment],
) -> dict[str, Any]:
    """Judge whether retrieved passages sufficiently cover each evidence need."""
    return _tools().reflect(task_id=task_id, assessments=assessments)


@mcp.tool()
def get_neighbor_evidence(
    chunk_id: str,
    window: int = 1,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read adjacent chunks to check the context around a retrieved passage."""
    return _tools().neighbors(chunk_id, window=window, task_id=task_id)


@mcp.tool()
def research_status(
    task_id: str,
    include_evidence_text: bool = False,
) -> dict[str, Any]:
    """Return research state; evidence text is omitted by default."""
    return _tools().status(task_id, include_evidence_text=include_evidence_text)


@mcp.tool()
def research_finalize(
    task_id: str,
    answered_sub_question_ids: list[str],
    cited_evidence_ids: list[str],
) -> dict[str, Any]:
    """Validate final citation IDs and locators after grounded generation."""
    return _tools().finalize(
        task_id=task_id,
        answered_sub_question_ids=answered_sub_question_ids,
        cited_evidence_ids=cited_evidence_ids,
    )


def main() -> None:
    logger.info("research MCP runtime preparation started")
    started_at = perf_counter()
    tools = _tools()
    tools.retrieval.prepare_indexes()
    tools.retrieval.prepare_models()
    logger.info(
        "research MCP runtime preparation completed elapsed_ms=%d",
        round((perf_counter() - started_at) * 1000),
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
