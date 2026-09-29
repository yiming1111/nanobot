"""FastMCP surface for nanobot's scientific-paper Agentic RAG tools."""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from time import perf_counter
from typing import Any

from mcp.server.fastmcp import FastMCP

from nanobot.research.config import ResearchConfig
from nanobot.research.models import (
    CitationLocatorCheck,
    CitationReference,
    ConstraintValue,
    EvidenceAssessment,
    EvidenceItem,
    EvidenceRelation,
    EvidenceSearchResult,
    ResearchPlan,
    ResearchState,
    ResearchTaskStatus,
    RetrievalExecutionStatus,
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
        self._retrieval_locks: dict[tuple[str, str], threading.Lock] = {}
        self._retrieval_locks_guard = threading.Lock()

    def _retrieval_lock(self, task_id: str, sub_question_id: str) -> threading.Lock:
        key = (task_id, sub_question_id)
        with self._retrieval_locks_guard:
            return self._retrieval_locks.setdefault(key, threading.Lock())

    def _reconcile_interrupted_retrievals(self, task_id: str) -> ResearchState:
        """Turn stale `running` records from an earlier process into failures."""
        state = self.states.load(task_id)
        changed = False
        for item in state.plan.sub_questions:
            execution = item.retrieval_execution
            if (
                execution is not None
                and execution.status == RetrievalExecutionStatus.RUNNING
                and not self._retrieval_lock(task_id, item.sub_question_id).locked()
            ):
                self.states.fail_retrieval(
                    task_id,
                    item.sub_question_id,
                    query=execution.query,
                    error="retrieval process ended before completion",
                )
                changed = True
        return self.states.load(task_id) if changed else state

    @staticmethod
    def _find_sub_question(state: ResearchState, sub_question_id: str) -> SubQuestion:
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
        return sub_question

    def _cached_retrieval_payload(
        self,
        state: ResearchState,
        sub_question: SubQuestion,
    ) -> dict[str, Any] | None:
        """Return a completed, not-yet-reflected round without searching again."""
        if not sub_question.retrieval_attempts:
            return None
        if len(sub_question.retrieval_attempts) == len(sub_question.reflections):
            return None
        attempt = sub_question.retrieval_attempts[-1]
        evidence = [
            state.evidence[f"{sub_question.sub_question_id}:{chunk_id}"]
            for chunk_id in attempt.chunk_ids
            if f"{sub_question.sub_question_id}:{chunk_id}" in state.evidence
        ]
        return {
            "task_id": state.task_id,
            "sub_question_id": sub_question.sub_question_id,
            "semantic_round": attempt.round_index,
            "max_semantic_rounds": state.max_retrieval_rounds,
            "retrieval_scope": attempt.scope.value,
            "retrieval_attempt": attempt.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "candidate_passages_found": bool(evidence),
            "semantic_sufficiency": "pending_reflect",
            "task_status": state.status.value,
            "sub_question_status": sub_question.status.value,
            "execution_status": "completed",
            "recovered": True,
            "next_step": "run research_reflect after all pending sub-questions are retrieved",
        }

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
        sub_question_id: str,
        results: list[EvidenceSearchResult],
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for result in results:
            evidence.append(
                EvidenceItem(
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
        sub_question = self._find_sub_question(state, sub_question_id)
        cached = self._cached_retrieval_payload(state, sub_question)
        if cached is not None and sub_question.retrieval_attempts[-1].query == query:
            return cached

        execution_lock = self._retrieval_lock(task_id, sub_question_id)
        if not execution_lock.acquire(blocking=False):
            return {
                "task_id": task_id,
                "sub_question_id": sub_question_id,
                "execution_status": "running",
                "task_status": state.status.value,
                "next_step": (
                    "call research_status with wait_seconds=10; "
                    "do not start a duplicate retrieval"
                ),
            }

        try:
            state = self.states.load(task_id)
            sub_question = self._find_sub_question(state, sub_question_id)
            cached = self._cached_retrieval_payload(state, sub_question)
            if (
                cached is not None
                and sub_question.retrieval_attempts[-1].query == query
            ):
                return cached
            if sub_question.status == SubQuestionStatus.SUFFICIENT:
                raise RuntimeError(
                    f"evidence is already sufficient for {sub_question_id}"
                )
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

            self.states.begin_retrieval(task_id, sub_question_id, query=query)
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
            updated_sub_question = self._find_sub_question(updated, sub_question_id)
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
                "execution_status": "completed",
                "recovered": False,
                "next_step": (
                    "run research_reflect after all pending sub-questions are retrieved"
                ),
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
                    "chunk_ids": [item.chunk_id for item in evidence],
                    "candidate_passages_found": bool(evidence),
                    "task_status": updated.status.value,
                },
            )
            return payload
        except Exception as exc:
            try:
                self.states.fail_retrieval(
                    task_id,
                    sub_question_id,
                    query=query,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._record_trace(
                    task_id,
                    stage="retrieve",
                    operation="research_retrieve",
                    elapsed_ms=round((perf_counter() - total_started_at) * 1000),
                    sub_question_id=sub_question_id,
                    input_summary={"query": query},
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                logger.exception("failed to persist retrieval failure task=%s", task_id)
            raise
        finally:
            execution_lock.release()

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
            if item.status == SubQuestionStatus.INSUFFICIENT and item.next_query is None
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
                output_summary={"chunk_ids": [item["chunk_id"] for item in payload]},
            )
        return payload

    def finalize(
        self,
        *,
        task_id: str,
        answered_sub_question_ids: list[str],
        citations: list[CitationReference],
    ) -> dict[str, Any]:
        """Validate citation locators after generating an evidence-grounded answer."""
        started_at = perf_counter()
        if not answered_sub_question_ids:
            raise ValueError("at least one answered sub-question is required")
        if len(answered_sub_question_ids) != len(set(answered_sub_question_ids)):
            raise ValueError("answered_sub_question_ids must be unique")
        citation_keys = [(item.sub_question_id, item.chunk_id) for item in citations]
        if len(citation_keys) != len(set(citation_keys)):
            raise ValueError("citation sub-question and chunk pairs must be unique")

        state = self.states.load(task_id)
        if state.status != ResearchTaskStatus.READY_TO_SYNTHESIZE:
            raise RuntimeError(
                "research_finalize requires reflected sufficient evidence"
            )
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
        for citation in citations:
            if citation.sub_question_id not in cited_by_sub_question:
                errors.append(
                    "citation belongs to an unanswered sub-question: "
                    f"{citation.sub_question_id}"
                )
                continue
            sub_question = sub_questions[citation.sub_question_id]
            if citation.chunk_id not in sub_question.supporting_chunk_ids:
                errors.append(
                    "citation chunk was not approved by Reflect for sub-question "
                    f"{citation.sub_question_id}: {citation.chunk_id}"
                )
                continue
            evidence = next(
                (
                    item
                    for item in state.evidence.values()
                    if item.sub_question_id == citation.sub_question_id
                    and item.chunk_id == citation.chunk_id
                ),
                None,
            )
            if evidence is None:
                checks.append(
                    CitationLocatorCheck(
                        sub_question_id=citation.sub_question_id,
                        chunk_id=citation.chunk_id,
                        reason="chunk is not present in this research task",
                    )
                )
                errors.append(
                    "unknown chunk for sub-question "
                    f"{citation.sub_question_id}: {citation.chunk_id}"
                )
                continue
            locatable = bool(
                evidence.title
                and evidence.chunk_id
                and evidence.page_start >= 1
                and evidence.page_end >= evidence.page_start
            )
            checks.append(
                CitationLocatorCheck(
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
            cited_by_sub_question[evidence.sub_question_id] += 1
            if not locatable:
                errors.append(f"citation locator is incomplete: {evidence.chunk_id}")
        for sub_question_id, citation_count in cited_by_sub_question.items():
            if citation_count == 0:
                errors.append(
                    f"answered sub-question has no citation: {sub_question_id}"
                )
        if errors:
            raise ValueError("; ".join(errors))

        updated = self.states.finalize(
            task_id,
            answered_sub_question_ids=answered_sub_question_ids,
            citations=citations,
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
                "citations": [item.model_dump(mode="json") for item in citations],
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
        wait_seconds: int = 0,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        wait_seconds = max(0, min(wait_seconds, 30))
        deadline = time.monotonic() + wait_seconds
        while wait_seconds:
            state = self._reconcile_interrupted_retrievals(task_id)
            running = [
                item
                for item in state.plan.sub_questions
                if item.retrieval_execution is not None
                and item.retrieval_execution.status == RetrievalExecutionStatus.RUNNING
                and self._retrieval_lock(task_id, item.sub_question_id).locked()
            ]
            if not running or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        payload = self._reconcile_interrupted_retrievals(task_id).model_dump(
            mode="json"
        )
        if not include_evidence_text:
            for evidence in payload["evidence"].values():
                evidence.pop("text", None)
        self._record_trace(
            task_id,
            stage="inspect",
            operation="research_status",
            elapsed_ms=round((perf_counter() - started_at) * 1000),
            input_summary={
                "include_evidence_text": include_evidence_text,
                "wait_seconds": wait_seconds,
            },
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
    wait_seconds: int = 0,
) -> dict[str, Any]:
    """Return task state, optionally waiting briefly for an active retrieval."""
    return _tools().status(
        task_id,
        include_evidence_text=include_evidence_text,
        wait_seconds=wait_seconds,
    )


@mcp.tool()
def research_finalize(
    task_id: str,
    answered_sub_question_ids: list[str],
    citations: list[CitationReference],
) -> dict[str, Any]:
    """Validate reflected chunk citations and their reader-facing locators."""
    return _tools().finalize(
        task_id=task_id,
        answered_sub_question_ids=answered_sub_question_ids,
        citations=citations,
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
