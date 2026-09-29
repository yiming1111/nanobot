"""FastMCP surface for nanobot's scientific-paper research tools."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

from mcp.server.fastmcp import FastMCP

from nanobot.research.config import ResearchConfig
from nanobot.research.models import (
    CitationCheck,
    CitationVerdict,
    ClaimDraft,
    ClaimState,
    ClaimStatus,
    ConstraintValue,
    EvidenceJudgment,
    EvidenceItem,
    EvidenceRelation,
    EvidenceSearchResult,
    ResearchPlan,
    ResearchState,
    ResearchTaskStatus,
    RetrievalScope,
    SubQuestion,
)
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
        self._retrieval = retrieval

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
            max_verification_rounds=self.config.max_verification_rounds,
        )
        self.states.create(state)
        return state.model_dump(mode="json")

    def search_papers(
        self,
        query: str,
        *,
        top_k: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.retrieval.search_papers(
                query,
                top_k=top_k,
                year_from=year_from,
                year_to=year_to,
            )
        ]

    def retrieve_evidence(
        self,
        *,
        task_id: str,
        sub_question_id: str,
        query: str,
        top_k: int | None = None,
        paper_ids: list[str] | None = None,
        sections: list[str] | None = None,
        scope: RetrievalScope = RetrievalScope.CANDIDATE_PAPERS,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        results = self.retrieval.retrieve_evidence(
            query,
            top_k=top_k,
            paper_ids=paper_ids,
            sections=sections,
        )
        sufficient, coverage_reason = self._assess_evidence(results)
        evidence = self._evidence_items(
            task_id=task_id,
            sub_question_id=sub_question_id,
            results=results,
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000)
        state = self.states.add_evidence(
            task_id,
            sub_question_id,
            evidence,
            query=query,
            paper_ids=paper_ids,
            scope=scope,
            elapsed_ms=elapsed_ms,
            sufficient=sufficient,
            coverage_reason=coverage_reason,
        )
        return {
            "task_id": task_id,
            "sub_question_id": sub_question_id,
            "query": query,
            "scope": scope.value,
            "paper_ids": paper_ids or [],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "coverage": {
                "sufficient": sufficient,
                "reason": coverage_reason,
                "basis": "retrieval_score",
            },
            "elapsed_ms": elapsed_ms,
            "sub_question_status": next(
                item.status.value
                for item in state.plan.sub_questions
                if item.sub_question_id == sub_question_id
            ),
        }

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

    def _assess_evidence(
        self,
        results: list[EvidenceSearchResult],
    ) -> tuple[bool, str]:
        if not results:
            return False, "no evidence passages were retrieved"
        rerank_scores = [
            item.rerank_score for item in results if item.rerank_score is not None
        ]
        if rerank_scores and max(rerank_scores) < self.config.min_evidence_rerank_score:
            return (
                False,
                "the highest rerank score is below the configured retrieval threshold",
            )
        return (
            True,
            "at least one candidate passage was retrieved; semantic support "
            "must be verified before synthesis",
        )

    def research_retrieve(
        self,
        *,
        task_id: str,
        sub_question_id: str,
        query: str,
        alternative_query: str | None = None,
        top_k: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        sections: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run bounded paper-first retrieval for one sub-question."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        alternative_query = (
            alternative_query.strip() if alternative_query and alternative_query.strip() else None
        )
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
        if sub_question.retrieval_attempts:
            raise RuntimeError(
                f"retrieval already ran for {sub_question_id}; inspect research_status instead"
            )

        logger.info(
            "research retrieval started task=%s sub_question=%s",
            task_id,
            sub_question_id,
        )
        paper_search_started = perf_counter()
        candidate_papers = self.retrieval.search_papers(
            query,
            top_k=self.config.expanded_paper_candidates,
            year_from=year_from,
            year_to=year_to,
        )
        paper_search_ms = round((perf_counter() - paper_search_started) * 1000)
        logger.info(
            "paper recall completed task=%s sub_question=%s candidates=%d elapsed_ms=%d",
            task_id,
            sub_question_id,
            len(candidate_papers),
            paper_search_ms,
        )
        initial_ids = [
            item.paper_id
            for item in candidate_papers[: self.config.initial_paper_candidates]
        ]
        expanded_ids = [item.paper_id for item in candidate_papers]
        attempts: list[dict[str, Any]] = []

        first = self._run_attempt(
            task_id=task_id,
            sub_question_id=sub_question_id,
            query=query,
            paper_ids=initial_ids,
            scope=RetrievalScope.CANDIDATE_PAPERS,
            top_k=top_k,
            sections=sections,
        )
        attempts.append(first)
        self._log_attempt(task_id, sub_question_id, first)

        if (
            not first["coverage"]["sufficient"]
            and len(attempts) < state.max_retrieval_rounds
            and len(expanded_ids) > len(initial_ids)
        ):
            second = self._run_attempt(
                task_id=task_id,
                sub_question_id=sub_question_id,
                query=alternative_query or query,
                paper_ids=expanded_ids,
                scope=RetrievalScope.EXPANDED_CANDIDATES,
                top_k=top_k,
                sections=sections,
            )
            attempts.append(second)
            self._log_attempt(task_id, sub_question_id, second)

        if (
            not attempts[-1]["coverage"]["sufficient"]
            and len(attempts) < state.max_retrieval_rounds
        ):
            fallback = self._run_attempt(
                task_id=task_id,
                sub_question_id=sub_question_id,
                query=alternative_query or query,
                paper_ids=None,
                scope=RetrievalScope.GLOBAL_FALLBACK,
                top_k=top_k,
                sections=sections,
            )
            attempts.append(fallback)
            self._log_attempt(task_id, sub_question_id, fallback)

        attempt_summaries: list[dict[str, Any]] = []
        for attempt in attempts:
            attempt_summaries.append(
                {
                    "query": attempt["query"],
                    "scope": attempt["scope"],
                    "paper_ids": attempt["paper_ids"],
                    "evidence_ids": [
                        item["evidence_id"] for item in attempt["evidence"]
                    ],
                    "coverage": attempt["coverage"],
                    "elapsed_ms": attempt["elapsed_ms"],
                }
            )
        final_state = self.states.load(task_id)
        final_sub_question = next(
            item
            for item in final_state.plan.sub_questions
            if item.sub_question_id == sub_question_id
        )
        return {
            "task_id": task_id,
            "sub_question_id": sub_question_id,
            "candidate_papers": [
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "year": item.year,
                    "fused_score": item.fused_score,
                    "rerank_score": item.rerank_score,
                }
                for item in candidate_papers
            ],
            "paper_search_ms": paper_search_ms,
            "attempts": attempt_summaries,
            "evidence": attempts[-1]["evidence"],
            "coverage": attempts[-1]["coverage"],
            "sub_question_status": final_sub_question.status.value,
            "stopped_after_rounds": len(attempts),
        }

    @staticmethod
    def _log_attempt(
        task_id: str,
        sub_question_id: str,
        attempt: dict[str, Any],
    ) -> None:
        logger.info(
            "evidence retrieval completed task=%s sub_question=%s scope=%s "
            "evidence=%d sufficient=%s elapsed_ms=%d",
            task_id,
            sub_question_id,
            attempt["scope"],
            len(attempt["evidence"]),
            attempt["coverage"]["sufficient"],
            attempt["elapsed_ms"],
        )

    def _run_attempt(
        self,
        *,
        task_id: str,
        sub_question_id: str,
        query: str,
        paper_ids: list[str] | None,
        scope: RetrievalScope,
        top_k: int | None,
        sections: list[str] | None,
    ) -> dict[str, Any]:
        if paper_ids == []:
            started_at = perf_counter()
            sufficient, coverage_reason = self._assess_evidence([])
            state = self.states.add_evidence(
                task_id,
                sub_question_id,
                [],
                query=query,
                paper_ids=[],
                scope=scope,
                elapsed_ms=round((perf_counter() - started_at) * 1000),
                sufficient=sufficient,
                coverage_reason=coverage_reason,
            )
            return {
                "task_id": task_id,
                "sub_question_id": sub_question_id,
                "query": query,
                "scope": scope.value,
                "paper_ids": [],
                "evidence": [],
                "coverage": {
                    "sufficient": False,
                    "reason": coverage_reason,
                    "basis": "retrieval_score",
                },
                "elapsed_ms": 0,
                "sub_question_status": next(
                    item.status.value
                    for item in state.plan.sub_questions
                    if item.sub_question_id == sub_question_id
                ),
            }
        return self.retrieve_evidence(
            task_id=task_id,
            sub_question_id=sub_question_id,
            query=query,
            top_k=top_k,
            paper_ids=paper_ids,
            sections=sections,
            scope=scope,
        )

    def neighbors(self, chunk_id: str, *, window: int = 1) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.retrieval.get_neighbors(chunk_id, window=window)
        ]

    def verify_claims(
        self,
        *,
        task_id: str,
        claims: list[ClaimDraft],
        judgments: list[EvidenceJudgment],
    ) -> dict[str, Any]:
        """Build a claim-evidence matrix from a separate semantic review pass."""
        if not claims:
            raise ValueError("at least one claim is required")
        claim_ids = [item.claim_id for item in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")

        state = self.states.load(task_id)
        if state.status not in {
            ResearchTaskStatus.READY_TO_SYNTHESIZE,
            ResearchTaskStatus.VERIFYING,
        }:
            raise RuntimeError(
                "claims can only be verified after retrieval and before final completion"
            )
        sub_question_ids = {
            item.sub_question_id for item in state.plan.sub_questions
        }
        for claim in claims:
            if claim.sub_question_id not in sub_question_ids:
                raise ValueError(
                    f"unknown sub-question for claim {claim.claim_id}: "
                    f"{claim.sub_question_id}"
                )

        judgment_by_pair: dict[tuple[str, str], EvidenceJudgment] = {}
        cited_pairs = {
            (claim.claim_id, evidence_id)
            for claim in claims
            for evidence_id in claim.cited_evidence_ids
        }
        for judgment in judgments:
            pair = (judgment.claim_id, judgment.evidence_id)
            if pair in judgment_by_pair:
                raise ValueError(
                    "duplicate semantic judgment for "
                    f"claim={judgment.claim_id}, evidence={judgment.evidence_id}"
                )
            if pair not in cited_pairs:
                raise ValueError(
                    "semantic judgment must reference evidence cited by its claim: "
                    f"claim={judgment.claim_id}, evidence={judgment.evidence_id}"
                )
            judgment_by_pair[pair] = judgment

        claim_states: list[ClaimState] = []
        citation_checks: list[CitationCheck] = []
        matrix: list[dict[str, Any]] = []
        for claim in claims:
            checks: list[CitationCheck] = []
            if not claim.cited_evidence_ids:
                checks.append(
                    CitationCheck(
                        claim_id=claim.claim_id,
                        verdict=CitationVerdict.MISSING_CITATION,
                        rationale="the claim has no cited evidence",
                    )
                )
            for evidence_id in claim.cited_evidence_ids:
                evidence = state.evidence.get(evidence_id)
                judgment = judgment_by_pair.get((claim.claim_id, evidence_id))
                if evidence is None:
                    check = CitationCheck(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        locatable=False,
                        verdict=CitationVerdict.UNSUPPORTED,
                        rationale="the cited evidence ID is not present in this task",
                    )
                elif evidence.sub_question_id != claim.sub_question_id:
                    check = CitationCheck(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        locatable=True,
                        verdict=CitationVerdict.UNSUPPORTED,
                        rationale=(
                            "the citation belongs to a different evidence need: "
                            f"{evidence.sub_question_id}"
                        ),
                    )
                elif judgment is None:
                    check = CitationCheck(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        locatable=True,
                        verdict=CitationVerdict.NOT_ENOUGH_INFORMATION,
                        rationale=(
                            "no independent semantic judgment was supplied for this "
                            "claim-evidence pair"
                        ),
                    )
                else:
                    locatable = bool(
                        evidence.title
                        and evidence.chunk_id
                        and evidence.page_start >= 1
                        and evidence.page_end >= evidence.page_start
                    )
                    verdict = judgment.verdict
                    rationale = judgment.rationale
                    if not locatable and verdict == CitationVerdict.SUPPORTED:
                        verdict = CitationVerdict.UNSUPPORTED
                        rationale = (
                            "semantic support was asserted, but the source locator is "
                            "incomplete"
                        )
                    check = CitationCheck(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        locatable=locatable,
                        verdict=verdict,
                        rationale=rationale,
                        revision=judgment.revision,
                    )
                checks.append(check)

            verdicts = [item.verdict for item in checks]
            has_support = CitationVerdict.SUPPORTED in verdicts
            has_partial = CitationVerdict.PARTIALLY_SUPPORTED in verdicts
            has_contradiction = CitationVerdict.CONTRADICTED in verdicts
            if verdicts and all(
                item.verdict == CitationVerdict.SUPPORTED and item.locatable
                for item in checks
            ):
                claim_status = ClaimStatus.SUPPORTED
            elif has_contradiction and (has_support or has_partial):
                claim_status = ClaimStatus.CONFLICTING
            elif has_contradiction:
                claim_status = ClaimStatus.CONTRADICTED
            elif has_support or has_partial:
                claim_status = ClaimStatus.PARTIALLY_SUPPORTED
            else:
                claim_status = ClaimStatus.INSUFFICIENT

            revision = next(
                (item.revision for item in checks if item.revision),
                None,
            )
            claim_state = ClaimState(
                claim_id=claim.claim_id,
                sub_question_id=claim.sub_question_id,
                text=claim.text,
                cited_evidence_ids=claim.cited_evidence_ids,
                supporting_evidence_ids=[
                    item.evidence_id
                    for item in checks
                    if item.evidence_id
                    and item.verdict
                    in {
                        CitationVerdict.SUPPORTED,
                        CitationVerdict.PARTIALLY_SUPPORTED,
                    }
                ],
                contradicting_evidence_ids=[
                    item.evidence_id
                    for item in checks
                    if item.evidence_id
                    and item.verdict == CitationVerdict.CONTRADICTED
                ],
                status=claim_status,
                revision=revision,
            )
            claim_states.append(claim_state)
            citation_checks.extend(checks)
            matrix.append(
                {
                    "claim_id": claim.claim_id,
                    "sub_question_id": claim.sub_question_id,
                    "claim": claim.text,
                    "status": claim_status.value,
                    "recommended_action": self._claim_action(claim_status, revision),
                    "checks": [
                        self._citation_check_payload(item, state) for item in checks
                    ],
                }
            )

        all_claims_supported = all(
            item.status == ClaimStatus.SUPPORTED for item in claim_states
        )
        covered_sub_question_ids = {
            item.sub_question_id for item in claim_states if item.sub_question_id
        }
        uncovered_sub_question_ids = sorted(
            sub_question_ids - covered_sub_question_ids
        )
        all_evidence_needs_covered = not uncovered_sub_question_ids
        ready_for_final = all_claims_supported and all_evidence_needs_covered
        supported_count = sum(
            item.status == ClaimStatus.SUPPORTED for item in claim_states
        )
        claims_with_locatable_citation = {
            item.claim_id
            for item in citation_checks
            if item.evidence_id is not None and item.locatable
        }
        cited_checks = [
            item for item in citation_checks if item.evidence_id is not None
        ]
        supported_citations = sum(
            item.verdict == CitationVerdict.SUPPORTED for item in cited_checks
        )
        next_round = state.verification_rounds + 1
        if ready_for_final:
            final_status = ResearchTaskStatus.COMPLETED
        elif next_round >= state.max_verification_rounds:
            final_status = (
                ResearchTaskStatus.COMPLETED_WITH_GAPS
                if supported_count
                else ResearchTaskStatus.REFUSED
            )
        else:
            final_status = ResearchTaskStatus.VERIFYING

        updated = self.states.record_verification(
            task_id,
            claims=claim_states,
            citation_checks=citation_checks,
            status=final_status,
        )
        return {
            "task_id": task_id,
            "status": updated.status.value,
            "verification_round": updated.verification_rounds,
            "remaining_verification_rounds": (
                updated.max_verification_rounds - updated.verification_rounds
            ),
            "all_claims_supported": all_claims_supported,
            "all_evidence_needs_covered": all_evidence_needs_covered,
            "uncovered_sub_question_ids": uncovered_sub_question_ids,
            "can_answer_with_supported_claims": supported_count > 0,
            "claim_support_rate": supported_count / len(claim_states),
            "citation_completeness": (
                len(claims_with_locatable_citation) / len(claim_states)
            ),
            "citation_correctness": (
                supported_citations / len(cited_checks) if cited_checks else 0.0
            ),
            "claim_evidence_matrix": matrix,
        }

    @staticmethod
    def _citation_check_payload(
        check: CitationCheck,
        state: ResearchState,
    ) -> dict[str, Any]:
        payload = check.model_dump(mode="json")
        evidence = state.evidence.get(check.evidence_id or "")
        payload["locator"] = (
            {
                "title": evidence.title,
                "page_start": evidence.page_start,
                "page_end": evidence.page_end,
                "chunk_id": evidence.chunk_id,
            }
            if evidence is not None
            else None
        )
        return payload

    @staticmethod
    def _claim_action(status: ClaimStatus, revision: str | None) -> str:
        if status == ClaimStatus.SUPPORTED:
            return "keep"
        if status == ClaimStatus.PARTIALLY_SUPPORTED:
            return "use_revision" if revision else "narrow_or_retrieve"
        if status == ClaimStatus.CONFLICTING:
            return "report_conflict_or_retrieve"
        if status == ClaimStatus.CONTRADICTED:
            return "remove_or_reverse"
        return "retrieve_or_remove"

    def retrieve_claim_gap(
        self,
        *,
        task_id: str,
        claim_id: str,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run one bounded evidence search for a claim that failed verification."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        state = self.states.load(task_id)
        if state.status != ResearchTaskStatus.VERIFYING:
            raise RuntimeError("gap retrieval is only available while verifying claims")
        claim = state.claims.get(claim_id)
        if claim is None:
            raise ValueError(f"unknown claim: {claim_id}")
        if claim.sub_question_id is None:
            raise ValueError("claim is not attached to a sub-question")
        if claim.status == ClaimStatus.SUPPORTED:
            raise RuntimeError("a supported claim does not need gap retrieval")
        gap_history = list(state.metadata.get("gap_retrieval_attempts", []))
        if any(item.get("claim_id") == claim_id for item in gap_history):
            raise RuntimeError(f"gap retrieval already ran for claim: {claim_id}")
        if len(gap_history) >= self.config.max_gap_retrievals:
            raise RuntimeError(
                f"maximum task gap retrievals reached: {self.config.max_gap_retrievals}"
            )
        sub_question = next(
            item
            for item in state.plan.sub_questions
            if item.sub_question_id == claim.sub_question_id
        )
        candidate_ids = sub_question.candidate_paper_ids
        started_at = perf_counter()
        scope = (
            RetrievalScope.CANDIDATE_PAPERS
            if candidate_ids
            else RetrievalScope.GLOBAL_FALLBACK
        )
        results = self.retrieval.retrieve_evidence(
            query,
            top_k=top_k,
            paper_ids=candidate_ids or None,
        )
        if not results and candidate_ids:
            scope = RetrievalScope.GLOBAL_FALLBACK
            results = self.retrieval.retrieve_evidence(query, top_k=top_k)
        evidence = self._evidence_items(
            task_id=task_id,
            sub_question_id=claim.sub_question_id,
            results=results,
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000)
        self.states.add_gap_evidence(
            task_id,
            claim.sub_question_id,
            claim_id,
            evidence,
            query=query,
            scope=scope,
            paper_ids=candidate_ids if scope == RetrievalScope.CANDIDATE_PAPERS else None,
            elapsed_ms=elapsed_ms,
            max_gap_retrievals=self.config.max_gap_retrievals,
        )
        return {
            "task_id": task_id,
            "claim_id": claim_id,
            "query": query,
            "scope": scope.value,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "elapsed_ms": elapsed_ms,
            "next_step": "revise the claim or run the final verification round",
        }

    def status(
        self,
        task_id: str,
        *,
        include_evidence_text: bool = False,
    ) -> dict[str, Any]:
        payload = self.states.load(task_id).model_dump(mode="json")
        if not include_evidence_text:
            for evidence in payload["evidence"].values():
                evidence.pop("text", None)
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
    """Create a typed research plan before searching a scientific-paper corpus."""
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
    alternative_query: str | None = None,
    top_k: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Run bounded paper-first evidence retrieval for one research sub-question."""
    return _tools().research_retrieve(
        task_id=task_id,
        sub_question_id=sub_question_id,
        query=query,
        alternative_query=alternative_query,
        top_k=top_k,
        year_from=year_from,
        year_to=year_to,
        sections=sections,
    )


@mcp.tool()
def get_neighbor_evidence(chunk_id: str, window: int = 1) -> list[dict[str, Any]]:
    """Read adjacent chunks to check the context around a retrieved passage."""
    return _tools().neighbors(chunk_id, window=window)


@mcp.tool()
def research_status(
    task_id: str,
    include_evidence_text: bool = False,
) -> dict[str, Any]:
    """Return research state; evidence text is omitted by default to bound context size."""
    return _tools().status(task_id, include_evidence_text=include_evidence_text)


@mcp.tool()
def research_verify(
    task_id: str,
    claims: list[ClaimDraft],
    judgments: list[EvidenceJudgment],
) -> dict[str, Any]:
    """Validate a claim-evidence matrix after an independent semantic review pass."""
    return _tools().verify_claims(
        task_id=task_id,
        claims=claims,
        judgments=judgments,
    )


@mcp.tool()
def research_retrieve_claim_gap(
    task_id: str,
    claim_id: str,
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Search once for evidence missing from a claim that failed verification."""
    return _tools().retrieve_claim_gap(
        task_id=task_id,
        claim_id=claim_id,
        query=query,
        top_k=top_k,
    )


def main() -> None:
    # Complete expensive local initialization before advertising MCP tools.
    # The gateway waits for MCP initialization, so users cannot start a research
    # call that would otherwise pay the model-loading cost and hit tool timeout.
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
