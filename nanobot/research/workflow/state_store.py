"""Durable storage for structured research-task state."""

from __future__ import annotations

import os
import re
from pathlib import Path

from filelock import FileLock

from nanobot.research.models import (
    CitationCheck,
    ClaimStatus,
    ClaimState,
    EvidenceItem,
    ResearchState,
    ResearchTaskStatus,
    RetrievalAttempt,
    RetrievalScope,
    SubQuestionStatus,
)

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ResearchStateStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        if _SAFE_TASK_ID.fullmatch(task_id) is None:
            raise ValueError("invalid research task id")
        return self.root / f"{task_id}.json"

    def _lock(self, task_id: str) -> FileLock:
        return FileLock(str(self._path(task_id)) + ".lock")

    def save(self, state: ResearchState) -> None:
        state.touch()
        path = self._path(state.task_id)
        payload = state.model_dump_json(indent=2)
        with self._lock(state.task_id):
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)

    def create(self, state: ResearchState) -> ResearchState:
        path = self._path(state.task_id)
        with self._lock(state.task_id):
            if path.exists():
                raise FileExistsError(f"research task already exists: {state.task_id}")
            state.touch()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
        return state

    def load(self, task_id: str) -> ResearchState:
        path = self._path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"research task not found: {task_id}")
        with self._lock(task_id):
            return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_for_session(self, session_key: str) -> ResearchState | None:
        """Return the most recently updated research task for one nanobot session."""
        normalized_key = session_key.strip()
        if not normalized_key:
            raise ValueError("session key must not be empty")
        latest: ResearchState | None = None
        for path in self.root.glob("*.json"):
            try:
                state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if state.session_key != normalized_key:
                continue
            if latest is None or state.updated_at > latest.updated_at:
                latest = state
        return latest

    def add_evidence(
        self,
        task_id: str,
        sub_question_id: str,
        evidence: list[EvidenceItem],
        *,
        query: str,
        paper_ids: list[str] | None = None,
        scope: RetrievalScope = RetrievalScope.CANDIDATE_PAPERS,
        elapsed_ms: int = 0,
        sufficient: bool = False,
        coverage_reason: str = "",
    ) -> ResearchState:
        with self._lock(task_id):
            path = self._path(task_id)
            if not path.exists():
                raise FileNotFoundError(f"research task not found: {task_id}")
            state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
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
            if len(sub_question.retrieval_attempts) >= state.max_retrieval_rounds:
                raise RuntimeError(
                    f"maximum retrieval rounds reached for {sub_question_id}: "
                    f"{state.max_retrieval_rounds}"
                )
            if query not in sub_question.queries:
                sub_question.queries.append(query)
            for paper_id in paper_ids or []:
                if paper_id not in sub_question.candidate_paper_ids:
                    sub_question.candidate_paper_ids.append(paper_id)
            for item in evidence:
                state.evidence[item.evidence_id] = item
                if item.evidence_id not in sub_question.evidence_ids:
                    sub_question.evidence_ids.append(item.evidence_id)
            sub_question.retrieval_attempts.append(
                RetrievalAttempt(
                    round_index=len(sub_question.retrieval_attempts) + 1,
                    query=query,
                    scope=scope,
                    paper_ids=paper_ids or [],
                    evidence_ids=[item.evidence_id for item in evidence],
                    elapsed_ms=elapsed_ms,
                    sufficient=sufficient,
                    coverage_reason=coverage_reason,
                )
            )
            if sufficient:
                sub_question.status = SubQuestionStatus.SUFFICIENT
            elif len(sub_question.retrieval_attempts) >= state.max_retrieval_rounds:
                sub_question.status = SubQuestionStatus.INSUFFICIENT
            elif sub_question.evidence_ids:
                sub_question.status = SubQuestionStatus.EVIDENCE_FOUND
            else:
                sub_question.status = SubQuestionStatus.INSUFFICIENT
            state.retrieval_calls += 1
            state.retrieval_rounds += 1
            if all(
                item.status == SubQuestionStatus.SUFFICIENT
                or len(item.retrieval_attempts) >= state.max_retrieval_rounds
                for item in state.plan.sub_questions
            ):
                state.status = ResearchTaskStatus.READY_TO_SYNTHESIZE
            state.touch()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
            return state

    def record_verification(
        self,
        task_id: str,
        *,
        claims: list[ClaimState],
        citation_checks: list[CitationCheck],
        status: ResearchTaskStatus,
    ) -> ResearchState:
        """Persist one complete claim-verification round atomically."""
        with self._lock(task_id):
            path = self._path(task_id)
            if not path.exists():
                raise FileNotFoundError(f"research task not found: {task_id}")
            state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            if state.verification_rounds >= state.max_verification_rounds:
                raise RuntimeError(
                    f"maximum verification rounds reached: {state.max_verification_rounds}"
                )
            state.claims = {item.claim_id: item for item in claims}
            state.citation_checks = citation_checks
            state.verification_rounds += 1
            state.status = status
            state.touch()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
            return state

    def add_gap_evidence(
        self,
        task_id: str,
        sub_question_id: str,
        claim_id: str,
        evidence: list[EvidenceItem],
        *,
        query: str,
        scope: RetrievalScope,
        paper_ids: list[str] | None,
        elapsed_ms: int,
        max_gap_retrievals: int,
    ) -> ResearchState:
        """Add evidence from one bounded post-verification gap search."""
        with self._lock(task_id):
            path = self._path(task_id)
            if not path.exists():
                raise FileNotFoundError(f"research task not found: {task_id}")
            state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            if state.status != ResearchTaskStatus.VERIFYING:
                raise RuntimeError("gap retrieval is only available while verifying claims")
            claim = state.claims.get(claim_id)
            if claim is None:
                raise ValueError(f"unknown claim: {claim_id}")
            if claim.sub_question_id != sub_question_id:
                raise ValueError("claim does not belong to the supplied sub-question")
            if claim.status == ClaimStatus.SUPPORTED:
                raise RuntimeError("a supported claim does not need gap retrieval")

            history = list(state.metadata.get("gap_retrieval_attempts", []))
            if any(item.get("claim_id") == claim_id for item in history):
                raise RuntimeError(f"gap retrieval already ran for claim: {claim_id}")
            if len(history) >= max_gap_retrievals:
                raise RuntimeError(
                    f"maximum task gap retrievals reached: {max_gap_retrievals}"
                )

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
            if query not in sub_question.queries:
                sub_question.queries.append(query)
            for item in evidence:
                state.evidence[item.evidence_id] = item
                if item.evidence_id not in sub_question.evidence_ids:
                    sub_question.evidence_ids.append(item.evidence_id)
            history.append(
                {
                    "claim_id": claim_id,
                    "sub_question_id": sub_question_id,
                    "query": query,
                    "scope": scope.value,
                    "paper_ids": paper_ids or [],
                    "evidence_ids": [item.evidence_id for item in evidence],
                    "elapsed_ms": elapsed_ms,
                }
            )
            state.metadata["gap_retrieval_attempts"] = history
            state.retrieval_calls += 1
            state.status = ResearchTaskStatus.VERIFYING
            state.touch()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
            return state
