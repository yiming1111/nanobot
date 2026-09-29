"""Durable storage for structured research-task state."""

from __future__ import annotations

import os
import re
from pathlib import Path

from filelock import FileLock

from nanobot.research.models import (
    CitationLocatorCheck,
    CitationReference,
    EvidenceAssessment,
    EvidenceItem,
    EvidenceReflection,
    ResearchState,
    ResearchTaskStatus,
    RetrievalAttempt,
    RetrievalScope,
    SubQuestion,
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

    @staticmethod
    def _write(path: Path, state: ResearchState) -> None:
        state.touch()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def save(self, state: ResearchState) -> None:
        path = self._path(state.task_id)
        with self._lock(state.task_id):
            self._write(path, state)

    def create(self, state: ResearchState) -> ResearchState:
        path = self._path(state.task_id)
        with self._lock(state.task_id):
            if path.exists():
                raise FileExistsError(f"research task already exists: {state.task_id}")
            self._write(path, state)
        return state

    def load(self, task_id: str) -> ResearchState:
        path = self._path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"research task not found: {task_id}")
        with self._lock(task_id):
            return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))

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

    def add_evidence(
        self,
        task_id: str,
        sub_question_id: str,
        evidence: list[EvidenceItem],
        *,
        query: str,
        paper_ids: list[str] | None = None,
        scope: RetrievalScope = RetrievalScope.FULL_CORPUS,
        elapsed_ms: int = 0,
        coverage_reason: str = "",
    ) -> ResearchState:
        """Persist one full-corpus semantic retrieval round."""
        with self._lock(task_id):
            path = self._path(task_id)
            if not path.exists():
                raise FileNotFoundError(f"research task not found: {task_id}")
            state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            sub_question = self._find_sub_question(state, sub_question_id)
            if len(sub_question.retrieval_attempts) >= state.max_retrieval_rounds:
                raise RuntimeError(
                    f"maximum semantic retrieval rounds reached for {sub_question_id}: "
                    f"{state.max_retrieval_rounds}"
                )
            if sub_question.retrieval_attempts and sub_question.next_query is None:
                raise RuntimeError(
                    f"run research_reflect before retrying {sub_question_id}"
                )

            if query not in sub_question.queries:
                sub_question.queries.append(query)
            for paper_id in paper_ids or []:
                if paper_id not in sub_question.candidate_paper_ids:
                    sub_question.candidate_paper_ids.append(paper_id)
            for item in evidence:
                state_key = f"{sub_question_id}:{item.chunk_id}"
                state.evidence[state_key] = item
                if item.chunk_id not in sub_question.chunk_ids:
                    sub_question.chunk_ids.append(item.chunk_id)
            sub_question.retrieval_attempts.append(
                RetrievalAttempt(
                    round_index=len(sub_question.retrieval_attempts) + 1,
                    query=query,
                    scope=scope,
                    paper_ids=paper_ids or [],
                    chunk_ids=[item.chunk_id for item in evidence],
                    elapsed_ms=elapsed_ms,
                    sufficient=False,
                    coverage_reason=coverage_reason,
                )
            )
            sub_question.next_query = None
            sub_question.status = (
                SubQuestionStatus.EVIDENCE_FOUND
                if evidence
                else SubQuestionStatus.INSUFFICIENT
            )
            state.retrieval_calls += 1
            state.retrieval_rounds += 1

            pending_ids = {
                item.sub_question_id
                for item in state.plan.sub_questions
                if item.status != SubQuestionStatus.SUFFICIENT
                and len(item.retrieval_attempts) == len(item.reflections)
                and len(item.retrieval_attempts) < state.max_retrieval_rounds
            }
            state.status = (
                ResearchTaskStatus.RETRIEVING
                if pending_ids
                else ResearchTaskStatus.REFLECTING
            )
            self._write(path, state)
            return state

    def record_reflections(
        self,
        task_id: str,
        assessments: list[EvidenceAssessment],
    ) -> ResearchState:
        """Persist one Reflect decision for every currently unreviewed evidence need."""
        with self._lock(task_id):
            path = self._path(task_id)
            if not path.exists():
                raise FileNotFoundError(f"research task not found: {task_id}")
            state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            if state.status != ResearchTaskStatus.REFLECTING:
                raise RuntimeError("research_reflect is only available after retrieval")
            assessment_ids = [item.sub_question_id for item in assessments]
            if len(assessment_ids) != len(set(assessment_ids)):
                raise ValueError("sub_question_id values must be unique")

            eligible = {
                item.sub_question_id: item
                for item in state.plan.sub_questions
                if len(item.retrieval_attempts) > len(item.reflections)
            }
            if set(assessment_ids) != set(eligible):
                raise ValueError(
                    "assessments must cover every unreviewed sub-question exactly once: "
                    f"{sorted(eligible)}"
                )

            for assessment in assessments:
                sub_question = eligible[assessment.sub_question_id]
                unknown_chunks = set(assessment.supporting_chunk_ids) - set(
                    sub_question.chunk_ids
                )
                if unknown_chunks:
                    raise ValueError(
                        "assessment references chunks outside its sub-question: "
                        f"{sorted(unknown_chunks)}"
                    )
                can_retry = (
                    len(sub_question.retrieval_attempts) < state.max_retrieval_rounds
                )
                if not assessment.sufficient and can_retry and not assessment.next_query:
                    raise ValueError(
                        f"next_query is required to retry {assessment.sub_question_id}"
                    )
                reflection = EvidenceReflection(
                    round_index=len(sub_question.reflections) + 1,
                    **assessment.model_dump(),
                )
                sub_question.reflections.append(reflection)
                latest_attempt = sub_question.retrieval_attempts[-1]
                latest_attempt.sufficient = assessment.sufficient
                latest_attempt.coverage_reason = assessment.reason
                if assessment.sufficient:
                    sub_question.status = SubQuestionStatus.SUFFICIENT
                    sub_question.supporting_chunk_ids = list(
                        assessment.supporting_chunk_ids
                    )
                    sub_question.next_query = None
                else:
                    sub_question.status = SubQuestionStatus.INSUFFICIENT
                    sub_question.supporting_chunk_ids = []
                    sub_question.next_query = (
                        assessment.next_query.strip()
                        if can_retry and assessment.next_query is not None
                        else None
                    )

            retry_ids = [
                item.sub_question_id
                for item in state.plan.sub_questions
                if item.status == SubQuestionStatus.INSUFFICIENT
                and item.next_query is not None
            ]
            terminal = all(
                item.status == SubQuestionStatus.SUFFICIENT
                or (
                    item.status == SubQuestionStatus.INSUFFICIENT
                    and len(item.retrieval_attempts) >= state.max_retrieval_rounds
                    and len(item.reflections) == len(item.retrieval_attempts)
                )
                for item in state.plan.sub_questions
            )
            if retry_ids:
                state.status = ResearchTaskStatus.RETRIEVING
            elif terminal:
                state.status = (
                    ResearchTaskStatus.READY_TO_SYNTHESIZE
                    if any(
                        item.status == SubQuestionStatus.SUFFICIENT
                        for item in state.plan.sub_questions
                    )
                    else ResearchTaskStatus.REFUSED
                )
            else:
                state.status = ResearchTaskStatus.REFLECTING
            self._write(path, state)
            return state

    def finalize(
        self,
        task_id: str,
        *,
        answered_sub_question_ids: list[str],
        citations: list[CitationReference],
        checks: list[CitationLocatorCheck],
    ) -> ResearchState:
        """Record deterministic citation validation after answer generation."""
        with self._lock(task_id):
            path = self._path(task_id)
            if not path.exists():
                raise FileNotFoundError(f"research task not found: {task_id}")
            state = ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            if state.status != ResearchTaskStatus.READY_TO_SYNTHESIZE:
                raise RuntimeError("research_finalize requires reflected sufficient evidence")
            state.citation_locator_checks = checks
            answered = set(answered_sub_question_ids)
            unresolved = [
                item.sub_question_id
                for item in state.plan.sub_questions
                if item.sub_question_id not in answered
            ]
            state.metadata["finalization"] = {
                "answered_sub_question_ids": answered_sub_question_ids,
                "citations": [item.model_dump(mode="json") for item in citations],
                "unresolved_sub_question_ids": unresolved,
            }
            state.status = (
                ResearchTaskStatus.COMPLETED
                if not unresolved
                else ResearchTaskStatus.COMPLETED_WITH_GAPS
            )
            self._write(path, state)
            return state
