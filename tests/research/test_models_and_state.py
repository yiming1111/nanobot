from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.research.models import (
    EvidenceAssessment,
    EvidenceItem,
    ResearchPlan,
    ResearchState,
    ResearchTaskStatus,
    SubQuestion,
    SubQuestionStatus,
)
from nanobot.research.workflow.state_store import ResearchStateStore


def _plan() -> ResearchPlan:
    return ResearchPlan(
        original_question="How accurate is A?",
        normalized_question="How accurate is A according to papers?",
        sub_questions=[
            SubQuestion(sub_question_id="SQ1", question="How accurate is A?")
        ],
    )


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="E-1",
        sub_question_id="SQ1",
        paper_id="P1",
        chunk_id="P1-C0001",
        title="Paper One",
        section="Experiments",
        page_start=3,
        page_end=3,
        text="A improves recall on the benchmark.",
    )


def test_plan_rejects_duplicate_sub_question_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        ResearchPlan(
            original_question="question",
            normalized_question="question",
            sub_questions=[
                SubQuestion(sub_question_id="SQ1", question="one"),
                SubQuestion(sub_question_id="SQ1", question="two"),
            ],
        )


def test_state_store_retrieves_then_reflects_sufficient_evidence(tmp_path: Path) -> None:
    store = ResearchStateStore(tmp_path / "states")
    state = ResearchState(
        task_id="task-1",
        plan=_plan(),
        status=ResearchTaskStatus.RETRIEVING,
        max_retrieval_rounds=2,
    )
    store.create(state)

    retrieved = store.add_evidence(
        "task-1",
        "SQ1",
        [_evidence()],
        query="A recall",
    )
    reflected = store.record_reflections(
        "task-1",
        [
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                evidence_ids=["E-1"],
                reason="The passage directly answers the evidence need.",
            )
        ],
    )

    assert retrieved.status == ResearchTaskStatus.REFLECTING
    assert reflected.status == ResearchTaskStatus.READY_TO_SYNTHESIZE
    assert reflected.plan.sub_questions[0].status == SubQuestionStatus.SUFFICIENT
    assert reflected.plan.sub_questions[0].retrieval_attempts[0].sufficient is True


def test_state_store_allows_only_one_reflect_triggered_retry(tmp_path: Path) -> None:
    store = ResearchStateStore(tmp_path / "states")
    store.create(
        ResearchState(
            task_id="task-2",
            plan=_plan(),
            status=ResearchTaskStatus.RETRIEVING,
            max_retrieval_rounds=2,
        )
    )
    store.add_evidence("task-2", "SQ1", [], query="first query")
    first = store.record_reflections(
        "task-2",
        [
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=False,
                reason="No direct evidence was found.",
                next_query="focused second query",
            )
        ],
    )
    assert first.status == ResearchTaskStatus.RETRIEVING
    assert first.plan.sub_questions[0].next_query == "focused second query"

    store.add_evidence("task-2", "SQ1", [], query="focused second query")
    second = store.record_reflections(
        "task-2",
        [
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=False,
                reason="The second search still found no direct evidence.",
            )
        ],
    )
    assert second.status == ResearchTaskStatus.REFUSED
    assert second.plan.sub_questions[0].next_query is None


def test_state_store_rejects_unsafe_task_id(tmp_path: Path) -> None:
    store = ResearchStateStore(tmp_path / "states")
    with pytest.raises(ValueError, match="invalid research task id"):
        store.load("../escape")
