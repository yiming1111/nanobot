from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.research.models import (
    CitationVerdict,
    EvidenceJudgment,
    EvidenceItem,
    ResearchPlan,
    ResearchState,
    SubQuestion,
    SubQuestionStatus,
)
from nanobot.research.workflow.state_store import ResearchStateStore


def _plan() -> ResearchPlan:
    return ResearchPlan(
        original_question="Compare A and B",
        normalized_question="Compare A and B using papers",
        requires_decomposition=True,
        sub_questions=[
            SubQuestion(sub_question_id="SQ1", question="How accurate is A?"),
            SubQuestion(sub_question_id="SQ2", question="How expensive is B?"),
        ],
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


def test_state_store_round_trip_and_evidence_update(tmp_path: Path) -> None:
    store = ResearchStateStore(tmp_path / "states")
    state = ResearchState(task_id="task-1", plan=_plan())
    store.create(state)

    evidence = EvidenceItem(
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
    updated = store.add_evidence("task-1", "SQ1", [evidence], query="A recall")

    assert updated.retrieval_calls == 1
    assert updated.retrieval_rounds == 1
    assert updated.plan.sub_questions[0].status == SubQuestionStatus.EVIDENCE_FOUND
    assert updated.plan.sub_questions[0].evidence_ids == ["E-1"]
    assert store.load("task-1").evidence["E-1"].page_start == 3


def test_state_store_rejects_unsafe_task_id(tmp_path: Path) -> None:
    store = ResearchStateStore(tmp_path / "states")
    with pytest.raises(ValueError, match="invalid research task id"):
        store.load("../escape")


def test_semantic_judgment_cannot_fake_missing_citation() -> None:
    with pytest.raises(ValidationError, match="generated"):
        EvidenceJudgment(
            claim_id="C1",
            evidence_id="E1",
            verdict=CitationVerdict.MISSING_CITATION,
            rationale="missing",
        )
