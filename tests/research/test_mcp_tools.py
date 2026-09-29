import json
from pathlib import Path

import pytest

from nanobot.research.config import ResearchConfig
from nanobot.research.mcp_server import ResearchTools
from nanobot.research.models import (
    CitationVerdict,
    ClaimDraft,
    EvidenceJudgment,
    EvidenceSearchResult,
    PaperSearchResult,
)


class _FakeRetrieval:
    def __init__(self, *, global_only: bool = False) -> None:
        self.global_only = global_only
        self.paper_calls: list[dict[str, object]] = []
        self.evidence_calls: list[dict[str, object]] = []

    def search_papers(self, query: str, **kwargs: object) -> list[PaperSearchResult]:
        self.paper_calls.append({"query": query, **kwargs})
        return [
            PaperSearchResult(
                paper_id="P1",
                title="Paper One",
                fused_score=0.1,
                rerank_score=0.9,
            ),
            PaperSearchResult(
                paper_id="P2",
                title="Paper Two",
                fused_score=0.08,
                rerank_score=0.7,
            ),
        ]

    def retrieve_evidence(self, query: str, **kwargs: object) -> list[EvidenceSearchResult]:
        self.evidence_calls.append({"query": query, **kwargs})
        paper_ids = kwargs.get("paper_ids")
        if self.global_only and paper_ids is not None:
            return []
        return [
            EvidenceSearchResult(
                chunk_id="P1-C0000",
                paper_id="P1",
                title="Paper One",
                section="Results",
                page_start=4,
                page_end=4,
                text="The method improves evidence recall.",
                fused_score=0.1,
                rerank_score=0.9,
            )
        ]

    def get_neighbors(self, chunk_id: str, **kwargs: object) -> list[EvidenceSearchResult]:
        return self.retrieve_evidence(chunk_id)


def test_research_tools_create_plan_and_accumulate_evidence(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="Compare A and B",
        normalized_question="Compare A and B with papers",
        sub_questions=["Accuracy?", "Cost?"],
        constraints={
            "year_from": 2024,
            "citation_fields": ["title", "page", "chunk_id"],
        },
        session_key="webui:test",
    )
    task_id = str(started["task_id"])

    result = tools.retrieve_evidence(
        task_id=task_id,
        sub_question_id="SQ1",
        query="A accuracy",
    )
    status = tools.status(task_id)

    assert result["evidence"][0]["chunk_id"] == "P1-C0000"
    assert status["plan"]["requires_decomposition"] is True
    assert status["plan"]["constraints"]["citation_fields"] == [
        "title",
        "page",
        "chunk_id",
    ]
    assert status["plan"]["sub_questions"][0]["status"] == "sufficient"
    assert status["retrieval_calls"] == 1


def test_research_tools_resume_and_link_follow_up_to_session(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    first = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
        session_key="websocket:session-1",
    )

    resumed = tools.resume(session_key="websocket:session-1")
    assert resumed["found"] is True
    assert resumed["task"]["task_id"] == first["task_id"]

    follow_up = tools.start(
        original_question="What about cost?",
        normalized_question="What is the cost?",
        sub_questions=["What is the cost?"],
        session_key="websocket:session-1",
        parent_task_id=str(first["task_id"]),
    )
    assert follow_up["session_key"] == "websocket:session-1"
    assert follow_up["parent_task_id"] == first["task_id"]
    assert tools.resume(session_key="websocket:session-1")["task"]["task_id"] == follow_up["task_id"]


def test_research_tools_reject_cross_session_parent(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    first = tools.start(
        original_question="Question",
        normalized_question="Question",
        sub_questions=["Question"],
        session_key="websocket:session-1",
    )

    with pytest.raises(ValueError, match="different session"):
        tools.start(
            original_question="Follow-up",
            normalized_question="Follow-up",
            sub_questions=["Follow-up"],
            session_key="websocket:session-2",
            parent_task_id=str(first["task_id"]),
        )
def test_research_retrieve_stops_after_sufficient_candidate_evidence(tmp_path: Path) -> None:
    retrieval = _FakeRetrieval()
    tools = ResearchTools(
        ResearchConfig(
            data_dir=tmp_path,
            initial_paper_candidates=1,
            expanded_paper_candidates=2,
        ),
        retrieval=retrieval,  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
    )

    result = tools.research_retrieve(
        task_id=str(started["task_id"]),
        sub_question_id="SQ1",
        query="evidence recall",
    )

    assert result["stopped_after_rounds"] == 1
    assert result["coverage"]["sufficient"] is True
    assert result["attempts"][0]["scope"] == "candidate_papers"
    assert retrieval.evidence_calls[0]["paper_ids"] == ["P1"]
    assert result["sub_question_status"] == "sufficient"


def test_research_retrieve_expands_then_uses_one_global_fallback(tmp_path: Path) -> None:
    retrieval = _FakeRetrieval(global_only=True)
    tools = ResearchTools(
        ResearchConfig(
            data_dir=tmp_path,
            initial_paper_candidates=1,
            expanded_paper_candidates=2,
            max_retrieval_rounds=3,
        ),
        retrieval=retrieval,  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
    )
    task_id = str(started["task_id"])

    result = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
        alternative_query="retrieval quality",
    )
    status = tools.status(task_id)

    assert [item["scope"] for item in result["attempts"]] == [
        "candidate_papers",
        "expanded_candidates",
        "global_fallback",
    ]
    assert [call["paper_ids"] for call in retrieval.evidence_calls] == [
        ["P1"],
        ["P1", "P2"],
        None,
    ]
    assert result["coverage"]["sufficient"] is True
    assert status["retrieval_rounds"] == 3

    with pytest.raises(RuntimeError, match="already ran"):
        tools.research_retrieve(
            task_id=task_id,
            sub_question_id="SQ1",
            query="evidence recall",
        )


def test_verify_claims_builds_supported_claim_evidence_matrix(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
    )
    task_id = str(started["task_id"])
    retrieved = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    evidence_id = retrieved["evidence"][0]["evidence_id"]

    result = tools.verify_claims(
        task_id=task_id,
        claims=[
            ClaimDraft(
                claim_id="C1",
                sub_question_id="SQ1",
                text="The method improves evidence recall.",
                cited_evidence_ids=[evidence_id],
            )
        ],
        judgments=[
            EvidenceJudgment(
                claim_id="C1",
                evidence_id=evidence_id,
                verdict=CitationVerdict.SUPPORTED,
                rationale="The passage states the same result.",
            )
        ],
    )

    assert result["status"] == "completed"
    assert result["all_claims_supported"] is True
    assert result["citation_completeness"] == 1.0
    row = result["claim_evidence_matrix"][0]
    assert row["recommended_action"] == "keep"
    assert row["checks"][0]["locator"] == {
        "title": "Paper One",
        "page_start": 4,
        "page_end": 4,
        "chunk_id": "P1-C0000",
    }


def test_commit_verified_memory_uses_native_history_and_is_idempotent(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path / "research"),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
        session_key="websocket:session-1",
    )
    task_id = str(started["task_id"])
    retrieved = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    evidence_id = retrieved["evidence"][0]["evidence_id"]
    tools.verify_claims(
        task_id=task_id,
        claims=[
            ClaimDraft(
                claim_id="C1",
                sub_question_id="SQ1",
                text="The method improves evidence recall.",
                cited_evidence_ids=[evidence_id],
            )
        ],
        judgments=[
            EvidenceJudgment(
                claim_id="C1",
                evidence_id=evidence_id,
                verdict=CitationVerdict.SUPPORTED,
                rationale="The passage directly supports the claim.",
            )
        ],
    )
    workspace = tmp_path / "workspace"

    first = tools.commit_verified_memory(
        task_id=task_id,
        workspace_path=str(workspace),
        session_key="websocket:session-1",
    )
    second = tools.commit_verified_memory(
        task_id=task_id,
        workspace_path=str(workspace),
        session_key="websocket:session-1",
    )

    records = [
        json.loads(line)
        for line in (workspace / "memory" / "history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert first["committed"] is True
    assert second["already_committed"] is True
    assert len(records) == 1
    assert records[0]["session_key"] == "websocket:session-1"
    assert "[VERIFIED_RESEARCH]" in records[0]["content"]
    assert '"claim_id":"C1"' in records[0]["content"]
    assert '"chunk_id":"P1-C0000"' in records[0]["content"]
    assert "evidence_text" not in records[0]["content"]


def test_commit_verified_memory_rejects_unverified_task(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path / "research"),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="Question",
        normalized_question="Question",
        sub_questions=["Question"],
        session_key="websocket:session-1",
    )

    with pytest.raises(RuntimeError, match="terminal verified"):
        tools.commit_verified_memory(
            task_id=str(started["task_id"]),
            workspace_path=str(tmp_path / "workspace"),
            session_key="websocket:session-1",
        )


def test_missing_citation_can_trigger_one_bounded_gap_search(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path, max_verification_rounds=2),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
    )
    task_id = str(started["task_id"])
    tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    first = tools.verify_claims(
        task_id=task_id,
        claims=[
            ClaimDraft(
                claim_id="C1",
                sub_question_id="SQ1",
                text="The method improves evidence recall.",
            )
        ],
        judgments=[],
    )
    assert first["status"] == "verifying"
    assert first["claim_evidence_matrix"][0]["recommended_action"] == (
        "retrieve_or_remove"
    )

    gap = tools.retrieve_claim_gap(
        task_id=task_id,
        claim_id="C1",
        query="method evidence recall improvement",
    )
    evidence_id = gap["evidence"][0]["evidence_id"]
    with pytest.raises(RuntimeError, match="already ran"):
        tools.retrieve_claim_gap(
            task_id=task_id,
            claim_id="C1",
            query="repeat query",
        )

    second = tools.verify_claims(
        task_id=task_id,
        claims=[
            ClaimDraft(
                claim_id="C1",
                sub_question_id="SQ1",
                text="The method improves evidence recall.",
                cited_evidence_ids=[evidence_id],
            )
        ],
        judgments=[
            EvidenceJudgment(
                claim_id="C1",
                evidence_id=evidence_id,
                verdict=CitationVerdict.SUPPORTED,
                rationale="The retrieved passage directly states the claim.",
            )
        ],
    )
    assert second["status"] == "completed"
    assert second["verification_round"] == 2


def test_unknown_evidence_id_is_recorded_as_unlocatable(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=["What improves recall?"],
    )
    task_id = str(started["task_id"])
    tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )

    result = tools.verify_claims(
        task_id=task_id,
        claims=[
            ClaimDraft(
                claim_id="C1",
                sub_question_id="SQ1",
                text="An unsupported statement.",
                cited_evidence_ids=["E-does-not-exist"],
            )
        ],
        judgments=[],
    )

    check = result["claim_evidence_matrix"][0]["checks"][0]
    assert check["locatable"] is False
    assert check["verdict"] == "unsupported"
    assert result["all_claims_supported"] is False


def test_verification_requires_coverage_for_every_sub_question(tmp_path: Path) -> None:
    tools = ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=_FakeRetrieval(),  # type: ignore[arg-type]
    )
    started = tools.start(
        original_question="Compare accuracy and cost",
        normalized_question="Compare accuracy and cost",
        sub_questions=["What is the accuracy?", "What is the cost?"],
    )
    task_id = str(started["task_id"])
    first = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="accuracy",
    )
    tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ2",
        query="cost",
    )
    evidence_id = first["evidence"][0]["evidence_id"]

    result = tools.verify_claims(
        task_id=task_id,
        claims=[
            ClaimDraft(
                claim_id="C1",
                sub_question_id="SQ1",
                text="The method improves evidence recall.",
                cited_evidence_ids=[evidence_id],
            )
        ],
        judgments=[
            EvidenceJudgment(
                claim_id="C1",
                evidence_id=evidence_id,
                verdict=CitationVerdict.SUPPORTED,
                rationale="The passage directly supports the claim.",
            )
        ],
    )

    assert result["all_claims_supported"] is True
    assert result["all_evidence_needs_covered"] is False
    assert result["uncovered_sub_question_ids"] == ["SQ2"]
    assert result["status"] == "verifying"
