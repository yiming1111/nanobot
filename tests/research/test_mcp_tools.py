from pathlib import Path

import pytest

from nanobot.research.config import ResearchConfig
from nanobot.research.mcp_server import ResearchTools
from nanobot.research.models import (
    EvidenceAssessment,
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
            PaperSearchResult(paper_id="P1", title="Paper One", fused_score=0.1),
            PaperSearchResult(paper_id="P2", title="Paper Two", fused_score=0.08),
        ]

    def retrieve_evidence(self, query: str, **kwargs: object) -> list[EvidenceSearchResult]:
        self.evidence_calls.append({"query": query, **kwargs})
        if self.global_only and kwargs.get("paper_ids") is not None:
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


def _tools(tmp_path: Path, retrieval: _FakeRetrieval | None = None) -> ResearchTools:
    return ResearchTools(
        ResearchConfig(data_dir=tmp_path),
        retrieval=retrieval or _FakeRetrieval(),  # type: ignore[arg-type]
    )


def _start(tools: ResearchTools, sub_questions: list[str] | None = None) -> str:
    started = tools.start(
        original_question="What improves recall?",
        normalized_question="What improves recall?",
        sub_questions=sub_questions or ["What improves recall?"],
    )
    return str(started["task_id"])


def test_retrieval_requires_reflect_before_generation(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    task_id = _start(tools)

    result = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )

    assert result["candidate_passages_found"] is True
    assert result["semantic_sufficiency"] == "pending_reflect"
    assert result["task_status"] == "reflecting"
    assert result["semantic_round"] == 1
    with pytest.raises(RuntimeError, match="research_reflect"):
        tools.research_retrieve(
            task_id=task_id,
            sub_question_id="SQ1",
            query="repeat before reflect",
        )


def test_empty_result_scope_fallback_is_one_semantic_round(tmp_path: Path) -> None:
    retrieval = _FakeRetrieval(global_only=True)
    tools = ResearchTools(
        ResearchConfig(
            data_dir=tmp_path,
            initial_paper_candidates=1,
            expanded_paper_candidates=2,
        ),
        retrieval=retrieval,  # type: ignore[arg-type]
    )
    task_id = _start(tools)

    result = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    status = tools.status(task_id)

    assert [item["scope"] for item in result["technical_attempts"]] == [
        "candidate_papers",
        "expanded_candidates",
        "global_fallback",
    ]
    assert [call["paper_ids"] for call in retrieval.evidence_calls] == [
        ["P1"],
        ["P1", "P2"],
        None,
    ]
    assert status["retrieval_rounds"] == 1
    assert result["semantic_round"] == 1


def test_reflect_sufficient_then_finalize_citations(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    task_id = _start(tools)
    retrieved = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    evidence_id = retrieved["evidence"][0]["evidence_id"]

    reflected = tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                evidence_ids=[evidence_id],
                reason="The passage directly answers the question.",
            )
        ],
    )
    finalized = tools.finalize(
        task_id=task_id,
        answered_sub_question_ids=["SQ1"],
        cited_evidence_ids=[evidence_id],
    )

    assert reflected["can_generate"] is True
    assert reflected["retry_sub_questions"] == []
    assert finalized["status"] == "completed"
    assert finalized["citations_valid"] is True
    assert finalized["citation_checks"][0]["chunk_id"] == "P1-C0000"


def test_reflect_allows_exactly_one_focused_retry(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    task_id = _start(tools)
    tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="broad query",
    )
    first = tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=False,
                reason="The passage is related but does not answer the question.",
                next_query="focused missing evidence",
            )
        ],
    )
    assert first["retry_sub_questions"][0]["next_query"] == (
        "focused missing evidence"
    )

    tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="focused missing evidence",
    )
    second = tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=False,
                reason="The second search is still insufficient.",
            )
        ],
    )
    assert second["must_abstain"] is True
    assert second["status"] == "refused"
    with pytest.raises(RuntimeError, match="maximum semantic retrieval rounds"):
        tools.research_retrieve(
            task_id=task_id,
            sub_question_id="SQ1",
            query="forbidden third search",
        )


def test_partial_sub_question_coverage_completes_with_gaps(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    task_id = _start(tools, ["Accuracy?", "Cost?"])
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
    tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                evidence_ids=[evidence_id],
                reason="Accuracy is directly supported.",
            ),
            EvidenceAssessment(
                sub_question_id="SQ2",
                sufficient=False,
                reason="Cost is not reported.",
                next_query="reported computation cost",
            ),
        ],
    )
    tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ2",
        query="reported computation cost",
    )
    reflected = tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ2",
                sufficient=False,
                reason="The retry still does not report cost.",
            )
        ],
    )
    finalized = tools.finalize(
        task_id=task_id,
        answered_sub_question_ids=["SQ1"],
        cited_evidence_ids=[evidence_id],
    )

    assert reflected["can_generate"] is True
    assert reflected["unresolved_sub_question_ids"] == ["SQ2"]
    assert finalized["status"] == "completed_with_gaps"
    assert finalized["unresolved_sub_question_ids"] == ["SQ2"]


def test_finalize_rejects_unknown_or_missing_citations(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    task_id = _start(tools)
    retrieved = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    evidence_id = retrieved["evidence"][0]["evidence_id"]
    tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                evidence_ids=[evidence_id],
                reason="The evidence is sufficient.",
            )
        ],
    )

    with pytest.raises(ValueError, match="unknown evidence ID"):
        tools.finalize(
            task_id=task_id,
            answered_sub_question_ids=["SQ1"],
            cited_evidence_ids=["E-does-not-exist"],
        )
    with pytest.raises(ValueError, match="has no citation"):
        tools.finalize(
            task_id=task_id,
            answered_sub_question_ids=["SQ1"],
            cited_evidence_ids=[],
        )
