from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Timer

import pytest

from nanobot.research.config import ResearchConfig
from nanobot.research.mcp_server import ResearchTools
from nanobot.research.models import (
    CitationReference,
    EvidenceAssessment,
    EvidenceSearchResult,
    PaperSearchResult,
)


class _FakeRetrieval:
    def __init__(self) -> None:
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


class _BlockingRetrieval(_FakeRetrieval):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def retrieve_evidence(
        self, query: str, **kwargs: object
    ) -> list[EvidenceSearchResult]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test retrieval was not released")
        return super().retrieve_evidence(query, **kwargs)


class _FlakyRetrieval(_FakeRetrieval):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.calls = 0

    def retrieve_evidence(
        self, query: str, **kwargs: object
    ) -> list[EvidenceSearchResult]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary retrieval failure")
        return super().retrieve_evidence(query, **kwargs)


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


def test_retrieval_searches_full_chunk_corpus_once(tmp_path: Path) -> None:
    retrieval = _FakeRetrieval()
    tools = _tools(tmp_path, retrieval)
    task_id = _start(tools)

    result = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    status = tools.status(task_id)

    assert result["retrieval_scope"] == "full_corpus"
    assert result["retrieval_attempt"]["scope"] == "full_corpus"
    assert [call["paper_ids"] for call in retrieval.evidence_calls] == [None]
    assert retrieval.paper_calls == []
    assert status["retrieval_rounds"] == 1
    assert result["semantic_round"] == 1


def test_duplicate_retrieval_does_not_start_while_original_is_running(
    tmp_path: Path,
) -> None:
    retrieval = _BlockingRetrieval()
    tools = _tools(tmp_path, retrieval)
    task_id = _start(tools)

    with ThreadPoolExecutor(max_workers=2) as executor:
        original = executor.submit(
            tools.research_retrieve,
            task_id=task_id,
            sub_question_id="SQ1",
            query="evidence recall",
        )
        assert retrieval.started.wait(timeout=2)
        duplicate = tools.research_retrieve(
            task_id=task_id,
            sub_question_id="SQ1",
            query="evidence recall",
        )
        assert duplicate["execution_status"] == "running"
        release_timer = Timer(0.1, retrieval.release.set)
        release_timer.start()
        waited_status = tools.status(task_id, wait_seconds=2)
        release_timer.join(timeout=1)
        completed = original.result(timeout=2)

    recovered = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    assert completed["execution_status"] == "completed"
    assert (
        waited_status["plan"]["sub_questions"][0]["retrieval_execution"]["status"]
        == "completed"
    )
    assert recovered["recovered"] is True
    assert len(retrieval.evidence_calls) == 1


def test_technical_failure_can_retry_once_without_consuming_semantic_round(
    tmp_path: Path,
) -> None:
    retrieval = _FlakyRetrieval(failures=1)
    tools = _tools(tmp_path, retrieval)
    task_id = _start(tools)

    with pytest.raises(RuntimeError, match="temporary retrieval failure"):
        tools.research_retrieve(
            task_id=task_id,
            sub_question_id="SQ1",
            query="evidence recall",
        )
    recovered = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    status = tools.status(task_id)

    execution = status["plan"]["sub_questions"][0]["retrieval_execution"]
    assert recovered["semantic_round"] == 1
    assert status["retrieval_rounds"] == 1
    assert execution["technical_attempt"] == 2
    assert execution["status"] == "completed"


def test_technical_failure_stops_after_one_retry(tmp_path: Path) -> None:
    retrieval = _FlakyRetrieval(failures=3)
    tools = _tools(tmp_path, retrieval)
    task_id = _start(tools)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="temporary retrieval failure"):
            tools.research_retrieve(
                task_id=task_id,
                sub_question_id="SQ1",
                query="evidence recall",
            )
    with pytest.raises(
        RuntimeError, match="technical retrieval retry already exhausted"
    ):
        tools.research_retrieve(
            task_id=task_id,
            sub_question_id="SQ1",
            query="evidence recall",
        )
    assert retrieval.calls == 2


def test_reflect_sufficient_then_finalize_citations(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    task_id = _start(tools)
    retrieved = tools.research_retrieve(
        task_id=task_id,
        sub_question_id="SQ1",
        query="evidence recall",
    )
    chunk_id = retrieved["evidence"][0]["chunk_id"]
    assert "evidence_id" not in retrieved["evidence"][0]

    reflected = tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                supporting_chunk_ids=[chunk_id],
                reason="The passage directly answers the question.",
            )
        ],
    )
    finalized = tools.finalize(
        task_id=task_id,
        answered_sub_question_ids=["SQ1"],
        citations=[CitationReference(sub_question_id="SQ1", chunk_id=chunk_id)],
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
    chunk_id = first["evidence"][0]["chunk_id"]
    tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                supporting_chunk_ids=[chunk_id],
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
        citations=[CitationReference(sub_question_id="SQ1", chunk_id=chunk_id)],
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
    chunk_id = retrieved["evidence"][0]["chunk_id"]
    tools.reflect(
        task_id=task_id,
        assessments=[
            EvidenceAssessment(
                sub_question_id="SQ1",
                sufficient=True,
                supporting_chunk_ids=[chunk_id],
                reason="The evidence is sufficient.",
            )
        ],
    )

    with pytest.raises(ValueError, match="not approved by Reflect"):
        tools.finalize(
            task_id=task_id,
            answered_sub_question_ids=["SQ1"],
            citations=[
                CitationReference(
                    sub_question_id="SQ1",
                    chunk_id="P1-C-does-not-exist",
                )
            ],
        )
    with pytest.raises(ValueError, match="has no citation"):
        tools.finalize(
            task_id=task_id,
            answered_sub_question_ids=["SQ1"],
            citations=[],
        )
