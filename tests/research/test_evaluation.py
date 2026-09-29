from nanobot.research.evaluation import RetrievalEvalCase, evaluate_retrieval
from nanobot.research.models import EvidenceSearchResult, PaperSearchResult


class _FakeRetrieval:
    def search_papers(self, query: str, **kwargs: object) -> list[PaperSearchResult]:
        return [
            PaperSearchResult(paper_id="P2", title="Two", fused_score=0.2),
            PaperSearchResult(paper_id="P1", title="One", fused_score=0.1),
        ]

    def retrieve_evidence(
        self, query: str, **kwargs: object
    ) -> list[EvidenceSearchResult]:
        return [
            EvidenceSearchResult(
                chunk_id="P1-C2",
                paper_id="P1",
                title="One",
                section="Results",
                page_start=2,
                page_end=2,
                text="Relevant text",
                fused_score=0.1,
            ),
            EvidenceSearchResult(
                chunk_id="P2-C1",
                paper_id="P2",
                title="Two",
                section="Methods",
                page_start=1,
                page_end=1,
                text="Other text",
                fused_score=0.09,
            ),
        ]


def test_evaluate_retrieval_computes_recall_and_mrr() -> None:
    report = evaluate_retrieval(
        _FakeRetrieval(),
        [
            RetrievalEvalCase(
                case_id="case-1",
                query="relevant query",
                relevant_paper_ids=["P1"],
                relevant_chunk_ids=["P1-C2"],
            )
        ],
        paper_top_k=2,
        evidence_top_k=2,
    )

    assert report.metrics["paper_recall_at_k"] == 1.0
    assert report.metrics["paper_mrr"] == 0.5
    assert report.metrics["evidence_recall_at_k"] == 1.0
    assert report.metrics["evidence_paper_recall_at_k"] == 1.0
