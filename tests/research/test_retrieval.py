from pathlib import Path

from nanobot.research.config import ResearchConfig
from nanobot.research.corpus.store import CorpusStore
from nanobot.research.models import PaperChunk, PaperRecord, SearchHit
from nanobot.research.retrieval.fusion import reciprocal_rank_fusion
from nanobot.research.retrieval.service import HybridRetrievalService
from nanobot.research.retrieval.sparse import BM25Index


class _FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        return [1.0 if "graph" in document.lower() else 0.1 for document in documents]


class _FakeEncoder:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(list(texts))
        return [[1.0] for _text in texts]


def _write_sparse_corpus(data_dir: Path) -> None:
    papers = [
        PaperRecord(
            paper_id="P1",
            source_path="p1.pdf",
            title="Graph Retrieval",
            year=2025,
            abstract="Graph retrieval for multi-hop questions.",
            search_text="Graph retrieval for multi-hop questions and global reasoning.",
            page_count=5,
        ),
        PaperRecord(
            paper_id="P2",
            source_path="p2.pdf",
            title="Lexical Retrieval",
            year=2023,
            abstract="A BM25 baseline.",
            search_text="BM25 lexical retrieval baseline.",
            page_count=4,
        ),
    ]
    chunks = [
        PaperChunk(
            chunk_id="P1-C0000",
            paper_id="P1",
            order=0,
            section="Experiments",
            page_start=3,
            page_end=3,
            text="Graph retrieval improves multi-hop recall.",
        ),
        PaperChunk(
            chunk_id="P2-C0000",
            paper_id="P2",
            order=0,
            section="Method",
            page_start=2,
            page_end=2,
            text="BM25 is a lexical retrieval baseline.",
        ),
    ]
    CorpusStore(data_dir).write(papers, chunks)
    paper_bm25 = BM25Index()
    paper_bm25.build({paper.paper_id: paper.search_text for paper in papers})
    paper_bm25.save(data_dir / "paper_bm25.json")
    chunk_bm25 = BM25Index()
    chunk_bm25.build({chunk.chunk_id: chunk.text for chunk in chunks})
    chunk_bm25.save(data_dir / "chunk_bm25.json")


def test_bm25_and_rrf_rank_expected_items() -> None:
    index = BM25Index()
    index.build({"A": "graph retrieval evidence", "B": "database transaction"})
    hits = index.search("graph evidence")
    assert hits[0].item_id == "A"

    fused = reciprocal_rank_fusion(
        {
            "dense": [SearchHit(item_id="A", score=0.8), SearchHit(item_id="B", score=0.7)],
            "sparse": [SearchHit(item_id="B", score=4.0), SearchHit(item_id="A", score=3.0)],
        }
    )
    assert {item.item_id for item in fused} == {"A", "B"}
    assert all(set(item.source_scores) == {"dense", "sparse"} for item in fused)


def test_sparse_service_filters_and_returns_provenance(tmp_path: Path) -> None:
    _write_sparse_corpus(tmp_path)
    config = ResearchConfig(data_dir=tmp_path, default_top_k=2)
    reranker = _FakeReranker()
    service = HybridRetrievalService(
        config,
        reranker=reranker,
        allow_sparse_only=True,
    )

    papers = service.search_papers("graph retrieval", year_from=2024)
    assert reranker.calls == 0
    evidence = service.retrieve_evidence(
        "multi-hop recall",
        year_from=2024,
        sections=["experiment"],
    )

    assert [paper.paper_id for paper in papers] == ["P1"]
    assert evidence[0].chunk_id == "P1-C0000"
    assert evidence[0].page_start == 3
    assert evidence[0].rerank_score == 1.0
    assert reranker.calls == 1


def test_full_corpus_chunk_search_applies_year_filter(tmp_path: Path) -> None:
    _write_sparse_corpus(tmp_path)
    service = HybridRetrievalService(
        ResearchConfig(data_dir=tmp_path, default_top_k=2),
        reranker=_FakeReranker(),
        allow_sparse_only=True,
    )

    evidence = service.retrieve_evidence("retrieval", year_to=2023)

    assert [item.paper_id for item in evidence] == ["P2"]


def test_paper_filter_is_applied_before_sparse_candidate_cutoff(tmp_path: Path) -> None:
    _write_sparse_corpus(tmp_path)
    config = ResearchConfig(
        data_dir=tmp_path,
        sparse_candidates=1,
        default_top_k=1,
    )
    service = HybridRetrievalService(
        config,
        reranker=_FakeReranker(),
        allow_sparse_only=True,
    )

    evidence = service.retrieve_evidence(
        "retrieval baseline",
        paper_ids=["P1"],
        top_k=1,
    )

    assert [item.paper_id for item in evidence] == ["P1"]


def test_explicit_source_resolution_uses_exact_filename_or_title(tmp_path: Path) -> None:
    _write_sparse_corpus(tmp_path)
    service = HybridRetrievalService(
        ResearchConfig(data_dir=tmp_path),
        reranker=_FakeReranker(),
        allow_sparse_only=True,
    )

    assert service.resolve_paper_scope(source_files=["p1.pdf"]) == ["P1"]
    assert service.resolve_paper_scope(paper_titles=["Graph Retrieval"]) == ["P1"]
    assert service.resolve_paper_scope(source_files=["graph.pdf"]) == []


def test_prepare_models_warms_encoder_and_reranker(tmp_path: Path) -> None:
    _write_sparse_corpus(tmp_path)
    encoder = _FakeEncoder()
    reranker = _FakeReranker()
    service = HybridRetrievalService(
        ResearchConfig(data_dir=tmp_path),
        encoder=encoder,
        reranker=reranker,
        allow_sparse_only=True,
    )

    service.prepare_models()

    assert encoder.inputs == [["retrieval warmup"]]
    assert reranker.calls == 1
