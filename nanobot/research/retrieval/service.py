"""High-level hybrid retrieval over the indexed paper corpus."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from threading import RLock
from time import perf_counter
from typing import Any, Protocol

from nanobot.research.config import ResearchConfig
from nanobot.research.corpus.store import CorpusStore
from nanobot.research.models import EvidenceSearchResult, PaperSearchResult, SearchHit
from nanobot.research.retrieval.dense import BGEM3Encoder, FaissDenseIndex
from nanobot.research.retrieval.fusion import FusedHit, reciprocal_rank_fusion
from nanobot.research.retrieval.reranker import BGEReranker
from nanobot.research.retrieval.sparse import BM25Index

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Any: ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class HybridRetrievalService:
    def __init__(
        self,
        config: ResearchConfig,
        *,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        allow_sparse_only: bool = False,
    ) -> None:
        started_at = perf_counter()
        self.config = config
        self.store = CorpusStore(config.data_dir)
        self.papers = self.store.load_papers()
        self.chunks = self.store.load_chunks()
        self.paper_by_id = {paper.paper_id: paper for paper in self.papers}
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.paper_sparse = BM25Index.load(config.data_dir / "paper_bm25.json")
        self.chunk_sparse = BM25Index.load(config.data_dir / "chunk_bm25.json")
        self._paper_dense: FaissDenseIndex | None = None
        self._chunk_dense: FaissDenseIndex | None = None
        self._encoder = encoder
        self._reranker = reranker
        self._allow_sparse_only = allow_sparse_only
        self._model_lock = RLock()
        logger.info(
            "retrieval service initialized papers=%d chunks=%d elapsed_ms=%d",
            len(self.papers),
            len(self.chunks),
            round((perf_counter() - started_at) * 1000),
        )

    def prepare_indexes(self) -> None:
        """Load the small persisted indexes without loading neural models."""
        self._load_dense()

    def prepare_models(self) -> None:
        """Load neural retrieval models and warm their first inference paths."""
        started_at = perf_counter()
        logger.info("retrieval model preparation started")
        encoder = self._get_encoder()
        encoder.encode(["retrieval warmup"])
        reranker = self._get_reranker()
        reranker.score("retrieval warmup", ["retrieval warmup"])
        logger.info(
            "retrieval model preparation completed elapsed_ms=%d",
            round((perf_counter() - started_at) * 1000),
        )

    def _load_dense(self) -> tuple[FaissDenseIndex | None, FaissDenseIndex | None]:
        if self._paper_dense is not None and self._chunk_dense is not None:
            return self._paper_dense, self._chunk_dense
        started_at = perf_counter()
        logger.info("FAISS index loading started")
        paper_index = self.config.data_dir / "paper.faiss"
        chunk_index = self.config.data_dir / "chunk.faiss"
        if not paper_index.exists() or not chunk_index.exists():
            if self._allow_sparse_only:
                return None, None
            raise FileNotFoundError(
                "dense research indexes are missing; run `nanobot-research index <corpus-dir>`"
            )
        self._paper_dense = FaissDenseIndex.load(
            paper_index, self.config.data_dir / "paper_ids.json"
        )
        self._chunk_dense = FaissDenseIndex.load(
            chunk_index, self.config.data_dir / "chunk_ids.json"
        )
        logger.info(
            "FAISS index loading completed elapsed_ms=%d",
            round((perf_counter() - started_at) * 1000),
        )
        return self._paper_dense, self._chunk_dense

    def _get_encoder(self) -> Encoder:
        with self._model_lock:
            if self._encoder is None:
                started_at = perf_counter()
                logger.info("BGE-M3 encoder initialization started")
                self._encoder = BGEM3Encoder(
                    self.config.dense_model,
                    device=self.config.device,
                    use_fp16=self.config.use_fp16,
                    batch_size=self.config.embedding_batch_size,
                    max_length=self.config.embedding_max_length,
                )
                logger.info(
                    "BGE-M3 encoder initialization completed elapsed_ms=%d",
                    round((perf_counter() - started_at) * 1000),
                )
        return self._encoder

    def _get_reranker(self) -> Reranker:
        with self._model_lock:
            if self._reranker is None:
                started_at = perf_counter()
                logger.info("BGE reranker initialization started")
                self._reranker = BGEReranker(
                    self.config.reranker_model,
                    device=self.config.device,
                    use_fp16=self.config.use_fp16,
                )
                logger.info(
                    "BGE reranker initialization completed elapsed_ms=%d",
                    round((perf_counter() - started_at) * 1000),
                )
        return self._reranker

    def _rankings(
        self,
        query: str,
        *,
        sparse: BM25Index,
        dense: FaissDenseIndex | None,
        allowed_ids: set[str] | None = None,
    ) -> dict[str, list[SearchHit]]:
        rankings = {
            "sparse": sparse.search(
                query,
                top_k=self.config.sparse_candidates,
                allowed_ids=allowed_ids,
            ),
        }
        if dense is not None:
            with self._model_lock:
                vector = self._get_encoder().encode([query])[0]
                rankings["dense"] = dense.search(
                    vector,
                    top_k=self.config.dense_candidates,
                    allowed_ids=allowed_ids,
                )
        return rankings

    def _rerank(
        self,
        query: str,
        fused: list[FusedHit],
        texts: dict[str, str],
        *,
        top_k: int,
    ) -> list[tuple[FusedHit, float | None]]:
        candidates = [item for item in fused if item.item_id in texts]
        candidates = candidates[: self.config.rerank_candidates]
        if not candidates:
            return []
        if self._allow_sparse_only and self._reranker is None:
            return [(item, None) for item in candidates[:top_k]]
        with self._model_lock:
            scores = self._get_reranker().score(
                query, [texts[item.item_id] for item in candidates]
            )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], -pair[0].fused_score, pair[0].item_id),
        )
        return [(item, score) for item, score in ranked[:top_k]]

    def search_papers(
        self,
        query: str,
        *,
        top_k: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[PaperSearchResult]:
        paper_dense, _chunk_dense = self._load_dense()
        rankings = self._rankings(query, sparse=self.paper_sparse, dense=paper_dense)
        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k)
        allowed = {
            paper.paper_id
            for paper in self.papers
            if (year_from is None or paper.year is not None and paper.year >= year_from)
            and (year_to is None or paper.year is not None and paper.year <= year_to)
        }
        fused = [item for item in fused if item.item_id in allowed]
        limit = top_k or self.config.default_top_k
        # Paper search is a recall-oriented routing stage. Cross-encoding long
        # document summaries on CPU is expensive and can discard papers before
        # chunk-level evidence retrieval gets a chance to inspect them.
        ranked = [(item, None) for item in fused[:limit]]
        results: list[PaperSearchResult] = []
        for item, rerank_score in ranked:
            paper = self.paper_by_id[item.item_id]
            results.append(
                PaperSearchResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    abstract=paper.abstract,
                    dense_score=item.source_scores.get("dense"),
                    sparse_score=item.source_scores.get("sparse"),
                    fused_score=item.fused_score,
                    rerank_score=rerank_score,
                )
            )
        return results

    def retrieve_evidence(
        self,
        query: str,
        *,
        top_k: int | None = None,
        paper_ids: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        sections: list[str] | None = None,
    ) -> list[EvidenceSearchResult]:
        _paper_dense, chunk_dense = self._load_dense()
        paper_filter = set(paper_ids or [])
        normalized_sections = [value.casefold() for value in sections or []]
        allowed_chunk_ids = {
            chunk.chunk_id
            for chunk in self.chunks
            if (not paper_filter or chunk.paper_id in paper_filter)
            and (
                year_from is None
                or self.paper_by_id[chunk.paper_id].year is not None
                and self.paper_by_id[chunk.paper_id].year >= year_from
            )
            and (
                year_to is None
                or self.paper_by_id[chunk.paper_id].year is not None
                and self.paper_by_id[chunk.paper_id].year <= year_to
            )
            and (
                not normalized_sections
                or any(value in chunk.section.casefold() for value in normalized_sections)
            )
        }
        rankings = self._rankings(
            query,
            sparse=self.chunk_sparse,
            dense=chunk_dense,
            allowed_ids=(
                allowed_chunk_ids
                if paper_filter
                or year_from is not None
                or year_to is not None
                or normalized_sections
                else None
            ),
        )
        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k)
        limit = top_k or self.config.default_top_k
        ranked = self._rerank(
            query,
            fused,
            {chunk.chunk_id: chunk.text for chunk in self.chunks},
            top_k=limit,
        )
        results: list[EvidenceSearchResult] = []
        for item, rerank_score in ranked:
            chunk = self.chunk_by_id[item.item_id]
            paper = self.paper_by_id[chunk.paper_id]
            results.append(
                EvidenceSearchResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    title=paper.title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    dense_score=item.source_scores.get("dense"),
                    sparse_score=item.source_scores.get("sparse"),
                    fused_score=item.fused_score,
                    rerank_score=rerank_score,
                )
            )
        return results

    def get_neighbors(self, chunk_id: str, *, window: int = 1) -> list[EvidenceSearchResult]:
        chunks = self.store.neighbors(chunk_id, window=window)
        return [
            EvidenceSearchResult(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                title=self.paper_by_id[chunk.paper_id].title,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                fused_score=0.0,
            )
            for chunk in chunks
        ]

    def get_chunks(self, chunk_ids: list[str]) -> list[EvidenceSearchResult]:
        """Load exact corpus chunks without running semantic retrieval."""
        missing = [
            chunk_id for chunk_id in chunk_ids if chunk_id not in self.chunk_by_id
        ]
        if missing:
            raise KeyError(f"unknown chunks: {missing}")
        return [
            EvidenceSearchResult(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                title=self.paper_by_id[chunk.paper_id].title,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                fused_score=0.0,
            )
            for chunk in (self.chunk_by_id[chunk_id] for chunk_id in chunk_ids)
        ]
