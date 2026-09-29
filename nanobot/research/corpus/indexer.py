"""Build persistent sparse and dense indexes from a directory of PDFs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from nanobot.research.config import ResearchConfig
from nanobot.research.corpus.chunker import SectionAwareChunker
from nanobot.research.corpus.parser import PdfPaperParser
from nanobot.research.corpus.store import CorpusStore
from nanobot.research.models import PaperChunk, PaperRecord
from nanobot.research.retrieval.dense import BGEM3Encoder, FaissDenseIndex
from nanobot.research.retrieval.sparse import BM25Index


class IndexBuildReport(BaseModel):
    papers: int
    chunks: int
    dense_enabled: bool
    failed_files: dict[str, str] = Field(default_factory=dict)


class CorpusIndexer:
    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.parser = PdfPaperParser()
        self.chunker = SectionAwareChunker(
            chunk_size_words=config.chunk_size_words,
            overlap_words=config.chunk_overlap_words,
        )
        self.store = CorpusStore(config.data_dir)

    def index_directory(self, corpus_dir: Path, *, build_dense: bool = True) -> IndexBuildReport:
        corpus_dir = corpus_dir.expanduser().resolve()
        if not corpus_dir.is_dir():
            raise NotADirectoryError(f"paper corpus directory not found: {corpus_dir}")
        self.config.ensure_directories()
        papers: list[PaperRecord] = []
        chunks: list[PaperChunk] = []
        failed: dict[str, str] = {}
        for path in sorted(corpus_dir.rglob("*.pdf")):
            try:
                parsed = self.parser.parse(path)
                paper_chunks = self.chunker.chunk(parsed.paper, parsed.sections)
                if not paper_chunks:
                    raise ValueError("no text chunks were produced")
                papers.append(parsed.paper)
                chunks.extend(paper_chunks)
            except Exception as exc:
                failed[str(path)] = f"{type(exc).__name__}: {exc}"
        if not papers:
            raise ValueError("no readable PDF papers were found")
        self.store.write(papers, chunks)
        self._build_sparse(papers, chunks)
        if build_dense:
            self._build_dense(papers, chunks)
        self._write_manifest(papers, chunks, build_dense, failed)
        return IndexBuildReport(
            papers=len(papers),
            chunks=len(chunks),
            dense_enabled=build_dense,
            failed_files=failed,
        )

    def _build_sparse(self, papers: list[PaperRecord], chunks: list[PaperChunk]) -> None:
        paper_index = BM25Index()
        paper_index.build({paper.paper_id: paper.search_text for paper in papers})
        paper_index.save(self.config.data_dir / "paper_bm25.json")
        chunk_index = BM25Index()
        chunk_index.build({chunk.chunk_id: chunk.text for chunk in chunks})
        chunk_index.save(self.config.data_dir / "chunk_bm25.json")

    def _build_dense(self, papers: list[PaperRecord], chunks: list[PaperChunk]) -> None:
        encoder = BGEM3Encoder(
            self.config.dense_model,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            batch_size=self.config.embedding_batch_size,
            max_length=self.config.embedding_max_length,
        )
        paper_vectors = encoder.encode([paper.search_text for paper in papers])
        FaissDenseIndex.build(
            [paper.paper_id for paper in papers], paper_vectors
        ).save(self.config.data_dir / "paper.faiss", self.config.data_dir / "paper_ids.json")
        chunk_vectors = encoder.encode([chunk.text for chunk in chunks])
        FaissDenseIndex.build(
            [chunk.chunk_id for chunk in chunks], chunk_vectors
        ).save(self.config.data_dir / "chunk.faiss", self.config.data_dir / "chunk_ids.json")

    def _write_manifest(
        self,
        papers: list[PaperRecord],
        chunks: list[PaperChunk],
        dense_enabled: bool,
        failed: dict[str, str],
    ) -> None:
        payload = {
            "papers": len(papers),
            "chunks": len(chunks),
            "dense_enabled": dense_enabled,
            "dense_model": self.config.dense_model if dense_enabled else None,
            "reranker_model": self.config.reranker_model,
            "failed_files": failed,
        }
        path = self.config.data_dir / "index_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
