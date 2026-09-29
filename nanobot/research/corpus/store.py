"""JSONL corpus persistence and provenance lookup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from nanobot.research.models import PaperChunk, PaperRecord

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _write_jsonl(path: Path, values: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(value.model_dump_json())
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_jsonl(path: Path, model: type[_ModelT]) -> list[_ModelT]:
    if not path.exists():
        raise FileNotFoundError(f"research corpus is not indexed: {path}")
    values: list[_ModelT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
    return values


class CorpusStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.papers_path = self.data_dir / "papers.jsonl"
        self.chunks_path = self.data_dir / "chunks.jsonl"

    def write(self, papers: list[PaperRecord], chunks: list[PaperChunk]) -> None:
        _write_jsonl(self.papers_path, papers)
        _write_jsonl(self.chunks_path, chunks)

    def load_papers(self) -> list[PaperRecord]:
        return _read_jsonl(self.papers_path, PaperRecord)

    def load_chunks(self) -> list[PaperChunk]:
        return _read_jsonl(self.chunks_path, PaperChunk)

    def paper_map(self) -> dict[str, PaperRecord]:
        return {paper.paper_id: paper for paper in self.load_papers()}

    def chunk_map(self) -> dict[str, PaperChunk]:
        return {chunk.chunk_id: chunk for chunk in self.load_chunks()}

    def neighbors(self, chunk_id: str, window: int = 1) -> list[PaperChunk]:
        if window < 0 or window > 10:
            raise ValueError("window must be between 0 and 10")
        chunks = self.load_chunks()
        target = next((chunk for chunk in chunks if chunk.chunk_id == chunk_id), None)
        if target is None:
            raise KeyError(f"unknown chunk: {chunk_id}")
        same_paper = sorted(
            (chunk for chunk in chunks if chunk.paper_id == target.paper_id),
            key=lambda chunk: chunk.order,
        )
        lower = max(0, target.order - window)
        upper = min(len(same_paper), target.order + window + 1)
        return same_paper[lower:upper]
