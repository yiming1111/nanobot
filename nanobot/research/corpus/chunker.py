"""Section-aware sliding-window chunking."""

from __future__ import annotations

import re

from nanobot.research.models import DocumentSection, PaperChunk, PaperRecord

_WORD = re.compile(r"\S+")


class SectionAwareChunker:
    def __init__(self, *, chunk_size_words: int = 350, overlap_words: int = 60) -> None:
        if chunk_size_words < 1:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0 or overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be non-negative and smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk(self, paper: PaperRecord, sections: list[DocumentSection]) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        step = self.chunk_size_words - self.overlap_words
        for section in sections:
            words = _WORD.findall(section.text)
            for start in range(0, len(words), step):
                window = words[start : start + self.chunk_size_words]
                if not window:
                    continue
                order = len(chunks)
                chunks.append(
                    PaperChunk(
                        chunk_id=f"{paper.paper_id}-C{order:04d}",
                        paper_id=paper.paper_id,
                        order=order,
                        section=section.title,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        text=" ".join(window),
                    )
                )
                if start + self.chunk_size_words >= len(words):
                    break
        for index, chunk in enumerate(chunks):
            chunk.previous_chunk_id = chunks[index - 1].chunk_id if index else None
            chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
        return chunks
