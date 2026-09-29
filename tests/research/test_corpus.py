from pathlib import Path

import pytest

from nanobot.research.corpus.chunker import SectionAwareChunker
from nanobot.research.corpus.parser import PageText, PdfPaperParser


def test_parser_detects_sections_and_provenance(tmp_path: Path) -> None:
    parsed = PdfPaperParser().parse_pages(
        path=tmp_path / "paper-2025.pdf",
        title="Evidence RAG",
        author="A. Researcher; B. Engineer",
        pages=[
            PageText(
                number=1,
                text="Evidence RAG\nAbstract\nWe study grounded answers.\n1 Introduction\nPrior systems hallucinate.",
            ),
            PageText(
                number=2,
                text="2 Experiments\nOur method improves citation recall.",
            ),
        ],
        content_hash="paper123",
    )

    assert parsed.paper.paper_id == "paper123"
    assert parsed.paper.year == 2025
    assert parsed.paper.authors == ["A. Researcher", "B. Engineer"]
    assert "grounded answers" in parsed.paper.abstract
    assert [section.title for section in parsed.sections] == [
        "Document",
        "Abstract",
        "1 Introduction",
        "2 Experiments",
    ]
    assert parsed.sections[-1].page_start == 2


def test_chunker_keeps_overlap_and_neighbor_links(tmp_path: Path) -> None:
    parsed = PdfPaperParser().parse_pages(
        path=tmp_path / "paper.pdf",
        title="Chunking",
        pages=[PageText(number=1, text="Abstract\n" + " ".join(f"w{i}" for i in range(12)))],
        content_hash="chunk-paper",
    )
    chunks = SectionAwareChunker(chunk_size_words=5, overlap_words=2).chunk(
        parsed.paper, parsed.sections
    )

    assert len(chunks) == 4
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[-1].next_chunk_id is None
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_parser_removes_repeated_headers_and_keeps_body(tmp_path: Path) -> None:
    header = "Proceedings of Example Conference 2025 - Page 1"
    pages = [
        PageText(
            number=index,
            text=(
                header.replace("1", str(index))
                + "\n"
                + " ".join(
                    f"page{index}_research_term_{word}" for word in range(80)
                )
            ),
        )
        for index in range(1, 4)
    ]

    parsed = PdfPaperParser().parse_pages(
        path=tmp_path / "paper.pdf",
        title="Useful Paper",
        pages=pages,
        content_hash="useful-paper",
    )

    extracted = "\n".join(section.text for section in parsed.sections)
    assert "Proceedings of Example Conference" not in extracted
    assert "page1_research_term_0" in extracted
    assert "page3_research_term_79" in extracted


def test_parser_rejects_boilerplate_only_pdf(tmp_path: Path) -> None:
    notice = (
        "Authorized licensed use limited to: Example University. "
        "Downloaded on February 12, 2026 from IEEE Xplore. Restrictions apply."
    )

    with pytest.raises(ValueError, match="no usable body text"):
        PdfPaperParser().parse_pages(
            path=tmp_path / "protected.pdf",
            title="Metadata Title",
            pages=[PageText(number=index, text=notice) for index in range(1, 8)],
            content_hash="protected-paper",
        )
