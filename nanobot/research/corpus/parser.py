"""PDF parsing with lightweight section detection and page provenance."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from nanobot.research.models import DocumentSection, PaperRecord

_KNOWN_HEADINGS = re.compile(
    r"^(?:\d+(?:\.\d+)*\s*)?"
    r"(?:abstract|introduction|background|related\s+work|method(?:ology)?|approach|"
    r"experiment(?:s|al\s+setup)?|results?|discussion|limitations?|conclusion(?:s)?|"
    r"references|appendix)\s*$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+\S.{0,80}$")
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_BOILERPLATE_LINE = re.compile(
    r"(?:authorized licensed use limited to|downloaded on .{0,100} from ieee xplore|"
    r"restrictions apply|©\s*\d{4}\s+ieee)",
    re.IGNORECASE,
)
_INFORMATIVE_CHARACTER = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")
_REPEATED_LINE_MAX_LENGTH = 300


@dataclass(frozen=True, slots=True)
class PageText:
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class ParsedPaper:
    paper: PaperRecord
    sections: list[DocumentSection]


def _clean_line(line: str) -> str:
    return " ".join(line.replace("\u00ad", "").split())


def _repetition_key(line: str) -> str:
    """Normalize changing page numbers/dates before comparing headers and footers."""
    return re.sub(r"\d+", "#", line.casefold())


def _remove_repeated_boilerplate(pages: list[PageText]) -> list[PageText]:
    """Remove short lines repeated across pages and known download notices."""
    cleaned_by_page = [
        [_clean_line(line) for line in page.text.splitlines() if _clean_line(line)]
        for page in pages
    ]
    repeated_keys: set[str] = set()
    if len(pages) > 1:
        page_frequency: Counter[str] = Counter()
        for lines in cleaned_by_page:
            page_frequency.update(
                {
                    _repetition_key(line)
                    for line in lines
                    if len(line) <= _REPEATED_LINE_MAX_LENGTH
                }
            )
        threshold = max(2, (len(pages) + 1) // 2)
        repeated_keys = {
            key for key, count in page_frequency.items() if count >= threshold
        }

    result: list[PageText] = []
    for page, lines in zip(pages, cleaned_by_page, strict=True):
        body_lines = [
            line
            for line in lines
            if not (
                len(line) <= _REPEATED_LINE_MAX_LENGTH
                and (
                    _repetition_key(line) in repeated_keys
                    or _BOILERPLATE_LINE.search(line)
                )
            )
        ]
        result.append(PageText(number=page.number, text="\n".join(body_lines)))
    return result


def _validate_body_text(pages: list[PageText]) -> None:
    full_text = "\n".join(page.text for page in pages).strip()
    if not full_text:
        raise ValueError("PDF contains no usable body text after boilerplate removal")
    if len(pages) >= 3:
        informative_count = len(_INFORMATIVE_CHARACTER.findall(full_text))
        minimum = min(1000, max(200, len(pages) * 40))
        if informative_count < minimum:
            raise ValueError(
                "PDF extracted body text is too sparse "
                f"({informative_count} informative characters; expected at least {minimum})"
            )


def _is_heading(line: str) -> bool:
    if not line or len(line) > 100 or line.endswith((".", ",", ";", ":")):
        return False
    return bool(_KNOWN_HEADINGS.fullmatch(line) or _NUMBERED_HEADING.fullmatch(line))


def _metadata_text(metadata: Any, key: str) -> str:
    value = getattr(metadata, key, None) if metadata is not None else None
    return value.strip() if isinstance(value, str) else ""


class PdfPaperParser:
    """Parse PDFs into a paper record plus section-level text blocks."""

    def parse(self, path: Path) -> ParsedPaper:
        path = path.expanduser().resolve()
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"expected a PDF file: {path}")
        reader = PdfReader(str(path))
        metadata = reader.metadata
        pages = [
            PageText(number=index + 1, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages)
        ]
        return self.parse_pages(
            path=path,
            pages=pages,
            title=_metadata_text(metadata, "title"),
            author=_metadata_text(metadata, "author"),
            creation_date=_metadata_text(metadata, "creation_date"),
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        )

    def parse_pages(
        self,
        *,
        path: Path,
        pages: list[PageText],
        title: str = "",
        author: str = "",
        creation_date: str = "",
        content_hash: str | None = None,
    ) -> ParsedPaper:
        if not pages:
            raise ValueError("PDF contains no pages")
        path = path.expanduser().resolve(strict=False)
        first_lines = [
            _clean_line(line)
            for line in pages[0].text.splitlines()
            if _clean_line(line)
        ]
        resolved_title = title or (first_lines[0] if first_lines else path.stem)
        paper_id = content_hash or hashlib.sha256(
            (str(path) + "\n" + "\n".join(page.text for page in pages)).encode("utf-8")
        ).hexdigest()[:16]
        pages = _remove_repeated_boilerplate(pages)
        _validate_body_text(pages)
        sections = self._sections(pages)
        abstract = next(
            (section.text for section in sections if "abstract" in section.title.lower()),
            "",
        )
        full_text = "\n\n".join(section.text for section in sections)
        year_match = _YEAR.search(creation_date) or _YEAR.search(path.name)
        authors = [item.strip() for item in re.split(r"[,;]", author) if item.strip()]
        paper = PaperRecord(
            paper_id=paper_id,
            source_path=str(path),
            title=resolved_title,
            authors=authors,
            year=int(year_match.group(1)) if year_match else None,
            abstract=abstract,
            search_text=(resolved_title + "\n" + abstract + "\n" + full_text[:6000]).strip(),
            page_count=len(pages),
        )
        return ParsedPaper(paper=paper, sections=sections)

    def _sections(self, pages: list[PageText]) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        title = "Document"
        page_start = pages[0].number
        page_end = page_start
        lines: list[str] = []

        def flush() -> None:
            nonlocal lines
            text = "\n".join(lines).strip()
            if text:
                sections.append(
                    DocumentSection(
                        title=title,
                        page_start=page_start,
                        page_end=page_end,
                        text=text,
                    )
                )
            lines = []

        for page in pages:
            for raw_line in page.text.splitlines():
                line = _clean_line(raw_line)
                if not line:
                    continue
                if _is_heading(line):
                    flush()
                    title = line
                    page_start = page.number
                    page_end = page.number
                    continue
                lines.append(line)
                page_end = page.number
        flush()
        if sections:
            return sections
        fallback = "\n".join(page.text for page in pages).strip()
        if not fallback:
            raise ValueError("PDF contains no extractable text")
        return [
            DocumentSection(
                title="Document",
                page_start=pages[0].number,
                page_end=pages[-1].number,
                text=fallback,
            )
        ]
