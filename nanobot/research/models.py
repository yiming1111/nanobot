"""Typed domain models for paper retrieval and research state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


ConstraintScalar = str | int | float | bool
ConstraintValue = ConstraintScalar | list[ConstraintScalar]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SubQuestionStatus(StrEnum):
    PENDING = "pending"
    SEARCHING = "searching"
    EVIDENCE_FOUND = "evidence_found"
    INSUFFICIENT = "insufficient"
    SUFFICIENT = "sufficient"
    CONFLICTING = "conflicting"


class ResearchTaskStatus(StrEnum):
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    REFLECTING = "reflecting"
    READY_TO_SYNTHESIZE = "ready_to_synthesize"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"
    REFUSED = "refused"


class EvidenceRelation(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"
    UNCLASSIFIED = "unclassified"


class RetrievalScope(StrEnum):
    FULL_CORPUS = "full_corpus"
    # Retained so persisted states from the earlier hierarchical flow still load.
    CANDIDATE_PAPERS = "candidate_papers"
    EXPANDED_CANDIDATES = "expanded_candidates"
    GLOBAL_FALLBACK = "global_fallback"


class ClaimStatus(StrEnum):
    PENDING = "pending"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTING = "conflicting"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


class CitationVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    MISSING_CITATION = "missing_citation"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class DocumentSection(ResearchModel):
    title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_page_range(self) -> "DocumentSection":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class PaperRecord(ResearchModel):
    paper_id: str
    source_path: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    search_text: str = ""
    page_count: int = Field(default=0, ge=0)


class PaperChunk(ResearchModel):
    chunk_id: str
    paper_id: str
    order: int = Field(ge=0)
    section: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


class RetrievalAttempt(ResearchModel):
    round_index: int = Field(ge=1)
    query: str = Field(min_length=1)
    scope: RetrievalScope
    paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    elapsed_ms: int = Field(default=0, ge=0)
    sufficient: bool = False
    coverage_reason: str = ""


class EvidenceAssessment(ResearchModel):
    """Reflect's semantic sufficiency decision for one evidence need."""

    sub_question_id: str = Field(min_length=1)
    sufficient: bool
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    next_query: str | None = None

    @model_validator(mode="after")
    def valid_assessment(self) -> "EvidenceAssessment":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique within an assessment")
        if self.sufficient and not self.evidence_ids:
            raise ValueError("a sufficient assessment must cite evidence")
        if self.next_query is not None and not self.next_query.strip():
            raise ValueError("next_query must contain non-whitespace characters")
        return self


class EvidenceReflection(EvidenceAssessment):
    round_index: int = Field(ge=1)


class SubQuestion(ResearchModel):
    sub_question_id: str
    question: str = Field(min_length=1)
    evidence_type: str = "general"
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    queries: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    next_query: str | None = None
    candidate_paper_ids: list[str] = Field(default_factory=list)
    retrieval_attempts: list[RetrievalAttempt] = Field(default_factory=list)
    reflections: list[EvidenceReflection] = Field(default_factory=list)


class ResearchPlan(ResearchModel):
    original_question: str = Field(min_length=1)
    normalized_question: str = Field(min_length=1)
    task_type: str = "paper_research"
    constraints: dict[str, ConstraintValue] = Field(default_factory=dict)
    requires_decomposition: bool = False
    sub_questions: list[SubQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sub_question_ids(self) -> "ResearchPlan":
        ids = [item.sub_question_id for item in self.sub_questions]
        if len(ids) != len(set(ids)):
            raise ValueError("sub_question_id values must be unique")
        return self


class EvidenceItem(ResearchModel):
    evidence_id: str
    sub_question_id: str
    paper_id: str
    chunk_id: str
    title: str
    section: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    relation: EvidenceRelation = EvidenceRelation.UNCLASSIFIED
    target_claim_id: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class ClaimState(ResearchModel):
    claim_id: str
    sub_question_id: str | None = None
    text: str = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.PENDING
    revision: str | None = None


class ClaimDraft(ResearchModel):
    """One independently checkable statement in an answer draft."""

    claim_id: str = Field(min_length=1)
    sub_question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_citations(self) -> "ClaimDraft":
        if len(self.cited_evidence_ids) != len(set(self.cited_evidence_ids)):
            raise ValueError("cited_evidence_ids must be unique within a claim")
        return self


class EvidenceJudgment(ResearchModel):
    """A semantic judgment made in a separate LLM verification pass."""

    claim_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    verdict: CitationVerdict
    rationale: str = Field(min_length=1)
    revision: str | None = None

    @model_validator(mode="after")
    def missing_citation_is_server_generated(self) -> "EvidenceJudgment":
        if self.verdict == CitationVerdict.MISSING_CITATION:
            raise ValueError(
                "missing_citation is generated when a claim has no cited evidence"
            )
        return self


class CitationCheck(ResearchModel):
    claim_id: str
    evidence_id: str | None = None
    locatable: bool = False
    verdict: CitationVerdict = CitationVerdict.NOT_ENOUGH_INFORMATION
    rationale: str = ""
    revision: str | None = None


class CitationLocatorCheck(ResearchModel):
    """Deterministic validation for a citation used in the final answer."""

    evidence_id: str
    sub_question_id: str | None = None
    locatable: bool = False
    reason: str = ""
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str | None = None


class ResearchState(ResearchModel):
    task_id: str = Field(default_factory=lambda: uuid4().hex)
    session_key: str | None = None
    status: ResearchTaskStatus = ResearchTaskStatus.PLANNING
    plan: ResearchPlan
    evidence: dict[str, EvidenceItem] = Field(default_factory=dict)
    claims: dict[str, ClaimState] = Field(default_factory=dict)
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    citation_locator_checks: list[CitationLocatorCheck] = Field(default_factory=list)
    retrieval_rounds: int = Field(default=0, ge=0)
    retrieval_calls: int = Field(default=0, ge=0)
    # Older persisted tasks may contain 3; new tasks are capped by ResearchConfig at 2.
    max_retrieval_rounds: int = Field(default=2, ge=1)
    verification_rounds: int = Field(default=0, ge=0)
    max_verification_rounds: int = Field(default=2, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now()


class SearchHit(ResearchModel):
    item_id: str
    score: float


class PaperSearchResult(ResearchModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float
    rerank_score: float | None = None


class EvidenceSearchResult(ResearchModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str
    page_start: int
    page_end: int
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float
    rerank_score: float | None = None
