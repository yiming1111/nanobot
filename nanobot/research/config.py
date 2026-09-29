"""Configuration for the scientific-paper research extension."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchConfig(BaseSettings):
    """Runtime settings owned by the research extension.

    Keeping these settings outside nanobot's global schema lets the extension
    evolve without expanding the core agent configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="NANOBOT_RESEARCH_",
        env_file=None,
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".nanobot" / "research")
    dense_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    use_fp16: bool = False
    embedding_batch_size: int = Field(default=8, ge=1, le=256)
    embedding_max_length: int = Field(default=1024, ge=128, le=8192)
    chunk_size_words: int = Field(default=350, ge=80, le=2000)
    chunk_overlap_words: int = Field(default=60, ge=0, le=500)
    dense_candidates: int = Field(default=40, ge=1, le=500)
    sparse_candidates: int = Field(default=40, ge=1, le=500)
    rerank_candidates: int = Field(default=12, ge=1, le=200)
    default_top_k: int = Field(default=4, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    # One initial semantic search plus one Reflect-triggered retry.
    max_retrieval_rounds: int = Field(default=2, ge=1, le=2)

    @property
    def corpus_file(self) -> Path:
        return self.data_dir / "papers.jsonl"

    @property
    def chunks_file(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def states_dir(self) -> Path:
        return self.data_dir / "states"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
