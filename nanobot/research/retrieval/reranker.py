"""Cross-encoder reranking using BAAI's FlagEmbedding package."""

from __future__ import annotations

from collections.abc import Sequence

from nanobot.research.retrieval.dense import _research_dependency_error


class BGEReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise _research_dependency_error("FlagEmbedding") from exc
        self._model = FlagReranker(model_name, devices=device, use_fp16=use_fp16)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        raw = self._model.compute_score(
            [[query, document] for document in documents],
            normalize=True,
        )
        values = raw if isinstance(raw, list) else [raw]
        return [float(value) for value in values]
