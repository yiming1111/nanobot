"""A small persistent BM25 implementation for paper and chunk indexes."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from nanobot.research.models import SearchHit

_LATIN_TERM = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English terms and CJK character unigrams/bigrams."""
    lowered = text.lower()
    tokens = _LATIN_TERM.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.document_tokens: dict[str, list[str]] = {}
        self.document_frequencies: dict[str, int] = {}
        self.average_length = 0.0

    def build(self, documents: dict[str, str]) -> None:
        self.document_tokens = {
            document_id: tokenize(text) for document_id, text in documents.items()
        }
        frequencies: Counter[str] = Counter()
        total_length = 0
        for tokens in self.document_tokens.values():
            frequencies.update(set(tokens))
            total_length += len(tokens)
        self.document_frequencies = dict(frequencies)
        self.average_length = total_length / max(len(self.document_tokens), 1)

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        allowed_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        query_terms = tokenize(query)
        if not query_terms or not self.document_tokens:
            return []
        documents = [
            (document_id, tokens)
            for document_id, tokens in self.document_tokens.items()
            if allowed_ids is None or document_id in allowed_ids
        ]
        if not documents:
            return []
        document_count = len(documents)
        average_length = sum(len(tokens) for _document_id, tokens in documents) / document_count
        scoped_frequencies = (
            self.document_frequencies
            if allowed_ids is None
            else {
                term: sum(1 for _document_id, tokens in documents if term in tokens)
                for term in set(query_terms)
            }
        )
        scores: list[SearchHit] = []
        for document_id, tokens in documents:
            term_counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = scoped_frequencies.get(term, 0)
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(average_length, 1.0)
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append(SearchHit(item_id=document_id, score=score))
        scores.sort(key=lambda item: (-item.score, item.item_id))
        return scores[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "document_tokens": self.document_tokens,
            "document_frequencies": self.document_frequencies,
            "average_length": self.average_length,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=float(payload["k1"]), b=float(payload["b"]))
        index.document_tokens = {
            str(key): [str(token) for token in tokens]
            for key, tokens in payload["document_tokens"].items()
        }
        index.document_frequencies = {
            str(key): int(value) for key, value in payload["document_frequencies"].items()
        }
        index.average_length = float(payload["average_length"])
        return index
