"""Reciprocal-rank fusion for heterogeneous result lists."""

from __future__ import annotations

from dataclasses import dataclass, field

from nanobot.research.models import SearchHit


@dataclass(slots=True)
class FusedHit:
    item_id: str
    fused_score: float = 0.0
    source_scores: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: dict[str, list[SearchHit]],
    *,
    k: int = 60,
) -> list[FusedHit]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    fused: dict[str, FusedHit] = {}
    for source, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            current = fused.setdefault(hit.item_id, FusedHit(item_id=hit.item_id))
            current.fused_score += 1.0 / (k + rank)
            current.source_scores[source] = hit.score
    return sorted(fused.values(), key=lambda item: (-item.fused_score, item.item_id))
