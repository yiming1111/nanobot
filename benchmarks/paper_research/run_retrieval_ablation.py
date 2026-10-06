from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from nanobot.research.config import ResearchConfig
from nanobot.research.evaluation import load_single_turn_cases
from nanobot.research.retrieval.dense import BGEM3Encoder, FaissDenseIndex
from nanobot.research.retrieval.fusion import reciprocal_rank_fusion
from nanobot.research.retrieval.sparse import BM25Index


def _score(chunk_ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    selected = chunk_ids[:k]
    matched = len(relevant.intersection(selected))
    first_rank = next(
        (rank for rank, chunk_id in enumerate(selected, start=1) if chunk_id in relevant),
        None,
    )
    return {
        "recall": matched / len(relevant),
        "precision": matched / k,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def _summarize(records: list[dict[str, Any]], variant: str, k: int) -> dict[str, float]:
    values = [item["scores"][variant][str(k)] for item in records]
    return {
        f"recall_at_{k}": mean(item["recall"] for item in values),
        f"precision_at_{k}": mean(item["precision"] for item in values),
        f"mrr_at_{k}": mean(item["mrr"] for item in values),
    }


def _point_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        key: round((value - before[key]) * 100, 3)
        for key, value in after.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 8])
    args = parser.parse_args()

    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"

    cases = load_single_turn_cases(args.benchmark)
    query_payload = json.loads(args.queries.read_text(encoding="utf-8"))
    query_records = {
        str(item["case_id"]): item for item in query_payload.get("cases", [])
    }
    scored_cases = [case for case in cases if case.relevant_chunk_ids]
    missing = [case.case_id for case in scored_cases if case.case_id not in query_records]
    if missing:
        raise ValueError(f"missing rewritten queries for: {', '.join(missing)}")

    config = ResearchConfig(data_dir=args.data_dir)
    dense = FaissDenseIndex.load(
        args.data_dir / "chunk.faiss", args.data_dir / "chunk_ids.json"
    )
    sparse = BM25Index.load(args.data_dir / "chunk_bm25.json")
    encoder = BGEM3Encoder(
        config.dense_model,
        device=config.device,
        use_fp16=config.use_fp16,
        batch_size=config.embedding_batch_size,
        max_length=config.embedding_max_length,
    )
    queries = [str(query_records[case.case_id]["rewritten_query"]) for case in scored_cases]
    vectors = encoder.encode(queries)

    records: list[dict[str, Any]] = []
    for case, query, vector in zip(scored_cases, queries, vectors, strict=True):
        dense_hits = dense.search(vector, top_k=args.candidate_k)
        sparse_hits = sparse.search(query, top_k=args.candidate_k)
        hybrid_hits = reciprocal_rank_fusion(
            {"dense": dense_hits, "sparse": sparse_hits}, k=config.rrf_k
        )
        rankings = {
            "dense": [item.item_id for item in dense_hits],
            "hybrid_rrf": [item.item_id for item in hybrid_hits],
            "hybrid_rerank": list(query_records[case.case_id].get("chunk_ids") or []),
        }
        relevant = set(case.relevant_chunk_ids)
        records.append(
            {
                "case_id": case.case_id,
                "query": query,
                "relevant_chunk_ids": list(case.relevant_chunk_ids),
                "rankings": {name: ids[: max(args.ks)] for name, ids in rankings.items()},
                "scores": {
                    name: {
                        str(k): _score(ids, relevant, k)
                        for k in args.ks
                    }
                    for name, ids in rankings.items()
                },
            }
        )

    metrics = {
        variant: {
            str(k): _summarize(records, variant, k)
            for k in args.ks
        }
        for variant in ("dense", "hybrid_rrf", "hybrid_rerank")
    }
    deltas: dict[str, dict[str, dict[str, float]]] = {
        "hybrid_rrf_minus_dense_pts": {},
        "hybrid_rerank_minus_hybrid_rrf_pts": {},
    }
    for k in args.ks:
        key = str(k)
        deltas["hybrid_rrf_minus_dense_pts"][key] = _point_delta(
            metrics["hybrid_rrf"][key], metrics["dense"][key]
        )
        deltas["hybrid_rerank_minus_hybrid_rrf_pts"][key] = _point_delta(
            metrics["hybrid_rerank"][key], metrics["hybrid_rrf"][key]
        )

    payload = {
        "run_kind": "retrieval_ablation",
        "case_count": len(records),
        "candidate_k": args.candidate_k,
        "ks": args.ks,
        "metrics": metrics,
        "deltas": deltas,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": metrics, "deltas": deltas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
