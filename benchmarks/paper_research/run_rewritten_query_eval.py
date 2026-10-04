from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from nanobot.cli.runtime_config import _load_runtime_config
from nanobot.providers.factory import make_provider
from nanobot.research.config import ResearchConfig
from nanobot.research.evaluation import load_single_turn_cases
from nanobot.research.retrieval.service import HybridRetrievalService


REWRITE_SYSTEM_PROMPT = """Convert the user's scientific-paper question into exactly one concise English retrieval query for searching English paper chunks.
Preserve every explicit paper name, algorithm name, formula, year, number, comparison side, negation, and scope.
Resolve only clear technical aliases. Do not answer the question. Do not add related concepts, examples, candidate answers, variables, methods, populations, or constraints that the user did not state.
Return only the English query on one line, without a label, explanation, quotation marks, or Markdown."""


def clean_rewritten_query(value: str | None) -> str:
    """Normalize a model response while rejecting explanatory output."""
    text = (value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:text|txt)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("query rewrite must contain exactly one non-empty line")
    query = re.sub(r"^(?:query|english query)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    query = query.strip().strip('"').strip("'").strip()
    if not re.search(r"[A-Za-z]", query):
        raise ValueError("query rewrite does not contain English text")
    if len(query) > 500:
        raise ValueError("query rewrite exceeds 500 characters")
    return query


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def build_metrics(records: list[dict[str, Any]], timeout_ms: int) -> dict[str, float | None]:
    scored = [item for item in records if item.get("recall_at_k") is not None]
    retrieval_latencies = [
        int(item["retrieval_ms"])
        for item in records
        if item.get("retrieval_ms") is not None
    ]
    rewrite_latencies = [
        int(item["rewrite_ms"])
        for item in records
        if item.get("rewrite_ms") is not None
    ]

    def average(field: str) -> float | None:
        values = [float(item[field]) for item in scored if item.get(field) is not None]
        return mean(values) if values else None

    denominator = len(records)
    return {
        "recall_at_k": average("recall_at_k"),
        "precision_at_k": average("precision_at_k"),
        "mrr": average("reciprocal_rank"),
        "mean_rewrite_ms": mean(rewrite_latencies) if rewrite_latencies else None,
        "mean_retrieval_ms": mean(retrieval_latencies) if retrieval_latencies else None,
        "p95_retrieval_ms": percentile(retrieval_latencies, 0.95),
        "timeout_rate": (
            sum(value > timeout_ms for value in retrieval_latencies) / denominator
            if denominator
            else None
        ),
        "rewrite_error_rate": (
            sum(item.get("rewrite_error") is not None for item in records) / denominator
            if denominator
            else None
        ),
        "retrieval_error_rate": (
            sum(item.get("retrieval_error") is not None for item in records) / denominator
            if denominator
            else None
        ),
    }


def score_retrieval_record(case: Any, record: dict[str, Any], top_k: int) -> None:
    """Recompute label-dependent metrics from saved retrieval output."""
    record["expected_behavior"] = case.expected_behavior.value
    record["recall_at_k"] = None
    record["precision_at_k"] = None
    record["reciprocal_rank"] = None
    relevant = set(case.relevant_chunk_ids)
    if not relevant or record.get("retrieval_error") is not None:
        return
    chunk_ids = list(record.get("chunk_ids") or [])
    matched = len(relevant.intersection(chunk_ids))
    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(chunk_ids, start=1)
            if chunk_id in relevant
        ),
        None,
    )
    record["recall_at_k"] = matched / len(relevant)
    record["precision_at_k"] = matched / top_k
    record["reciprocal_rank"] = 1.0 / first_rank if first_rank else 0.0


async def rewrite_query(provider: Any, question: str) -> str:
    response = await provider.chat_with_retry(
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=128,
        temperature=0.0,
    )
    if response.finish_reason == "error":
        detail = response.content or response.error_code or response.error_kind or "unknown error"
        raise RuntimeError(detail)
    return clean_rewritten_query(response.content)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")

    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = load_single_turn_cases(args.benchmark)
    runtime_config = _load_runtime_config(None, str(args.workspace))
    model = runtime_config.resolve_preset().model
    existing: dict[str, dict[str, Any]] = {}
    created_at = datetime.now(UTC).isoformat()
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        created_at = saved.get("created_at", created_at)
        existing = {item["case_id"]: item for item in saved.get("cases", [])}

    records = [
        existing.get(
            case.case_id,
            {
                "case_id": case.case_id,
                "original_query": case.query,
                "rewritten_query": None,
                "expected_behavior": case.expected_behavior.value,
                "chunk_ids": [],
                "recall_at_k": None,
                "precision_at_k": None,
                "reciprocal_rank": None,
                "rewrite_ms": None,
                "retrieval_ms": None,
                "timed_out": False,
                "rewrite_error": None,
                "retrieval_error": None,
            },
        )
        for case in cases
    ]
    for case, record in zip(cases, records, strict=True):
        if record.get("retrieval_ms") is not None:
            score_retrieval_record(case, record, args.top_k)

    def save(status: str) -> None:
        payload = {
            "created_at": created_at,
            "status": status,
            "model": model,
            "case_count": len(cases),
            "scored_case_count": sum(bool(case.relevant_chunk_ids) for case in cases),
            "top_k": args.top_k,
            "timeout_ms": args.timeout_seconds * 1000,
            "metrics": build_metrics(records, args.timeout_seconds * 1000),
            "cases": records,
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    pending_rewrites = [item for item in records if not item.get("rewritten_query")]
    if pending_rewrites:
        provider = make_provider(runtime_config)
        try:
            for index, (case, record) in enumerate(zip(cases, records, strict=True), start=1):
                if record.get("rewritten_query"):
                    continue
                started = perf_counter()
                try:
                    record["rewritten_query"] = await rewrite_query(provider, case.query)
                    record["rewrite_error"] = None
                except Exception as exc:
                    record["rewrite_error"] = f"{type(exc).__name__}: {exc}"
                record["rewrite_ms"] = round((perf_counter() - started) * 1000)
                save("rewriting")
                print(
                    f"[rewrite {index}/{len(cases)}] {case.case_id}: "
                    f"{record.get('rewritten_query') or record.get('rewrite_error')}",
                    flush=True,
                )
        finally:
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()

    service = None
    if any(
        item.get("retrieval_ms") is None and item.get("rewritten_query")
        for item in records
    ):
        if args.hf_home is not None:
            os.environ["HF_HOME"] = str(args.hf_home)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        service = HybridRetrievalService(ResearchConfig(data_dir=args.data_dir))
        service.prepare_indexes()
        service.prepare_models()

    for index, (case, record) in enumerate(zip(cases, records, strict=True), start=1):
        if record.get("retrieval_ms") is not None:
            continue
        query = record.get("rewritten_query")
        if not query:
            continue
        assert service is not None
        started = perf_counter()
        evidence = []
        try:
            evidence = service.retrieve_evidence(str(query), top_k=args.top_k)
            record["retrieval_error"] = None
        except Exception as exc:
            record["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        retrieval_ms = round((perf_counter() - started) * 1000)
        record["retrieval_ms"] = retrieval_ms
        record["timed_out"] = retrieval_ms > args.timeout_seconds * 1000
        record["chunk_ids"] = [item.chunk_id for item in evidence]
        score_retrieval_record(case, record, args.top_k)
        save("retrieving")
        suffix = f" error={record['retrieval_error']}" if record["retrieval_error"] else ""
        print(f"[retrieve {index}/{len(cases)}] {case.case_id} {retrieval_ms}ms{suffix}", flush=True)

    save("completed")


if __name__ == "__main__":
    asyncio.run(main())
