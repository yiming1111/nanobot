from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Any

import json_repair

from nanobot.cli.runtime_config import _load_runtime_config
from nanobot.providers.factory import make_provider
from nanobot.research.evaluation import (
    AnswerBehavior,
    load_single_turn_cases,
    load_single_turn_predictions,
)
from nanobot.research.models import PaperChunk


SYSTEM_PROMPT = """You are evaluating a paper-grounded RAG system.
Return one JSON object only, without markdown.

Evaluate four dimensions from the supplied question, reference answer, generated answer,
and retrieved contexts.

1. Faithfulness: split the generated answer into independently verifiable factual claims.
Mark a claim supported only when the retrieved contexts entail it. Do not count formatting,
citations, or statements that merely acknowledge an evidence gap as factual claims.
2. Answer relevancy: score how directly and completely the generated answer addresses the
question, using a number from 0 to 1. A properly limited partial answer can still be relevant.
Do not reward factual correctness here; that belongs to faithfulness.
3. Context relevancy: list the IDs of retrieved contexts that provide information directly
useful for answering the question. Mere topic overlap is insufficient.
4. Context recall: split the reference answer into independently verifiable factual claims
and mark whether each claim is covered by at least one retrieved context. Do not count
benchmark instructions or statements that merely describe missing corpus evidence.

Use this schema exactly:
{
  "answer_claims": [{"text": "...", "supported": true}],
  "answer_relevancy": 0.0,
  "answer_relevancy_reason": "...",
  "relevant_context_ids": ["chunk-id"],
  "reference_claims": [{"text": "...", "covered": true}]
}
"""


def load_chunks(path: Path) -> dict[str, PaperChunk]:
    chunks: dict[str, PaperChunk] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            chunk = PaperChunk.model_validate_json(raw_line)
            chunks[chunk.chunk_id] = chunk
    return chunks


def parse_judgment(content: str) -> dict[str, Any]:
    parsed = json_repair.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    answer_claims = parsed.get("answer_claims")
    reference_claims = parsed.get("reference_claims")
    relevant_context_ids = parsed.get("relevant_context_ids")
    if not isinstance(answer_claims, list) or not isinstance(reference_claims, list):
        raise ValueError("judge response is missing claim lists")
    if not isinstance(relevant_context_ids, list):
        raise ValueError("judge response is missing relevant_context_ids")
    relevancy = float(parsed.get("answer_relevancy"))
    if not 0 <= relevancy <= 1:
        raise ValueError("answer_relevancy must be between 0 and 1")
    return parsed


def ratio(items: list[dict[str, Any]], field: str) -> float:
    if not items:
        return 1.0
    return sum(bool(item.get(field)) for item in items) / len(items)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("error") is None]

    def average(field: str) -> float | None:
        values = [float(record[field]) for record in scored if record.get(field) is not None]
        return mean(values) if values else None

    return {
        "case_count": len(records),
        "scored_case_count": len(scored),
        "error_count": len(records) - len(scored),
        "faithfulness": average("faithfulness"),
        "answer_relevancy": average("answer_relevancy"),
        "context_relevancy": average("context_relevancy"),
        "context_recall": average("context_recall"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    cases = [
        case
        for case in load_single_turn_cases(args.benchmark)
        if case.expected_behavior != AnswerBehavior.ABSTAIN
    ]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    predictions = {
        prediction.case_id: prediction
        for prediction in load_single_turn_predictions(args.predictions)
    }
    chunks = load_chunks(args.chunks)
    records: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        records = list(saved.get("cases", []))
    completed = {str(record.get("case_id")) for record in records}

    runtime_config = _load_runtime_config(None, None)
    provider = make_provider(runtime_config)
    model = runtime_config.resolve_preset().model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def evaluate_case(case: Any) -> dict[str, Any]:
        prediction = predictions.get(case.case_id)
        if prediction is None:
            return {"case_id": case.case_id, "error": "prediction missing"}
        context_ids = list(dict.fromkeys(prediction.retrieved_chunk_ids))
        contexts = [chunks[chunk_id] for chunk_id in context_ids if chunk_id in chunks]
        context_text = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.section}; pages {chunk.page_start}-{chunk.page_end}\n"
            f"{chunk.text}"
            for chunk in contexts
        ) or "(no retrieved context)"
        user_prompt = (
            f"QUESTION:\n{case.query}\n\n"
            f"REFERENCE ANSWER:\n{case.ground_truth}\n\n"
            f"GENERATED ANSWER:\n{prediction.answer_text}\n\n"
            f"RETRIEVED CONTEXTS:\n{context_text}"
        )
        try:
            async with semaphore:
                response = await provider.chat_with_retry(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                )
            if response.finish_reason == "error" or not response.content:
                raise RuntimeError(response.content or "empty judge response")
            judgment = parse_judgment(response.content)
            answer_claims = judgment["answer_claims"]
            reference_claims = judgment["reference_claims"]
            relevant_ids = {
                str(value)
                for value in judgment["relevant_context_ids"]
                if str(value) in context_ids
            }
            return {
                "case_id": case.case_id,
                "context_count": len(context_ids),
                "faithfulness": ratio(answer_claims, "supported"),
                "answer_relevancy": float(judgment["answer_relevancy"]),
                "context_relevancy": (
                    len(relevant_ids) / len(context_ids) if context_ids else 0.0
                ),
                "context_recall": ratio(reference_claims, "covered"),
                "answer_claims": answer_claims,
                "reference_claims": reference_claims,
                "relevant_context_ids": sorted(relevant_ids),
                "answer_relevancy_reason": judgment.get("answer_relevancy_reason", ""),
                "error": None,
            }
        except Exception as exc:
            return {
                "case_id": case.case_id,
                "error": f"{type(exc).__name__}: {exc}",
            }

    pending = [case for case in cases if case.case_id not in completed]
    order = {case.case_id: index for index, case in enumerate(cases)}
    tasks = [asyncio.create_task(evaluate_case(case)) for case in pending]
    finished = len(completed)
    for task in asyncio.as_completed(tasks):
        record = await task
        records.append(record)
        records.sort(key=lambda item: order.get(str(item.get("case_id")), len(order)))
        finished += 1
        report = {"model": model, "metrics": summarize(records), "cases": records}
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "progress": f"{finished}/{len(cases)}",
                    "case_id": record.get("case_id"),
                    "faithfulness": record.get("faithfulness"),
                    "answer_relevancy": record.get("answer_relevancy"),
                    "context_relevancy": record.get("context_relevancy"),
                    "context_recall": record.get("context_recall"),
                    "error": record.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
