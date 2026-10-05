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
from nanobot.research.evaluation import AnswerBehavior, load_multi_turn_cases
from nanobot.research.models import PaperChunk


SYSTEM_PROMPT = """You are evaluating one turn of a multi-turn paper QA conversation.
Return one JSON object only, without markdown.

Judge the generated answer against the reference answer and the required conversational
facts in MUST_PRESERVE. Do not require verbatim wording. A fact is preserved when the
answer resolves the reference or constraint consistently with the earlier conversation.
For a partial-answer case, an answer is correct only when it gives the supported part and
states the evidence gap. Do not use outside knowledge.

Also evaluate four RAG dimensions from the retrieved contexts:
1. Split the generated answer into factual claims and mark a claim supported only when
   the retrieved contexts entail it.
2. Score how directly and completely the answer addresses the question from 0 to 1.
3. List only context IDs that directly help answer the question.
4. Split the reference answer into factual claims and mark whether each is covered by
   at least one retrieved context.

Use this schema exactly:
{
  "answer_correct": true,
  "answer_reason": "...",
  "preserved": [{"text": "...", "preserved": true}],
  "answer_claims": [{"text": "...", "supported": true}],
  "answer_relevancy": 0.0,
  "answer_relevancy_reason": "...",
  "relevant_context_ids": ["chunk-id"],
  "reference_claims": [{"text": "...", "covered": true}]
}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_chunks(path: Path) -> dict[str, PaperChunk]:
    chunks: dict[str, PaperChunk] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            chunk = PaperChunk.model_validate_json(raw_line)
            chunks[chunk.chunk_id] = chunk
    return chunks


def _ratio(items: list[dict[str, Any]], field: str) -> float:
    if not items:
        return 1.0
    return sum(bool(item.get(field)) for item in items) / len(items)


def _parse_judgment(content: str, required: list[str]) -> dict[str, Any]:
    parsed = json_repair.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    preserved = parsed.get("preserved")
    if not isinstance(preserved, list):
        raise ValueError("judge response has no preserved list")
    values = [bool(item.get("preserved")) for item in preserved if isinstance(item, dict)]
    if len(values) != len(required):
        raise ValueError(
            f"judge returned {len(values)} preservation decisions for {len(required)} facts"
        )
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
    return {
        "answer_correct": bool(parsed.get("answer_correct")),
        "answer_reason": str(parsed.get("answer_reason") or ""),
        "preserved": [
            {"text": text, "preserved": value}
            for text, value in zip(required, values, strict=True)
        ],
        "answer_claims": answer_claims,
        "answer_relevancy": relevancy,
        "answer_relevancy_reason": str(
            parsed.get("answer_relevancy_reason") or ""
        ),
        "relevant_context_ids": [str(value) for value in relevant_context_ids],
        "reference_claims": reference_claims,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in records if item.get("error") is None]
    semantic_scored = [
        item
        for item in scored
        if item.get("expected_behavior") != AnswerBehavior.ABSTAIN.value
    ]
    with_constraints = [item for item in scored if item.get("preserved")]
    memory_probes = [item for item in scored if item.get("memory_probe")]
    post_compaction = [
        item
        for item in memory_probes
        if item.get("compaction_before_turn")
    ]

    def all_preserved(item: dict[str, Any]) -> bool:
        return all(bool(value.get("preserved")) for value in item.get("preserved") or [])

    preservation_values = [
        bool(value.get("preserved"))
        for item in with_constraints
        for value in item["preserved"]
    ]

    def average(field: str) -> float | None:
        values = [
            float(item[field])
            for item in semantic_scored
            if item.get(field) is not None
        ]
        return mean(values) if values else None

    return {
        "case_count": len(records),
        "scored_case_count": len(scored),
        "error_count": len(records) - len(scored),
        "answer_correctness": (
            mean(bool(item["answer_correct"]) for item in scored) if scored else None
        ),
        "semantic_case_count": len(semantic_scored),
        "faithfulness": average("faithfulness"),
        "answer_relevancy": average("answer_relevancy"),
        "context_relevancy": average("context_relevancy"),
        "context_recall": average("context_recall"),
        "preserved_fact_accuracy": (
            mean(preservation_values) if preservation_values else None
        ),
        "followup_consistency_rate": (
            mean(all_preserved(item) for item in with_constraints)
            if with_constraints
            else None
        ),
        "memory_probe_consistency_rate": (
            mean(all_preserved(item) for item in memory_probes)
            if memory_probes
            else None
        ),
        "post_compaction_probe_count": len(post_compaction),
        "post_compaction_memory_consistency_rate": (
            mean(all_preserved(item) for item in post_compaction)
            if post_compaction
            else None
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    cases = {case.case_id: case for case in load_multi_turn_cases(args.benchmark)}
    predictions = {
        str(item["case_id"]): item for item in _read_jsonl(args.predictions)
    }
    chunks = _load_chunks(args.chunks)
    unknown = sorted(set(predictions).difference(cases))
    if unknown:
        raise ValueError(f"predictions contain unknown cases: {', '.join(unknown)}")
    records: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved.get("evaluation_version") == 2:
            records = list(saved.get("cases", []))
    completed = {str(item.get("case_id")) for item in records}

    runtime_config = _load_runtime_config(None, None)
    provider = make_provider(runtime_config)
    model = runtime_config.resolve_preset().model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def evaluate(case_id: str, prediction: dict[str, Any]) -> dict[str, Any]:
        case = cases[case_id]
        if prediction.get("error"):
            return {
                "case_id": case_id,
                "dialogue_id": case.dialogue_id,
                "turn_index": case.turn_index,
                "memory_probe": case.memory_probe,
                "expected_behavior": case.expected_behavior.value,
                "compaction_before_turn": bool(
                    prediction.get("compaction_before_turn")
                ),
                "error": f"Upstream agent error: {prediction['error']}",
            }
        context_ids = list(
            dict.fromkeys(
                str(value)
                for value in prediction.get("retrieved_chunk_ids") or []
            )
        )
        contexts = [chunks[chunk_id] for chunk_id in context_ids if chunk_id in chunks]
        context_text = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.section}; pages {chunk.page_start}-{chunk.page_end}\n"
            f"{chunk.text}"
            for chunk in contexts
        ) or "(no retrieved context)"
        prompt = (
            f"QUESTION:\n{case.query}\n\n"
            f"EXPECTED_BEHAVIOR:\n{case.expected_behavior.value}\n\n"
            f"REFERENCE_ANSWER:\n{case.ground_truth}\n\n"
            f"MUST_PRESERVE:\n{json.dumps(case.must_preserve, ensure_ascii=False)}\n\n"
            f"GENERATED_ANSWER:\n{prediction.get('answer_text') or ''}\n\n"
            f"RETRIEVED_CONTEXTS:\n{context_text}"
        )
        try:
            async with semaphore:
                response = await provider.chat_with_retry(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                )
            if response.finish_reason == "error" or not response.content:
                raise RuntimeError(response.content or "empty judge response")
            judgment = _parse_judgment(response.content, case.must_preserve)
            relevant_ids = {
                value
                for value in judgment["relevant_context_ids"]
                if value in context_ids
            }
            semantic_scored = case.expected_behavior != AnswerBehavior.ABSTAIN
            return {
                "case_id": case_id,
                "dialogue_id": case.dialogue_id,
                "turn_index": case.turn_index,
                "memory_probe": case.memory_probe,
                "expected_behavior": case.expected_behavior.value,
                "compaction_before_turn": bool(
                    prediction.get("compaction_before_turn")
                ),
                "answer_correct": judgment["answer_correct"],
                "answer_reason": judgment["answer_reason"],
                "preserved": judgment["preserved"],
                "faithfulness": (
                    _ratio(judgment["answer_claims"], "supported")
                    if semantic_scored
                    else None
                ),
                "answer_relevancy": (
                    judgment["answer_relevancy"] if semantic_scored else None
                ),
                "context_relevancy": (
                    len(relevant_ids) / len(context_ids)
                    if semantic_scored and context_ids
                    else 0.0
                    if semantic_scored
                    else None
                ),
                "context_recall": (
                    _ratio(judgment["reference_claims"], "covered")
                    if semantic_scored
                    else None
                ),
                "context_count": len(context_ids),
                "answer_claims": judgment["answer_claims"],
                "reference_claims": judgment["reference_claims"],
                "relevant_context_ids": sorted(relevant_ids),
                "answer_relevancy_reason": judgment["answer_relevancy_reason"],
                "error": None,
            }
        except Exception as exc:
            return {"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"}

    order = {case_id: index for index, case_id in enumerate(predictions)}
    pending = [
        (case_id, prediction)
        for case_id, prediction in predictions.items()
        if case_id not in completed
    ]
    tasks = [
        asyncio.create_task(evaluate(case_id, prediction))
        for case_id, prediction in pending
    ]
    finished = len(completed)
    for task in asyncio.as_completed(tasks):
        record = await task
        records.append(record)
        records.sort(key=lambda item: order.get(str(item.get("case_id")), len(order)))
        finished += 1
        report = {
            "evaluation_version": 2,
            "model": model,
            "metrics": _summarize(records),
            "cases": records,
        }
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "progress": f"{finished}/{len(predictions)}",
                    "case_id": record.get("case_id"),
                    "answer_correct": record.get("answer_correct"),
                    "error": record.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
