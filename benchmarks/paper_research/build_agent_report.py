from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from nanobot.research.evaluation import (
    AnswerBehavior,
    SingleTurnPrediction,
    load_single_turn_cases,
    score_single_turn_outputs,
)
from nanobot.research.models import ResearchState, ResearchTaskStatus


STATUS_BEHAVIOR = {
    ResearchTaskStatus.COMPLETED: AnswerBehavior.ANSWER,
    ResearchTaskStatus.COMPLETED_WITH_GAPS: AnswerBehavior.PARTIAL,
    ResearchTaskStatus.REFUSED: AnswerBehavior.ABSTAIN,
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _load_state(result: dict[str, Any], state_dir: Path) -> ResearchState | None:
    task_ids = result.get("task_ids") or []
    if len(task_ids) != 1:
        return None
    path = state_dir / f"{task_ids[0]}.json"
    if not path.exists():
        return None
    return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))


def _tool_count(result: dict[str, Any], suffix: str) -> int:
    return sum(
        1
        for progress in result.get("progress") or []
        for event in progress.get("tool_events") or []
        if str(event.get("name") or "").endswith(suffix)
        and event.get("phase") == "end"
        and not event.get("error")
    )


def build_report(
    benchmark_path: Path,
    raw_path: Path,
    state_dir: Path,
    *,
    model: str | None = None,
) -> tuple[list[SingleTurnPrediction], dict[str, Any]]:
    cases = load_single_turn_cases(benchmark_path)
    cases_by_id = {case.case_id: case for case in cases}
    raw_results = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_by_id = {str(item["case_id"]): item for item in raw_results}
    predictions: list[SingleTurnPrediction] = []
    details: list[dict[str, Any]] = []

    for case in cases:
        result = raw_by_id.get(case.case_id)
        if result is None:
            continue
        state = _load_state(result, state_dir)
        retrieval_queries: list[str] = []
        retrieved_chunk_ids: list[str] = []
        cited_chunk_ids: list[str] = []
        actual_behavior = None
        locator_valid = None
        task_status = "no_state"
        retrieval_calls = 0
        sub_question_count = 0

        if state is not None:
            task_status = state.status.value
            actual_behavior = STATUS_BEHAVIOR.get(state.status)
            retrieval_calls = state.retrieval_calls
            sub_question_count = len(state.plan.sub_questions)
            retrieval_queries = _unique(
                [
                    query
                    for sub_question in state.plan.sub_questions
                    for query in sub_question.queries
                ]
            )
            retrieved_chunk_ids = _unique(
                [
                    chunk_id
                    for sub_question in state.plan.sub_questions
                    for chunk_id in sub_question.chunk_ids
                ]
            )
            citations = state.metadata.get("finalization", {}).get("citations", [])
            cited_chunk_ids = _unique(
                [
                    str(item["chunk_id"])
                    for item in citations
                    if isinstance(item, dict) and item.get("chunk_id")
                ]
            )
            if cited_chunk_ids:
                locator_valid = bool(state.citation_locator_checks) and all(
                    item.locatable for item in state.citation_locator_checks
                )

        prediction = SingleTurnPrediction(
            case_id=case.case_id,
            answer_text=str(result.get("answer") or ""),
            actual_behavior=actual_behavior,
            retrieval_queries=retrieval_queries,
            retrieved_chunk_ids=retrieved_chunk_ids,
            cited_chunk_ids=cited_chunk_ids,
            citation_locators_valid=locator_valid,
            task_id=state.task_id if state is not None else None,
            total_ms=result.get("elapsed_ms"),
            error=result.get("error"),
        )
        predictions.append(prediction)

        relevant = set(case.relevant_chunk_ids)
        matched = relevant.intersection(retrieved_chunk_ids)
        retrieval_recall = len(matched) / len(relevant) if relevant else None
        behavior_matches = actual_behavior == case.expected_behavior
        citations_required = case.expected_behavior != AnswerBehavior.ABSTAIN
        pipeline_success = bool(
            state is not None
            and result.get("error") is None
            and behavior_matches
            and (not citations_required or locator_valid is True)
        )
        details.append(
            {
                "case_id": case.case_id,
                "expected_behavior": case.expected_behavior.value,
                "actual_behavior": actual_behavior.value if actual_behavior else None,
                "task_status": task_status,
                "pipeline_executed": state is not None,
                "pipeline_success": pipeline_success,
                "retrieval_call_count": retrieval_calls,
                "reflect_call_count": _tool_count(result, "research_reflect"),
                "sub_question_count": sub_question_count,
                "retrieval_queries": retrieval_queries,
                "retrieved_chunk_count": len(retrieved_chunk_ids),
                "target_chunk_recall": retrieval_recall,
                "cited_chunk_count": len(cited_chunk_ids),
                "citation_locators_valid": locator_valid,
                "total_ms": result.get("elapsed_ms"),
                "error": result.get("error"),
            }
        )

    deterministic = score_single_turn_outputs(cases, predictions)
    latencies = [
        int(item["total_ms"])
        for item in details
        if item["total_ms"] is not None
    ]
    scored_recall = [
        float(item["target_chunk_recall"])
        for item in details
        if item["target_chunk_recall"] is not None
    ]
    behavior_latency: dict[str, dict[str, float | int | None]] = {}
    for behavior in AnswerBehavior:
        values = [
            int(item["total_ms"])
            for item in details
            if item["expected_behavior"] == behavior.value
            and item["total_ms"] is not None
        ]
        behavior_latency[behavior.value] = {
            "case_count": len(values),
            "mean_total_ms": mean(values) if values else None,
            "p95_total_ms": _percentile(values, 0.95),
        }

    pipeline_executed = sum(item["pipeline_executed"] for item in details)
    pipeline_successes = sum(item["pipeline_success"] for item in details)
    no_state_ids = [
        item["case_id"] for item in details if not item["pipeline_executed"]
    ]
    zero_target_hit_ids = [
        item["case_id"]
        for item in details
        if item["target_chunk_recall"] == 0.0
    ]
    mismatch_ids = [
        item["case_id"]
        for item in details
        if item["pipeline_executed"]
        and item["actual_behavior"] != item["expected_behavior"]
    ]
    report = {
        "run_kind": "isolated_agent_full",
        "run_id": next(
            (str(item.get("run_id")) for item in raw_results if item.get("run_id")),
            None,
        ),
        "model": model,
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "metrics": {
            **deterministic.metrics,
            "pipeline_execution_rate": pipeline_executed / len(cases),
            "pipeline_success_rate": pipeline_successes / len(cases),
            "mean_target_chunk_recall": mean(scored_recall) if scored_recall else None,
            "zero_target_hit_rate": (
                sum(value == 0.0 for value in scored_recall) / len(scored_recall)
                if scored_recall
                else None
            ),
            "mean_total_ms": mean(latencies) if latencies else None,
            "p95_total_ms": _percentile(latencies, 0.95),
            "max_total_ms": max(latencies) if latencies else None,
            "mean_retrieval_calls": mean(
                int(item["retrieval_call_count"])
                for item in details
                if item["pipeline_executed"]
            ),
            "mean_reflect_calls": mean(
                int(item["reflect_call_count"])
                for item in details
                if item["pipeline_executed"]
            ),
        },
        "latency_by_expected_behavior": behavior_latency,
        "task_status_counts": dict(Counter(item["task_status"] for item in details)),
        "pipeline_bypass_case_ids": no_state_ids,
        "zero_target_hit_case_ids": zero_target_hit_ids,
        "behavior_mismatch_case_ids": mismatch_ids,
        "cases": details,
    }
    return predictions, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    predictions, report = build_report(
        args.benchmark,
        args.raw,
        args.state_dir,
        model=args.model,
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "\n".join(item.model_dump_json() for item in predictions) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
