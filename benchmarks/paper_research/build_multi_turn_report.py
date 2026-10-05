from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from nanobot.research.evaluation import AnswerBehavior, load_multi_turn_cases
from nanobot.research.models import ResearchState, ResearchTaskStatus


STATUS_BEHAVIOR = {
    ResearchTaskStatus.COMPLETED: AnswerBehavior.ANSWER,
    ResearchTaskStatus.COMPLETED_WITH_GAPS: AnswerBehavior.PARTIAL,
    ResearchTaskStatus.REFUSED: AnswerBehavior.ABSTAIN,
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


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


def _result_error(result: dict[str, Any]) -> str | None:
    explicit = result.get("error")
    if explicit:
        return str(explicit)
    answer = str(result.get("answer") or "").strip()
    normalized = answer.casefold()
    if normalized.startswith("error:") or (
        "budget configured in budget management has been exhausted" in normalized
    ):
        return answer
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    cases = {case.case_id: case for case in load_multi_turn_cases(args.benchmark)}
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("raw result must contain a JSON array")
    details: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    evidence_by_dialogue: dict[str, dict[str, Any]] = {}

    for result in raw:
        case_id = str(result.get("case_id") or "")
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"raw result contains unknown case: {case_id}")
        state = _load_state(result, args.state_dir)
        result_error = _result_error(result)
        task_status = state.status.value if state is not None else "no_state"
        actual_behavior = STATUS_BEHAVIOR.get(state.status) if state is not None else None
        retrieval_queries: list[str] = []
        retrieved_chunk_ids: list[str] = []
        cited_chunk_ids: list[str] = []
        locator_valid = None
        if state is not None:
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
        carried = evidence_by_dialogue.setdefault(
            case.dialogue_id,
            {"retrieved": [], "cited": [], "locators_valid": False},
        )
        state_completed = actual_behavior is not None and result_error is None
        if state_completed:
            carried["retrieved"] = _unique(
                [*carried["retrieved"], *retrieved_chunk_ids]
            )
            carried["cited"] = _unique([*carried["cited"], *cited_chunk_ids])
            carried["locators_valid"] = bool(
                carried["locators_valid"] or locator_valid is True
            )
        if result_error is not None:
            source_mode = "failed"
        elif state_completed:
            source_mode = "new_retrieval"
        elif carried["retrieved"]:
            source_mode = "session_reuse"
        else:
            source_mode = "untraced"
        available_chunk_ids = list(carried["retrieved"])
        relevant = set(case.relevant_chunk_ids)
        target_recall = (
            len(relevant.intersection(available_chunk_ids)) / len(relevant)
            if relevant
            else None
        )
        behavior_matches = (
            actual_behavior == case.expected_behavior if state_completed else None
        )
        citations_required = case.expected_behavior != AnswerBehavior.ABSTAIN
        pipeline_success = bool(
            state_completed
            and behavior_matches is True
            and (not citations_required or locator_valid is True)
        )
        detail = {
            "case_id": case.case_id,
            "dialogue_id": case.dialogue_id,
            "turn_index": case.turn_index,
            "expected_behavior": case.expected_behavior.value,
            "actual_behavior": actual_behavior.value if actual_behavior else None,
            "task_status": task_status,
            "task_count": len(result.get("task_ids") or []),
            "source_mode": source_mode,
            "new_retrieval_executed": state is not None,
            "new_retrieval_success": pipeline_success,
            "retrieval_call_count": state.retrieval_calls if state is not None else 0,
            "reflect_call_count": _tool_count(result, "research_reflect"),
            "retrieved_chunk_count": len(retrieved_chunk_ids),
            "available_evidence_chunk_count": len(available_chunk_ids),
            "target_chunk_recall": target_recall,
            "citation_locators_valid": locator_valid,
            "memory_probe": case.memory_probe,
            "must_preserve": case.must_preserve,
            "compaction_before_turn": bool(result.get("compaction_before_turn")),
            "compaction_applied": bool(result.get("compaction_applied")),
            "total_ms": result.get("elapsed_ms"),
            "error": result_error,
        }
        details.append(detail)
        predictions.append(
            {
                **detail,
                "query": case.query,
                "ground_truth": case.ground_truth,
                "answer_text": str(result.get("answer") or ""),
                "retrieval_queries": retrieval_queries,
                "retrieved_chunk_ids": available_chunk_ids,
                "current_retrieved_chunk_ids": retrieved_chunk_ids,
                "cited_chunk_ids": list(carried["cited"]),
                "current_cited_chunk_ids": cited_chunk_ids,
            }
        )

    denominator = len(details)
    retrieval_turns = [item for item in details if item["source_mode"] == "new_retrieval"]
    retrieval_non_abstain = [
        item
        for item in retrieval_turns
        if item["expected_behavior"] != AnswerBehavior.ABSTAIN.value
    ]
    recall_values = [
        float(item["target_chunk_recall"])
        for item in details
        if item["target_chunk_recall"] is not None and item["error"] is None
    ]
    latencies = [
        int(item["total_ms"])
        for item in details
        if item["total_ms"] is not None
    ]
    dialogue_ids = {item["dialogue_id"] for item in details}
    completed_dialogues = sum(
        all(item["error"] is None for item in details if item["dialogue_id"] == dialogue_id)
        for dialogue_id in dialogue_ids
    )
    report = {
        "run_kind": "multi_turn_agent",
        "run_id": next((str(item.get("run_id")) for item in raw if item.get("run_id")), None),
        "model": args.model,
        "dialogue_count": len(dialogue_ids),
        "turn_count": denominator,
        "metrics": {
            "completion_rate": (
                sum(item["error"] is None for item in details) / denominator
                if denominator
                else None
            ),
            "dialogue_completion_rate": (
                completed_dialogues / len(dialogue_ids) if dialogue_ids else None
            ),
            "decision_accuracy": (
                sum(
                    item["actual_behavior"] == item["expected_behavior"]
                    for item in retrieval_turns
                )
                / len(retrieval_turns)
                if retrieval_turns
                else None
            ),
            "citation_accuracy": (
                sum(
                    item["citation_locators_valid"] is True
                    for item in retrieval_non_abstain
                )
                / len(retrieval_non_abstain)
                if retrieval_non_abstain
                else None
            ),
            "new_retrieval_turn_rate": (
                sum(item["source_mode"] == "new_retrieval" for item in details)
                / denominator
                if denominator
                else None
            ),
            "session_reuse_turn_rate": (
                sum(item["source_mode"] == "session_reuse" for item in details)
                / denominator
                if denominator
                else None
            ),
            "untraced_turn_rate": (
                sum(item["source_mode"] == "untraced" for item in details)
                / denominator
                if denominator
                else None
            ),
            "retrieval_pipeline_success_rate": (
                sum(item["new_retrieval_success"] for item in retrieval_turns)
                / len(retrieval_turns)
                if retrieval_turns
                else None
            ),
            "mean_target_chunk_recall": (
                mean(recall_values) if recall_values else None
            ),
            "mean_total_ms": (
                mean(latencies) if latencies else None
            ),
            "compaction_turn_count": sum(item["compaction_applied"] for item in details),
            "post_compaction_probe_count": sum(
                item["memory_probe"] and item["compaction_before_turn"]
                for item in details
            ),
        },
        "details": details,
    }
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
