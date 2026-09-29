from pathlib import Path

from nanobot.research.observability.trace import PipelineTraceStore


def test_trace_store_persists_ordered_summaries_without_evidence_text(
    tmp_path: Path,
) -> None:
    store = PipelineTraceStore(tmp_path)
    store.append(
        "task-1",
        stage="plan",
        operation="research_start",
        elapsed_ms=3,
        input_summary={"sub_question_count": 1},
        output_summary={"status": "retrieving"},
    )
    store.append(
        "task-1",
        stage="retrieve",
        operation="research_retrieve",
        elapsed_ms=9,
        output_summary={"evidence_ids": ["E-1", "E-2"]},
    )

    summary = store.summary("task-1")

    assert summary["event_count"] == 2
    assert summary["tool_call_count"] == 2
    assert summary["total_recorded_ms"] == 12
    assert summary["stage_latency_ms"] == {"plan": 3, "retrieve": 9}
    assert summary["evidence_ids"] == ["E-1", "E-2"]
    assert summary["latest_verification"] is None
    assert "evidence text" not in str(summary)
    assert [item["sequence"] for item in summary["events"]] == [1, 2]
