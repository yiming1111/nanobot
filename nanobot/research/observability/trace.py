"""Structured, privacy-conscious traces for paper-research tasks."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import Field

from nanobot.research.models import ResearchModel, utc_now

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class PipelineTraceEvent(ResearchModel):
    """One completed research operation without copying full evidence text."""

    sequence: int = Field(ge=1)
    stage: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    status: Literal["completed", "failed"] = "completed"
    elapsed_ms: int = Field(default=0, ge=0)
    sub_question_id: str | None = None
    claim_id: str | None = None
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class PipelineTrace(ResearchModel):
    task_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    events: list[PipelineTraceEvent] = Field(default_factory=list)


class PipelineTraceStore:
    """Persist one append-only JSON trace per research task."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        if _SAFE_TASK_ID.fullmatch(task_id) is None:
            raise ValueError("invalid research task id")
        return self.root / f"{task_id}.json"

    def _lock(self, task_id: str) -> FileLock:
        return FileLock(str(self._path(task_id)) + ".lock")

    def append(
        self,
        task_id: str,
        *,
        stage: str,
        operation: str,
        elapsed_ms: int = 0,
        sub_question_id: str | None = None,
        claim_id: str | None = None,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        status: Literal["completed", "failed"] = "completed",
        error: str | None = None,
    ) -> PipelineTrace:
        path = self._path(task_id)
        with self._lock(task_id):
            if path.exists():
                trace = PipelineTrace.model_validate_json(path.read_text(encoding="utf-8"))
            else:
                trace = PipelineTrace(task_id=task_id)
            trace.events.append(
                PipelineTraceEvent(
                    sequence=len(trace.events) + 1,
                    stage=stage,
                    operation=operation,
                    status=status,
                    elapsed_ms=elapsed_ms,
                    sub_question_id=sub_question_id,
                    claim_id=claim_id,
                    input_summary=input_summary or {},
                    output_summary=output_summary or {},
                    error=error,
                )
            )
            trace.updated_at = utc_now()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
            return trace

    def load(self, task_id: str) -> PipelineTrace:
        path = self._path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"research trace not found: {task_id}")
        with self._lock(task_id):
            return PipelineTrace.model_validate_json(path.read_text(encoding="utf-8"))

    def summary(self, task_id: str) -> dict[str, Any]:
        trace = self.load(task_id)
        stage_latency_ms: dict[str, int] = {}
        chunk_ids: set[str] = set()
        errors: list[dict[str, str | None]] = []
        latest_reflection: dict[str, Any] | None = None
        latest_finalization: dict[str, Any] | None = None
        for event in trace.events:
            stage_latency_ms[event.stage] = (
                stage_latency_ms.get(event.stage, 0) + event.elapsed_ms
            )
            raw_ids = event.output_summary.get("chunk_ids", [])
            if isinstance(raw_ids, list):
                chunk_ids.update(str(value) for value in raw_ids)
            if event.status == "failed":
                errors.append({"operation": event.operation, "error": event.error})
            if event.operation == "research_reflect":
                latest_reflection = dict(event.output_summary)
            if event.operation == "research_finalize":
                latest_finalization = dict(event.output_summary)
        return {
            "task_id": task_id,
            "event_count": len(trace.events),
            "tool_call_count": sum(
                event.operation.startswith("research_")
                or event.operation == "get_neighbor_evidence"
                for event in trace.events
            ),
            "total_recorded_ms": sum(event.elapsed_ms for event in trace.events),
            "stage_latency_ms": stage_latency_ms,
            "chunk_ids": sorted(chunk_ids),
            "latest_reflection": latest_reflection,
            "latest_finalization": latest_finalization,
            "errors": errors,
            "events": [event.model_dump(mode="json") for event in trace.events],
        }
