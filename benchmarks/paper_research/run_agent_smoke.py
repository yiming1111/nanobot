from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.cli.runtime_config import _load_runtime_config
from nanobot.cron.service import CronService
from nanobot.providers.factory import make_provider
from nanobot.providers.image_generation import image_gen_provider_configs
from nanobot.utils.helpers import sync_workspace_templates


CASE_IDS = [
    "st-001-vehicular-game",
    "st-013-consensus-information",
    "st-037-colocated-resources",
    "st-041-partial-vehicular-field",
    "st-051-abstain-quantum",
]


def is_quota_error(error: str | None) -> bool:
    if error is None:
        return False
    normalized = error.casefold()
    return "out of quota" in normalized or "account is in arrears" in normalized


def load_cases(path: Path, *, all_cases: bool = False) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if all_cases:
        return cases
    wanted = set(CASE_IDS)
    selected = {case["case_id"]: case for case in cases if case["case_id"] in wanted}
    return [selected[case_id] for case_id in CASE_IDS]


def _find_task_ids(value: Any) -> set[str]:
    """Extract research task IDs without retaining large MCP result payloads."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "task_id" and isinstance(item, str) and item:
                found.add(item)
            else:
                found.update(_find_task_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_task_ids(item))
    elif isinstance(value, str):
        found.update(
            re.findall(r'["\']task_id["\']\s*:\s*["\']([0-9a-f]{32})["\']', value)
        )
    return found


def _compact_tool_events(events: Any) -> tuple[list[dict[str, Any]], set[str]]:
    compact: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        task_ids.update(_find_task_ids(event.get("arguments")))
        task_ids.update(_find_task_ids(event.get("result")))
        compact.append(
            {
                "phase": event.get("phase"),
                "name": event.get("name"),
                "arguments": event.get("arguments") or {},
                "error": event.get("error"),
            }
        )
    return compact, task_ids


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--all", dest="all_cases", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if not isinstance(saved, list):
            raise ValueError("resume output must contain a JSON array")
        results = saved
    saved_run_ids = {str(item.get("run_id")) for item in results if item.get("run_id")}
    if len(saved_run_ids) > 1:
        raise ValueError("resume output contains multiple run IDs")
    saved_run_id = next(iter(saved_run_ids), None)
    if args.run_id and saved_run_id and args.run_id != saved_run_id:
        raise ValueError("--run-id does not match the existing resume output")
    run_id = (
        args.run_id
        or saved_run_id
        or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )

    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(args.workspace, silent=True)
    runtime_config = _load_runtime_config(None, str(args.workspace))
    provider = make_provider(runtime_config)
    bus = MessageBus()
    cron = CronService(args.workspace / "cron" / "jobs.json")
    tools = ToolRegistry()
    mcp_provider = MCPProvider.from_config(runtime_config, tools)
    agent_loop = AgentLoop.from_config(
        runtime_config,
        bus,
        provider=provider,
        cron_service=cron,
        image_generation_provider_configs=image_gen_provider_configs(runtime_config),
        hook_factories=[create_file_edit_activity_hook],
        tool_registry=tools,
    )

    completed_case_ids = {str(item.get("case_id")) for item in results}
    try:
        await mcp_provider.connect()
        for case in load_cases(args.benchmark, all_cases=args.all_cases):
            case_id = case["case_id"]
            if case_id in completed_case_ids:
                continue
            progress: list[dict[str, Any]] = []
            task_ids: set[str] = set()

            async def on_progress(content: str, **kwargs: Any) -> None:
                compact_events, discovered_task_ids = _compact_tool_events(
                    kwargs.get("tool_events")
                )
                task_ids.update(discovered_task_ids)
                item: dict[str, Any] = {
                    "content": content,
                    "tool_hint": bool(kwargs.get("tool_hint")),
                }
                if compact_events:
                    item["tool_events"] = compact_events
                progress.append(item)
                if kwargs.get("tool_hint"):
                    print(f"[{case_id}] {content}", flush=True)

            started = time.perf_counter()
            error = None
            answer = ""
            response_metadata: dict[str, Any] = {}
            try:
                response = await agent_loop.process_direct(
                    case["query"],
                    f"eval:{run_id}:{case_id}",
                    on_progress=on_progress,
                    ephemeral=True,
                )
                answer = response.content if response else ""
                response_metadata = dict(response.metadata or {}) if response else {}
                if response_metadata.get("_stop_reason") == "error":
                    error = answer or "Agent stopped with an LLM error"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            result = {
                "run_id": run_id,
                "case_id": case_id,
                "query": case["query"],
                "expected_behavior": case["expected_behavior"],
                "answer": answer,
                "elapsed_ms": elapsed_ms,
                "error": error,
                "response_metadata": response_metadata,
                "task_ids": sorted(task_ids),
                "progress": progress,
            }
            results.append(result)
            completed_case_ids.add(case_id)
            args.output.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "elapsed_ms": elapsed_ms,
                        "error": error,
                        "answer_chars": len(answer),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if is_quota_error(error):
                print(
                    f"[{case_id}] provider quota exhausted; stopping this run",
                    flush=True,
                )
                break
    finally:
        await agent_loop.aclose()
        await mcp_provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
