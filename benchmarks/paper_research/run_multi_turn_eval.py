from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import OrderedDict
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
from nanobot.research.evaluation import MultiTurnEvalCase, load_multi_turn_cases
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import sync_workspace_templates


def _group_cases(
    cases: list[MultiTurnEvalCase], max_dialogues: int | None
) -> list[tuple[str, list[MultiTurnEvalCase]]]:
    grouped: OrderedDict[str, list[MultiTurnEvalCase]] = OrderedDict()
    for case in cases:
        grouped.setdefault(case.dialogue_id, []).append(case)
    items = list(grouped.items())
    return items if max_dialogues is None else items[:max_dialogues]


def _find_task_ids(value: Any) -> set[str]:
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


def _is_quota_error(error: str | None) -> bool:
    normalized = (error or "").casefold()
    return any(
        marker in normalized
        for marker in (
            "out of quota",
            "account is in arrears",
            "budget configured in budget management has been exhausted",
            "budget limit is increased or reset",
        )
    )


def _response_error(answer: str, metadata: dict[str, Any]) -> str | None:
    if metadata.get("_stop_reason") == "error":
        return answer or "Agent stopped with an LLM error"
    normalized = answer.strip().casefold()
    if normalized.startswith("error:") or _is_quota_error(answer):
        return answer or "Agent stopped with an LLM error"
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_dialogues is not None and args.max_dialogues < 1:
        raise ValueError("--max-dialogues must be at least 1")

    dialogues = _group_cases(
        load_multi_turn_cases(args.benchmark), args.max_dialogues
    )
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
    completed = {
        (str(item.get("dialogue_id")), int(item.get("turn_index", 0)))
        for item in results
    }

    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(args.workspace, silent=True)
    runtime_config = _load_runtime_config(None, str(args.workspace))
    provider = make_provider(runtime_config)
    bus = MessageBus()
    cron = CronService(args.workspace / "cron" / "jobs.json")
    tools = ToolRegistry()
    mcp_provider = MCPProvider.from_config(runtime_config, tools)
    session_manager = (
        SessionManager(runtime_config.workspace_path, sessions_root=args.sessions_root)
        if args.sessions_root is not None
        else None
    )
    agent_loop = AgentLoop.from_config(
        runtime_config,
        bus,
        provider=provider,
        cron_service=cron,
        image_generation_provider_configs=image_gen_provider_configs(runtime_config),
        hook_factories=[create_file_edit_activity_hook],
        tool_registry=tools,
        session_manager=session_manager,
    )

    stop_run = False
    try:
        await mcp_provider.connect()
        required_mcp_server = "paperResearch"
        if required_mcp_server not in mcp_provider.connected_server_names:
            status = mcp_provider.runtime_status().get(required_mcp_server, "not configured")
            raise RuntimeError(
                f"Required MCP server {required_mcp_server!r} is unavailable "
                f"(status: {status}); aborting evaluation before any turn runs"
            )
        for dialogue_id, turns in dialogues:
            if stop_run:
                break
            session_key = f"eval:{run_id}:{dialogue_id}"
            for case in turns:
                key = (dialogue_id, case.turn_index)
                if key in completed:
                    continue
                progress: list[dict[str, Any]] = []
                task_ids: set[str] = set()

                async def on_progress(content: str, **kwargs: Any) -> None:
                    compact_events, discovered = _compact_tool_events(
                        kwargs.get("tool_events")
                    )
                    task_ids.update(discovered)
                    item: dict[str, Any] = {
                        "content": content,
                        "tool_hint": bool(kwargs.get("tool_hint")),
                    }
                    if compact_events:
                        item["tool_events"] = compact_events
                    progress.append(item)
                    if kwargs.get("tool_hint"):
                        print(
                            f"[{dialogue_id} turn {case.turn_index}] {content}",
                            flush=True,
                        )

                before = agent_loop.sessions.get_or_create(session_key)
                archived_before = before.last_archived
                compaction_before_turn = archived_before > 0
                started = time.perf_counter()
                answer = ""
                error = None
                response_metadata: dict[str, Any] = {}
                try:
                    response = await agent_loop.process_direct(
                        case.query,
                        session_key,
                        on_progress=on_progress,
                        ephemeral=False,
                    )
                    answer = response.content if response else ""
                    response_metadata = dict(response.metadata or {}) if response else {}
                    error = _response_error(answer, response_metadata)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                after = agent_loop.sessions.get_or_create(session_key)
                last_summary = after.metadata.get("_last_summary")
                summary_text = (
                    str(last_summary.get("text") or "")
                    if isinstance(last_summary, dict)
                    else ""
                )
                result = {
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "dialogue_id": dialogue_id,
                    "turn_index": case.turn_index,
                    "query": case.query,
                    "expected_behavior": case.expected_behavior.value,
                    "memory_probe": case.memory_probe,
                    "answer": answer,
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                    "response_metadata": response_metadata,
                    "task_ids": sorted(task_ids),
                    "progress": progress,
                    "session_message_count": len(after.messages),
                    "session_replay_message_count": len(after.get_history()),
                    "last_archived": after.last_archived,
                    "compaction_before_turn": compaction_before_turn,
                    "compaction_applied": after.last_archived > archived_before,
                    "summary_chars": len(summary_text),
                    "usage": after.metadata.get("_last_usage"),
                }
                results.append(result)
                completed.add(key)
                args.output.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "case_id": case.case_id,
                            "elapsed_ms": elapsed_ms,
                            "error": error,
                            "answer_chars": len(answer),
                            "task_count": len(task_ids),
                            "compaction_applied": result["compaction_applied"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if _is_quota_error(error):
                    stop_run = True
                    break
    finally:
        await agent_loop.aclose()
        await mcp_provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
