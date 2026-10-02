from __future__ import annotations

import argparse
import asyncio
import json
import time
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


def load_cases(path: Path) -> list[dict[str, Any]]:
    wanted = set(CASE_IDS)
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {case["case_id"]: case for case in cases if case["case_id"] in wanted}
    return [selected[case_id] for case_id in CASE_IDS]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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

    results: list[dict[str, Any]] = []
    try:
        await mcp_provider.connect()
        for case in load_cases(args.benchmark):
            case_id = case["case_id"]
            progress: list[dict[str, Any]] = []

            async def on_progress(content: str, **kwargs: Any) -> None:
                progress.append({"content": content, **kwargs})
                if kwargs.get("tool_hint"):
                    print(f"[{case_id}] {content}", flush=True)

            started = time.perf_counter()
            error = None
            answer = ""
            response_metadata: dict[str, Any] = {}
            try:
                response = await agent_loop.process_direct(
                    case["query"],
                    f"eval:{case_id}",
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
                "case_id": case_id,
                "query": case["query"],
                "expected_behavior": case["expected_behavior"],
                "answer": answer,
                "elapsed_ms": elapsed_ms,
                "error": error,
                "response_metadata": response_metadata,
                "progress": progress,
            }
            results.append(result)
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
    finally:
        await agent_loop.aclose()
        await mcp_provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
