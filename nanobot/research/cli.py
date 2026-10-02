"""Command-line entry point for indexing a local paper corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from nanobot.research.config import ResearchConfig
from nanobot.research.corpus.indexer import CorpusIndexer
from nanobot.research.evaluation import (
    evaluate_single_turn_retrieval,
    load_single_turn_cases,
    load_single_turn_predictions,
    score_single_turn_outputs,
)
from nanobot.research.observability.trace import PipelineTraceStore
from nanobot.research.retrieval.service import HybridRetrievalService

app = typer.Typer(help="Build and inspect nanobot's scientific-paper research index.")


@app.command("index")
def index_corpus(
    corpus_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    data_dir: Path | None = typer.Option(None, help="Directory for indexes and research state."),
    skip_dense: bool = typer.Option(
        False,
        help="Build only BM25 indexes. Intended for quick smoke tests.",
    ),
) -> None:
    config = ResearchConfig(data_dir=data_dir) if data_dir is not None else ResearchConfig()
    report = CorpusIndexer(config).index_directory(corpus_dir, build_dense=not skip_dense)
    typer.echo(report.model_dump_json(indent=2))


@app.command("status")
def index_status(
    data_dir: Path | None = typer.Option(None, help="Directory containing the index."),
) -> None:
    config = ResearchConfig(data_dir=data_dir) if data_dir is not None else ResearchConfig()
    manifest = config.data_dir / "index_manifest.json"
    if not manifest.exists():
        raise typer.BadParameter(f"index manifest not found: {manifest}")
    typer.echo(manifest.read_text(encoding="utf-8"))


@app.command("trace")
def task_trace(
    task_id: str = typer.Argument(..., help="Research task ID."),
    data_dir: Path | None = typer.Option(None, help="Directory containing research state."),
) -> None:
    config = ResearchConfig(data_dir=data_dir) if data_dir is not None else ResearchConfig()
    summary = PipelineTraceStore(config.traces_dir).summary(task_id)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command("eval-single")
def evaluate_single_turn(
    benchmark: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    data_dir: Path | None = typer.Option(None, help="Directory containing the index."),
    output: Path | None = typer.Option(None, help="Optional JSON report path."),
    top_k: int = typer.Option(4, min=1, max=100),
    timeout_seconds: int = typer.Option(240, min=1),
    hf_home: Path | None = typer.Option(
        None,
        help="Optional Hugging Face model-cache directory.",
    ),
    offline: bool = typer.Option(
        True,
        "--offline/--online",
        help="Use local model files without Hugging Face network checks.",
    ),
) -> None:
    if hf_home is not None:
        os.environ["HF_HOME"] = str(hf_home)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = ResearchConfig(data_dir=data_dir) if data_dir is not None else ResearchConfig()
    service = HybridRetrievalService(config)
    service.prepare_indexes()
    service.prepare_models()
    report = evaluate_single_turn_retrieval(
        service,
        load_single_turn_cases(benchmark),
        top_k=top_k,
        timeout_ms=timeout_seconds * 1000,
        on_case_complete=lambda index, total, result: typer.echo(
            f"[{index}/{total}] {result.case_id} {result.retrieval_ms}ms"
            + (f" error={result.error}" if result.error else ""),
            err=True,
        ),
    )
    payload = report.model_dump_json(indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload)


@app.command("score-single")
def score_single_turn(
    benchmark: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    predictions: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, help="Optional JSON report path."),
) -> None:
    report = score_single_turn_outputs(
        load_single_turn_cases(benchmark),
        load_single_turn_predictions(predictions),
    )
    payload = report.model_dump_json(indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
