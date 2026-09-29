"""Command-line entry point for indexing a local paper corpus."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from nanobot.research.config import ResearchConfig
from nanobot.research.corpus.indexer import CorpusIndexer
from nanobot.research.evaluation import evaluate_retrieval, load_retrieval_cases
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


@app.command("eval")
def evaluate_index(
    benchmark: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    data_dir: Path | None = typer.Option(None, help="Directory containing the index."),
    output: Path | None = typer.Option(None, help="Optional JSON report path."),
    paper_top_k: int = typer.Option(10, min=1, max=100),
    evidence_top_k: int = typer.Option(8, min=1, max=100),
) -> None:
    config = ResearchConfig(data_dir=data_dir) if data_dir is not None else ResearchConfig()
    service = HybridRetrievalService(config)
    service.prepare_indexes()
    service.prepare_models()
    report = evaluate_retrieval(
        service,
        load_retrieval_cases(benchmark),
        paper_top_k=paper_top_k,
        evidence_top_k=evidence_top_k,
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
