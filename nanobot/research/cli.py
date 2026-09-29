"""Command-line entry point for indexing a local paper corpus."""

from __future__ import annotations

from pathlib import Path

import typer

from nanobot.research.config import ResearchConfig
from nanobot.research.corpus.indexer import CorpusIndexer

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
