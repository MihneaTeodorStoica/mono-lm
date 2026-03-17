from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config
from .pipeline import DatasetPipeline
from .utils import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mono-lm dataset preparation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build dataset artifacts from a config file")
    build_parser.add_argument("--config", required=True, help="Path to a dataset TOML config")

    inspect_parser = subparsers.add_parser("inspect", help="print sample previews from a pipeline artifact directory")
    inspect_parser.add_argument("--artifact-dir", required=True, help="Path to a built artifact directory")
    inspect_parser.add_argument(
        "--stage",
        default="selected",
        choices=["raw", "normalized", "deduped", "selected", "rejected", "train", "validation", "test"],
        help="Which artifact file to inspect",
    )
    inspect_parser.add_argument("--family", help="Optional family filter")
    inspect_parser.add_argument("--source", help="Optional source filter")
    inspect_parser.add_argument("--limit", type=int, default=3, help="Number of samples to print")
    inspect_parser.add_argument("--preview-chars", type=int, default=500, help="Preview length per sample")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return _run_build(args.config)
    if args.command == "inspect":
        return _run_inspect(
            artifact_dir=Path(args.artifact_dir),
            stage=args.stage,
            family=args.family,
            source=args.source,
            limit=args.limit,
            preview_chars=args.preview_chars,
        )
    return 1


def _run_build(config_path: str) -> int:
    config = load_config(config_path)
    outputs = DatasetPipeline(config).build()
    print(f"Built dataset artifacts in {outputs.output_dir}")
    print(f"Manifest: {outputs.manifest_path}")
    print(f"Report: {outputs.report_markdown_path}")
    print(f"Train text: {outputs.train_path}")
    print(f"Validation text: {outputs.validation_path}")
    print(f"Test text: {outputs.test_path}")
    return 0


def _run_inspect(
    artifact_dir: Path,
    stage: str,
    family: str | None,
    source: str | None,
    limit: int,
    preview_chars: int,
) -> int:
    stage_map = {
        "raw": artifact_dir / "intermediate" / "raw_samples.jsonl",
        "normalized": artifact_dir / "intermediate" / "normalized_samples.jsonl",
        "deduped": artifact_dir / "intermediate" / "deduped_samples.jsonl",
        "selected": artifact_dir / "intermediate" / "selected_samples.jsonl",
        "rejected": artifact_dir / "intermediate" / "rejected_samples.jsonl",
        "train": artifact_dir / "splits" / "train.jsonl",
        "validation": artifact_dir / "splits" / "validation.jsonl",
        "test": artifact_dir / "splits" / "test.jsonl",
    }
    path = stage_map[stage]
    rows = read_jsonl(path)
    if family:
        rows = [row for row in rows if row.get("family") == family]
    if source:
        rows = [row for row in rows if row.get("source_name") == source]
    for index, row in enumerate(rows[:limit], start=1):
        print(f"--- sample {index}: {row.get('sample_id', '<unknown>')} ---")
        print(f"family={row.get('family')} source={row.get('source_name')}")
        if stage == "rejected":
            print(f"stage={row.get('stage')} reason={row.get('reason')} detail={row.get('detail')}")
            preview = row.get("preview", "")
        else:
            preview = row.get("formatted_text") or row.get("normalized_text") or row.get("text") or ""
        print(preview[:preview_chars].rstrip())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
