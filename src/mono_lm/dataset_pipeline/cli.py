from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

from .config import load_config
from .pipeline import DatasetPipeline
from .sources import load_sources
from .utils import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mono-lm dataset preparation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build dataset artifacts from a config file")
    build_parser.add_argument("--config", required=True, help="Path to a dataset TOML config")

    inventory_parser = subparsers.add_parser("inventory", help="summarize configured source inventory before a full build")
    inventory_parser.add_argument("--config", required=True, help="Path to a dataset TOML config")
    inventory_parser.add_argument("--limit", type=int, default=20, help="Maximum number of sources to print")

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
    if args.command == "inventory":
        return _run_inventory(args.config, limit=args.limit)
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


def _run_inventory(config_path: str, limit: int) -> int:
    config = load_config(config_path)
    raw_samples, rejected = load_sources(config)
    family_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"samples": 0, "chars": 0})
    source_totals: dict[str, dict[str, int | str]] = defaultdict(
        lambda: {"family": "", "samples": 0, "chars": 0}
    )

    for sample in raw_samples:
        char_count = _raw_sample_chars(sample)
        family_totals[sample.family]["samples"] += 1
        family_totals[sample.family]["chars"] += char_count
        source_totals[sample.source_name]["family"] = sample.family
        source_totals[sample.source_name]["samples"] = int(source_totals[sample.source_name]["samples"]) + 1
        source_totals[sample.source_name]["chars"] = int(source_totals[sample.source_name]["chars"]) + char_count

    print(f"Config: {config.config_path}")
    print(f"Pipeline: {config.pipeline.name}")
    print(f"Enabled sources: {len({sample.source_name for sample in raw_samples})}")
    print(f"Loaded raw samples: {len(raw_samples)}")
    print(f"Ingest rejections: {len(rejected)}")
    if config.source_catalog_paths:
        print("Source catalogs:")
        for catalog_path in config.source_catalog_paths:
            print(f"  - {catalog_path}")
    print()
    print("Families:")
    for family, stats in sorted(family_totals.items(), key=lambda item: (-item[1]["chars"], item[0])):
        print(f"  - {family}: {stats['samples']} samples / {stats['chars']} chars")
    print()
    print(f"Top sources (top {limit}):")
    ordered_sources = sorted(
        source_totals.items(),
        key=lambda item: (-int(item[1]["chars"]), item[0]),
    )
    for source_name, stats in ordered_sources[:limit]:
        print(
            f"  - {source_name} [{stats['family']}]: "
            f"{int(stats['samples'])} samples / {int(stats['chars'])} chars"
        )
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


def _raw_sample_chars(sample: dict | object) -> int:
    if isinstance(sample, dict):
        turns = sample.get("turns", [])
        turn_texts = [str(turn.get("text", "")) for turn in turns if isinstance(turn, dict)]
        values = [
            sample.get("title"),
            sample.get("text"),
            sample.get("context"),
            sample.get("question"),
            sample.get("answer"),
            *turn_texts,
        ]
        return sum(len(str(value)) for value in values if value)

    values = [
        getattr(sample, "title", None),
        getattr(sample, "text", None),
        getattr(sample, "context", None),
        getattr(sample, "question", None),
        getattr(sample, "answer", None),
        *[turn.text for turn in getattr(sample, "turns", [])],
    ]
    return sum(len(value) for value in values if value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
