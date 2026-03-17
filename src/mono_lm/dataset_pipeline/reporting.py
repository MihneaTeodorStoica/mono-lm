from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .config import BuildConfig
from .dedup import DedupSummary
from .models import ProcessedSample, RejectedSample
from .utils import ensure_dir, render_char, write_json


def build_report(
    config: BuildConfig,
    raw_samples: list,
    quality_kept: list[ProcessedSample],
    selected_samples: list[ProcessedSample],
    rejected_samples: list[RejectedSample],
    dedup_summary: DedupSummary,
    mixture_summary: dict[str, dict[str, float | int]],
    split_assignments: dict[str, list[ProcessedSample]],
) -> dict:
    split_stats = {
        split: {
            "samples": len(samples),
            "chars": sum(int(sample.quality_metrics.get("char_count", len(sample.normalized_text))) for sample in samples),
        }
        for split, samples in split_assignments.items()
    }
    char_counter = Counter()
    for split, samples in split_assignments.items():
        for index, sample in enumerate(samples):
            char_counter.update(sample.formatted_text)
            if index < len(samples) - 1:
                char_counter.update(config.formatting.document_separator)

    source_breakdown = _source_breakdown(raw_samples, quality_kept, selected_samples)
    rejection_counts = Counter((rejection.stage, rejection.reason) for rejection in rejected_samples)

    report = {
        "pipeline_name": config.pipeline.name,
        "config_path": str(config.config_path),
        "output_dir": str(config.pipeline.output_dir),
        "seed": config.pipeline.seed,
        "stage_counts": {
            "raw_loaded": len(raw_samples),
            "quality_kept": len(quality_kept),
            "final_selected": len(selected_samples),
            "rejected_total": len(rejected_samples),
        },
        "dedup": {
            "exact_clusters": dedup_summary.exact_clusters,
            "exact_removed": dedup_summary.exact_removed,
            "near_clusters": dedup_summary.near_clusters,
            "near_removed": dedup_summary.near_removed,
        },
        "mixture": mixture_summary,
        "splits": split_stats,
        "rejections": {
            f"{stage}:{reason}": count
            for (stage, reason), count in sorted(rejection_counts.items())
        },
        "sources": source_breakdown,
        "character_inventory": {
            "unique_characters": len(char_counter),
            "total_characters": sum(char_counter.values()),
            "top_characters": [
                {
                    "char": render_char(char),
                    "codepoint": f"U+{ord(char):04X}",
                    "count": count,
                    "ratio": round(count / max(1, sum(char_counter.values())), 6),
                }
                for char, count in char_counter.most_common(40)
            ],
        },
    }
    return report


def write_character_inventory(path: Path, text: str) -> None:
    counter = Counter(text)
    total = sum(counter.values()) or 1
    lines = ["char\tcodepoint\tcount\tratio"]
    for char, count in counter.most_common():
        lines.append(f"{render_char(char)}\tU+{ord(char):04X}\t{count}\t{count / total:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# mono-lm dataset report",
        "",
        "## Overview",
        "",
        f"- Pipeline: `{report['pipeline_name']}`",
        f"- Config: `{report['config_path']}`",
        f"- Output: `{report['output_dir']}`",
        f"- Seed: `{report['seed']}`",
        "",
        "## Stage counts",
        "",
    ]
    for key, value in report["stage_counts"].items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    lines.extend(
        [
            "",
            "## Deduplication",
            "",
            f"- Exact clusters: {report['dedup']['exact_clusters']}",
            f"- Exact removed: {report['dedup']['exact_removed']}",
            f"- Near clusters: {report['dedup']['near_clusters']}",
            f"- Near removed: {report['dedup']['near_removed']}",
            "",
            "## Mixture",
            "",
        ]
    )
    for family, stats in sorted(report["mixture"].items()):
        lines.append(
            "- "
            f"{family}: selected {stats['selected_samples']} samples / {stats['selected_chars']} chars "
            f"(available {stats['available_samples']} / {stats['available_chars']} chars, target {stats['target_chars']})"
        )
    lines.extend(["", "## Splits", ""])
    for split, stats in report["splits"].items():
        lines.append(f"- {split}: {stats['samples']} samples / {stats['chars']} chars")
    lines.extend(["", "## Rejections", ""])
    if report["rejections"]:
        for key, value in report["rejections"].items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- None")
    lines.extend(["", "## Character inventory", ""])
    for item in report["character_inventory"]["top_characters"][:20]:
        lines.append(f"- {item['char']} ({item['codepoint']}): {item['count']} [{item['ratio']:.4f}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inspection_markdown(
    path: Path,
    selected_samples: list[ProcessedSample],
    rejected_samples: list[RejectedSample],
    per_family: int,
    preview_chars: int,
) -> None:
    by_family: dict[str, list[ProcessedSample]] = defaultdict(list)
    for sample in selected_samples:
        by_family[sample.family].append(sample)

    lines = [
        "# Inspection deck",
        "",
        "Representative cleaned samples from the selected corpus.",
        "",
    ]
    for family in sorted(by_family):
        lines.extend([f"## {family}", ""])
        for sample in by_family[family][:per_family]:
            preview = sample.formatted_text[:preview_chars].rstrip()
            lines.extend(
                [
                    f"### {sample.sample_id}",
                    "",
                    f"- Source: `{sample.source_name}`",
                    f"- Cluster: `{sample.duplicate_cluster}`",
                    f"- Quality score: {sample.quality_score:.4f}",
                    "",
                    "```text",
                    preview,
                    "```",
                    "",
                ]
            )

    lines.extend(["## Rejected samples", ""])
    for rejection in rejected_samples[: max(5, per_family * 2)]:
        lines.extend(
            [
                f"- `{rejection.sample_id}` [{rejection.stage}/{rejection.reason}]: {rejection.detail}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict) -> None:
    write_json(path, payload)


def _source_breakdown(
    raw_samples: list,
    quality_kept: list[ProcessedSample],
    selected_samples: list[ProcessedSample],
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"loaded": 0, "after_quality": 0, "selected": 0})
    for sample in raw_samples:
        stats[sample.source_name]["loaded"] += 1
    for sample in quality_kept:
        stats[sample.source_name]["after_quality"] += 1
    for sample in selected_samples:
        stats[sample.source_name]["selected"] += 1
    return dict(stats)
