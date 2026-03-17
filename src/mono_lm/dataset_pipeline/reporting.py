from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .config import BuildConfig
from .dedup import DedupSummary
from .models import ProcessedSample, RawSample, RejectedSample
from .utils import render_char, write_json


def build_report(
    config: BuildConfig,
    raw_samples: list[RawSample],
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
            "chars": sum(_sample_chars(sample) for sample in samples),
        }
        for split, samples in split_assignments.items()
    }
    char_counter = Counter()
    for split, samples in split_assignments.items():
        for index, sample in enumerate(samples):
            char_counter.update(sample.formatted_text)
            if index < len(samples) - 1:
                char_counter.update(config.formatting.document_separator)

    selected_total_chars = sum(_sample_chars(sample) for sample in selected_samples)
    family_breakdown = _family_breakdown(selected_samples, mixture_summary, selected_total_chars)
    source_breakdown = _source_breakdown(raw_samples, quality_kept, selected_samples, selected_total_chars)
    rejection_counts = Counter((rejection.stage, rejection.reason) for rejection in rejected_samples)

    report = {
        "pipeline_name": config.pipeline.name,
        "config_path": str(config.config_path),
        "source_catalogs": [str(path) for path in config.source_catalog_paths],
        "output_dir": str(config.pipeline.output_dir),
        "seed": config.pipeline.seed,
        "stage_counts": {
            "raw_loaded_samples": len(raw_samples),
            "quality_kept_samples": len(quality_kept),
            "final_selected_samples": len(selected_samples),
            "rejected_total_samples": len(rejected_samples),
        },
        "stage_characters": {
            "quality_kept_chars": sum(_sample_chars(sample) for sample in quality_kept),
            "selected_chars": selected_total_chars,
        },
        "dedup": {
            "exact_clusters": dedup_summary.exact_clusters,
            "exact_removed": dedup_summary.exact_removed,
            "near_clusters": dedup_summary.near_clusters,
            "near_removed": dedup_summary.near_removed,
        },
        "mixture": mixture_summary,
        "families": family_breakdown,
        "sources": source_breakdown,
        "dominance": {
            "threshold": config.reporting.dominance_threshold,
            "families": _dominant_entries(family_breakdown, config.reporting.dominance_threshold),
            "sources": _dominant_entries(source_breakdown, config.reporting.dominance_threshold),
        },
        "splits": split_stats,
        "rejections": {
            f"{stage}:{reason}": count
            for (stage, reason), count in sorted(rejection_counts.items())
        },
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


def write_breakdown_tsv(path: Path, rows: Iterable[dict[str, object]], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        parts = [str(row.get(column, "")) for column in columns]
        lines.append("\t".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: dict, top_sources: int) -> None:
    lines = [
        "# mono-lm dataset report",
        "",
        "## Overview",
        "",
        f"- Pipeline: `{report['pipeline_name']}`",
        f"- Config: `{report['config_path']}`",
        f"- Output: `{report['output_dir']}`",
        f"- Seed: `{report['seed']}`",
    ]
    if report["source_catalogs"]:
        lines.append(f"- Source catalogs: {', '.join(f'`{item}`' for item in report['source_catalogs'])}")

    lines.extend(
        [
            "",
            "## Stage counts",
            "",
        ]
    )
    for key, value in report["stage_counts"].items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    for key, value in report["stage_characters"].items():
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
            "## Family composition",
            "",
        ]
    )
    for family, stats in sorted(report["families"].items(), key=lambda item: (-int(item[1]["selected_chars"]), item[0])):
        lines.append(
            "- "
            f"{family}: {stats['selected_samples']} samples / {stats['selected_chars']} chars "
            f"({float(stats['selected_share']) * 100:.2f}% of final text, target {stats['target_chars']})"
        )

    lines.extend(["", f"## Top sources (top {top_sources})", ""])
    source_items = sorted(
        report["sources"].items(),
        key=lambda item: (-int(item[1]["selected_chars"]), item[0]),
    )
    for source_name, stats in source_items[:top_sources]:
        lines.append(
            "- "
            f"{source_name} [{stats['family']}]: {stats['selected_samples']} samples / {stats['selected_chars']} chars "
            f"({float(stats['selected_share']) * 100:.2f}% of final text)"
        )

    lines.extend(["", "## Dominance alerts", ""])
    if report["dominance"]["families"] or report["dominance"]["sources"]:
        for entry in report["dominance"]["families"]:
            lines.append(
                f"- Family dominance: {entry['name']} at {float(entry['selected_share']) * 100:.2f}% "
                f"(threshold {float(report['dominance']['threshold']) * 100:.2f}%)"
            )
        for entry in report["dominance"]["sources"]:
            lines.append(
                f"- Source dominance: {entry['name']} at {float(entry['selected_share']) * 100:.2f}% "
                f"(threshold {float(report['dominance']['threshold']) * 100:.2f}%)"
            )
    else:
        lines.append("- None")

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
    preview_sources_per_family: int,
    preview_samples_per_source: int,
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
        family_samples = sorted(
            by_family[family],
            key=lambda sample: (-_sample_chars(sample), sample.source_name, sample.sample_id),
        )
        lines.extend([f"## {family}", ""])
        for sample in family_samples[:per_family]:
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

        lines.extend(["### Source spotlights", ""])
        source_groups: dict[str, list[ProcessedSample]] = defaultdict(list)
        for sample in family_samples:
            source_groups[sample.source_name].append(sample)
        ordered_sources = sorted(
            source_groups.items(),
            key=lambda item: (
                -sum(_sample_chars(sample) for sample in item[1]),
                item[0],
            ),
        )
        for source_name, source_samples in ordered_sources[:preview_sources_per_family]:
            lines.append(f"#### {source_name}")
            lines.append("")
            for sample in source_samples[:preview_samples_per_source]:
                lines.extend(
                    [
                        f"- Selected chars: {_sample_chars(sample)}",
                        "```text",
                        sample.formatted_text[:preview_chars].rstrip(),
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


def _sample_chars(sample: ProcessedSample) -> int:
    return int(sample.quality_metrics.get("char_count", len(sample.normalized_text)))


def _family_breakdown(
    selected_samples: list[ProcessedSample],
    mixture_summary: dict[str, dict[str, float | int]],
    selected_total_chars: int,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    selected_counts = Counter(sample.family for sample in selected_samples)
    selected_chars = Counter()
    for sample in selected_samples:
        selected_chars[sample.family] += _sample_chars(sample)

    for family, mixture_stats in mixture_summary.items():
        chars = int(selected_chars.get(family, 0))
        stats[family] = {
            "available_samples": int(mixture_stats["available_samples"]),
            "available_chars": int(mixture_stats["available_chars"]),
            "selected_samples": int(selected_counts.get(family, 0)),
            "selected_chars": chars,
            "target_chars": int(mixture_stats["target_chars"]),
            "selected_share": round(chars / max(1, selected_total_chars), 6),
        }
    return stats


def _source_breakdown(
    raw_samples: list[RawSample],
    quality_kept: list[ProcessedSample],
    selected_samples: list[ProcessedSample],
    selected_total_chars: int,
) -> dict[str, dict[str, float | int | str]]:
    stats: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {
            "family": "",
            "loaded_samples": 0,
            "loaded_chars": 0,
            "after_quality_samples": 0,
            "after_quality_chars": 0,
            "selected_samples": 0,
            "selected_chars": 0,
            "selected_share": 0.0,
        }
    )
    for sample in raw_samples:
        row = stats[sample.source_name]
        row["family"] = sample.family
        row["loaded_samples"] = int(row["loaded_samples"]) + 1
        row["loaded_chars"] = int(row["loaded_chars"]) + _raw_sample_chars(sample)
    for sample in quality_kept:
        row = stats[sample.source_name]
        row["family"] = sample.family
        row["after_quality_samples"] = int(row["after_quality_samples"]) + 1
        row["after_quality_chars"] = int(row["after_quality_chars"]) + _sample_chars(sample)
    for sample in selected_samples:
        row = stats[sample.source_name]
        row["family"] = sample.family
        row["selected_samples"] = int(row["selected_samples"]) + 1
        row["selected_chars"] = int(row["selected_chars"]) + _sample_chars(sample)

    for row in stats.values():
        row["selected_share"] = round(int(row["selected_chars"]) / max(1, selected_total_chars), 6)
    return dict(stats)


def _dominant_entries(
    rows: dict[str, dict[str, float | int | str]],
    threshold: float,
) -> list[dict[str, float | int | str]]:
    entries = []
    for name, stats in rows.items():
        share = float(stats.get("selected_share", 0.0))
        if share >= threshold:
            entries.append(
                {
                    "name": name,
                    "selected_chars": int(stats.get("selected_chars", 0)),
                    "selected_share": round(share, 6),
                }
            )
    return sorted(entries, key=lambda item: (-int(item["selected_chars"]), str(item["name"])))


def _raw_sample_chars(sample: RawSample) -> int:
    values = [
        sample.title,
        sample.text,
        sample.context,
        sample.question,
        sample.answer,
        *[turn.text for turn in sample.turns],
    ]
    return sum(len(value) for value in values if value)
