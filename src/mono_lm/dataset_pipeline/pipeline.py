from __future__ import annotations

import shutil

from .cleaning import normalize_raw_sample
from .config import BuildConfig
from .dedup import DedupSummary, deduplicate_samples
from .filters import assess_sample
from .mixing import select_mixture
from .models import PipelineOutputs, RejectedSample
from .reporting import (
    build_report,
    write_breakdown_tsv,
    write_character_inventory,
    write_inspection_markdown,
    write_manifest,
    write_markdown_report,
)
from .sources import load_sources
from .splitting import assign_splits
from .utils import ensure_dir, preview_text, write_json, write_jsonl


class DatasetPipeline:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def build(self) -> PipelineOutputs:
        output_dir = ensure_dir(self.config.pipeline.output_dir)
        intermediate_dir = ensure_dir(output_dir / "intermediate")
        splits_dir = ensure_dir(output_dir / "splits")
        final_dir = ensure_dir(output_dir / "final")
        reports_dir = ensure_dir(output_dir / "reports")

        raw_samples, ingest_rejections = load_sources(self.config)
        normalized_samples = []
        rejected_samples = list(ingest_rejections)
        for raw_sample in raw_samples:
            processed = normalize_raw_sample(raw_sample, self.config.cleaning, self.config.formatting)
            outcome = assess_sample(processed, self.config.quality)
            processed.quality_metrics = outcome.metrics
            processed.quality_score = outcome.quality_score
            if not outcome.accepted:
                rejected_samples.append(
                    RejectedSample(
                        sample_id=processed.sample_id,
                        source_name=processed.source_name,
                        family=processed.family,
                        stage="quality",
                        reason=outcome.reason or "quality_reject",
                        detail=outcome.detail,
                        origin_path=processed.origin_path,
                        preview=preview_text(processed.formatted_text or processed.normalized_text),
                        metadata={"metrics": outcome.metrics},
                    )
                )
                continue
            normalized_samples.append(processed)

        deduped_samples, dedup_rejections, dedup_summary = deduplicate_samples(normalized_samples, self.config.dedup)
        rejected_samples.extend(dedup_rejections)

        selected_samples, mixture_summary = select_mixture(deduped_samples, self.config.mixture, self.config.pipeline)
        for sample in selected_samples:
            sample.selected_for_final = True

        split_assignments = assign_splits(selected_samples, self.config.split, self.config.pipeline.seed)
        final_texts = {
            split: self.config.formatting.document_separator.join(sample.formatted_text for sample in samples).strip() + "\n"
            if samples
            else ""
            for split, samples in split_assignments.items()
        }

        write_jsonl(intermediate_dir / "raw_samples.jsonl", (sample.to_dict() for sample in raw_samples))
        write_jsonl(intermediate_dir / "normalized_samples.jsonl", (sample.to_dict() for sample in normalized_samples))
        write_jsonl(intermediate_dir / "deduped_samples.jsonl", (sample.to_dict() for sample in deduped_samples))
        write_jsonl(intermediate_dir / "selected_samples.jsonl", (sample.to_dict() for sample in selected_samples))
        write_jsonl(intermediate_dir / "rejected_samples.jsonl", (sample.to_dict() for sample in rejected_samples))
        for split, samples in split_assignments.items():
            write_jsonl(splits_dir / f"{split}.jsonl", (sample.to_dict() for sample in samples))
            (final_dir / f"{split}.txt").write_text(final_texts[split], encoding="utf-8")

        report = build_report(
            config=self.config,
            raw_samples=raw_samples,
            quality_kept=normalized_samples,
            selected_samples=selected_samples,
            rejected_samples=rejected_samples,
            dedup_summary=dedup_summary,
            mixture_summary=mixture_summary,
            split_assignments=split_assignments,
        )
        write_json(reports_dir / "report.json", report)
        write_markdown_report(reports_dir / "report.md", report, top_sources=self.config.reporting.top_sources)
        write_character_inventory(reports_dir / "character_inventory.tsv", "".join(final_texts.values()))
        write_breakdown_tsv(
            reports_dir / "family_breakdown.tsv",
            (
                {"family": family, **stats}
                for family, stats in sorted(
                    report["families"].items(),
                    key=lambda item: (-int(item[1]["selected_chars"]), item[0]),
                )
            ),
            columns=[
                "family",
                "available_samples",
                "available_chars",
                "selected_samples",
                "selected_chars",
                "target_chars",
                "selected_share",
            ],
        )
        write_breakdown_tsv(
            reports_dir / "source_breakdown.tsv",
            (
                {"source_name": source_name, **stats}
                for source_name, stats in sorted(
                    report["sources"].items(),
                    key=lambda item: (-int(item[1]["selected_chars"]), item[0]),
                )
            ),
            columns=[
                "source_name",
                "family",
                "loaded_samples",
                "loaded_chars",
                "after_quality_samples",
                "after_quality_chars",
                "selected_samples",
                "selected_chars",
                "selected_share",
            ],
        )
        write_inspection_markdown(
            reports_dir / "inspection.md",
            selected_samples=selected_samples,
            rejected_samples=rejected_samples,
            per_family=self.config.pipeline.inspection_samples_per_family,
            preview_chars=self.config.pipeline.preview_chars,
            preview_sources_per_family=self.config.reporting.preview_sources_per_family,
            preview_samples_per_source=self.config.reporting.preview_samples_per_source,
        )

        config_snapshot = output_dir / "config.used.toml"
        shutil.copy2(self.config.config_path, config_snapshot)
        manifest = {
            "pipeline": self.config.pipeline.name,
            "config_snapshot": str(config_snapshot),
            "source_catalogs": [str(path) for path in self.config.source_catalog_paths],
            "reports": {
                "json": str(reports_dir / "report.json"),
                "markdown": str(reports_dir / "report.md"),
                "character_inventory": str(reports_dir / "character_inventory.tsv"),
                "family_breakdown": str(reports_dir / "family_breakdown.tsv"),
                "source_breakdown": str(reports_dir / "source_breakdown.tsv"),
                "inspection": str(reports_dir / "inspection.md"),
            },
            "intermediate": {
                "raw": str(intermediate_dir / "raw_samples.jsonl"),
                "normalized": str(intermediate_dir / "normalized_samples.jsonl"),
                "deduped": str(intermediate_dir / "deduped_samples.jsonl"),
                "selected": str(intermediate_dir / "selected_samples.jsonl"),
                "rejected": str(intermediate_dir / "rejected_samples.jsonl"),
            },
            "splits": {
                split: {
                    "jsonl": str(splits_dir / f"{split}.jsonl"),
                    "text": str(final_dir / f"{split}.txt"),
                }
                for split in split_assignments
            },
        }
        manifest_path = output_dir / "manifest.json"
        write_manifest(manifest_path, manifest)
        return PipelineOutputs(
            output_dir=output_dir,
            manifest_path=manifest_path,
            report_json_path=reports_dir / "report.json",
            report_markdown_path=reports_dir / "report.md",
            train_path=final_dir / "train.txt",
            validation_path=final_dir / "validation.txt",
            test_path=final_dir / "test.txt",
        )
