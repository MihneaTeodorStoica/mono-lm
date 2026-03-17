from __future__ import annotations

from collections import defaultdict
import math
import random

from .config import MixtureConfig, PipelineConfig
from .models import ProcessedSample


def select_mixture(
    samples: list[ProcessedSample],
    mixture: MixtureConfig,
    pipeline: PipelineConfig,
) -> tuple[list[ProcessedSample], dict[str, dict[str, float | int]]]:
    family_groups: dict[str, list[ProcessedSample]] = defaultdict(list)
    for sample in samples:
        family_groups[sample.family].append(sample)

    ordered = {
        family: _weighted_order(
            family_samples,
            mixture.source_weights,
            seed=pipeline.seed + index,
        )
        for index, (family, family_samples) in enumerate(sorted(family_groups.items()))
    }

    if pipeline.target_total_chars is None:
        selected = [sample for family in sorted(ordered) for sample in ordered[family]]
        return selected, _summarize_mixture(selected, ordered, {})

    quotas = _initial_family_quotas(family_groups, mixture, pipeline.target_total_chars)
    selected: list[ProcessedSample] = []
    selected_ids: set[str] = set()
    remaining_budget = pipeline.target_total_chars
    family_targets = quotas.copy()

    changed = True
    while changed and remaining_budget > 0:
        changed = False
        for family in sorted(ordered):
            family_samples = [sample for sample in ordered[family] if sample.sample_id not in selected_ids]
            if not family_samples:
                continue
            target_chars = family_targets.get(family, 0)
            current_chars = sum(
                int(sample.quality_metrics.get("char_count", len(sample.normalized_text)))
                for sample in selected
                if sample.family == family
            )
            minimum_needed = mixture.minimum_samples_per_family if current_chars == 0 else 0
            while family_samples and remaining_budget > 0:
                next_sample = family_samples[0]
                char_count = int(next_sample.quality_metrics.get("char_count", len(next_sample.normalized_text)))
                if current_chars >= target_chars and minimum_needed <= 0:
                    break
                if char_count > remaining_budget and selected:
                    break
                family_samples.pop(0)
                selected.append(next_sample)
                selected_ids.add(next_sample.sample_id)
                current_chars += char_count
                remaining_budget -= char_count
                minimum_needed = max(0, minimum_needed - 1)
                changed = True
                if current_chars >= target_chars and minimum_needed <= 0:
                    break

        if remaining_budget > 0 and changed:
            unselected = [sample for family in sorted(ordered) for sample in ordered[family] if sample.sample_id not in selected_ids]
            if not unselected:
                break
            for sample in unselected:
                char_count = int(sample.quality_metrics.get("char_count", len(sample.normalized_text)))
                if char_count > remaining_budget and selected:
                    continue
                selected.append(sample)
                selected_ids.add(sample.sample_id)
                remaining_budget -= char_count
                changed = True
                if remaining_budget <= 0:
                    break

    selected.sort(key=lambda sample: (sample.family, sample.source_name, sample.sample_id))
    return selected, _summarize_mixture(selected, ordered, quotas)


def _initial_family_quotas(
    family_groups: dict[str, list[ProcessedSample]],
    mixture: MixtureConfig,
    target_total_chars: int,
) -> dict[str, int]:
    available_chars = {
        family: sum(int(sample.quality_metrics.get("char_count", len(sample.normalized_text))) for sample in samples)
        for family, samples in family_groups.items()
    }
    active_weights = {
        family: mixture.family_weights.get(family, 1.0)
        for family, chars in available_chars.items()
        if chars > 0
    }
    weight_sum = sum(active_weights.values()) or 1.0
    quotas: dict[str, int] = {}
    for family, chars in available_chars.items():
        target = round(target_total_chars * active_weights.get(family, 1.0) / weight_sum)
        quotas[family] = min(chars, target)
    return quotas


def _weighted_order(
    samples: list[ProcessedSample],
    source_weights: dict[str, float],
    seed: int,
) -> list[ProcessedSample]:
    rng = random.Random(seed)
    weighted = []
    for sample in samples:
        weight = max(1e-6, source_weights.get(sample.source_name, sample.metadata.get("sampling_weight", 1.0) if isinstance(sample.metadata.get("sampling_weight"), (int, float)) else 1.0))
        priority = -math.log(max(rng.random(), 1e-9)) / weight
        weighted.append((priority, -sample.quality_score, sample.sample_id, sample))
    weighted.sort()
    return [item[-1] for item in weighted]


def _summarize_mixture(
    selected: list[ProcessedSample],
    ordered: dict[str, list[ProcessedSample]],
    quotas: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for family, samples in ordered.items():
        selected_family = [sample for sample in selected if sample.family == family]
        available_chars = sum(int(sample.quality_metrics.get("char_count", len(sample.normalized_text))) for sample in samples)
        selected_chars = sum(int(sample.quality_metrics.get("char_count", len(sample.normalized_text))) for sample in selected_family)
        summary[family] = {
            "available_samples": len(samples),
            "available_chars": available_chars,
            "selected_samples": len(selected_family),
            "selected_chars": selected_chars,
            "target_chars": quotas.get(family, available_chars),
        }
    return summary
