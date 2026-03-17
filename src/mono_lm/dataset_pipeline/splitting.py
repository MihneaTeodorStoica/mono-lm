from __future__ import annotations

from collections import defaultdict

from .config import SplitConfig
from .models import ProcessedSample
from .utils import stable_hash


def assign_splits(
    samples: list[ProcessedSample],
    split_config: SplitConfig,
    seed: int,
) -> dict[str, list[ProcessedSample]]:
    groups: dict[str, list[ProcessedSample]] = defaultdict(list)
    for sample in samples:
        group_key = sample.duplicate_cluster or sample.split_group or sample.sample_id
        groups[group_key].append(sample)

    targets = split_config.as_dict()
    total_chars = sum(_sample_chars(sample) for sample in samples)
    target_chars = {split: total_chars * ratio for split, ratio in targets.items()}
    current_chars = {split: 0 for split in targets}
    assignments: dict[str, list[ProcessedSample]] = {split: [] for split in targets}

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -sum(_sample_chars(sample) for sample in item[1]),
            stable_hash(item[0], seed),
        ),
    )

    for index, (group_key, group_samples) in enumerate(ordered_groups):
        remaining_groups = len(ordered_groups) - index
        empty_splits = [split for split, assigned in assignments.items() if not assigned]
        candidate_splits = empty_splits if empty_splits and len(empty_splits) >= remaining_groups else list(assignments.keys())
        chosen = max(
            candidate_splits,
            key=lambda split: (
                target_chars[split] - current_chars[split],
                -stable_hash(f"{group_key}:{split}", seed),
            ),
        )
        assignments[chosen].extend(group_samples)
        current_chars[chosen] += sum(_sample_chars(sample) for sample in group_samples)
        for sample in group_samples:
            sample.split = chosen
    return assignments


def _sample_chars(sample: ProcessedSample) -> int:
    return int(sample.quality_metrics.get("char_count", len(sample.normalized_text)))
