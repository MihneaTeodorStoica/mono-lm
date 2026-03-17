from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re

from .config import DedupConfig
from .models import ProcessedSample, RejectedSample
from .utils import preview_text, sha1_text


@dataclass(slots=True)
class DedupSummary:
    exact_clusters: int
    exact_removed: int
    near_clusters: int
    near_removed: int


def deduplicate_samples(
    samples: list[ProcessedSample],
    config: DedupConfig,
) -> tuple[list[ProcessedSample], list[RejectedSample], DedupSummary]:
    if not config.enabled:
        for sample in samples:
            sample.duplicate_cluster = f"cluster-{sha1_text(sample.sample_id)[:12]}"
        return samples, [], DedupSummary(0, 0, len(samples), 0)

    exact_kept, exact_rejected, exact_clusters = _exact_dedup(samples, config)
    near_kept, near_rejected, near_clusters = _near_dedup(exact_kept, config)
    summary = DedupSummary(
        exact_clusters=exact_clusters,
        exact_removed=len(exact_rejected),
        near_clusters=near_clusters,
        near_removed=len(near_rejected),
    )
    return near_kept, [*exact_rejected, *near_rejected], summary


def _exact_dedup(
    samples: list[ProcessedSample],
    config: DedupConfig,
) -> tuple[list[ProcessedSample], list[RejectedSample], int]:
    if not config.exact:
        for sample in samples:
            sample.exact_hash = sha1_text(sample.normalized_text)
        return samples, [], len(samples)

    grouped: dict[str, list[ProcessedSample]] = defaultdict(list)
    for sample in samples:
        sample.exact_hash = sha1_text(sample.normalized_text)
        grouped[sample.exact_hash].append(sample)

    kept: list[ProcessedSample] = []
    rejected: list[RejectedSample] = []
    for exact_hash, group in grouped.items():
        representative = _choose_representative(group)
        representative.duplicate_cluster = f"cluster-exact-{exact_hash[:12]}"
        kept.append(representative)
        for duplicate in group:
            if duplicate is representative:
                continue
            rejected.append(
                RejectedSample(
                    sample_id=duplicate.sample_id,
                    source_name=duplicate.source_name,
                    family=duplicate.family,
                    stage="dedup_exact",
                    reason="exact_duplicate",
                    detail=f"duplicate_of={representative.sample_id}",
                    origin_path=duplicate.origin_path,
                    preview=preview_text(duplicate.formatted_text),
                    metadata={"duplicate_cluster": representative.duplicate_cluster},
                )
            )
    return kept, rejected, len(grouped)


def _near_dedup(
    samples: list[ProcessedSample],
    config: DedupConfig,
) -> tuple[list[ProcessedSample], list[RejectedSample], int]:
    if not config.near or len(samples) < 2:
        for sample in samples:
            sample.duplicate_cluster = sample.duplicate_cluster or f"cluster-{sample.exact_hash[:12]}"
        return samples, [], len(samples)

    token_lists = [_tokens(sample.normalized_text) for sample in samples]
    for sample, tokens in zip(samples, token_lists):
        if len(tokens) >= config.minimum_tokens_for_near_dedup:
            sample.simhash = _simhash(tokens)

    union_find = _UnionFind(len(samples))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.simhash is None:
            continue
        for band_index in range(4):
            band = (sample.simhash >> (band_index * 16)) & 0xFFFF
            key = (band_index, band)
            for other_index in buckets[key]:
                other = samples[other_index]
                if other.simhash is None:
                    continue
                if _length_ratio(sample, other) > 1.45:
                    continue
                if _hamming_distance(sample.simhash, other.simhash) <= config.simhash_threshold:
                    union_find.union(index, other_index)
            buckets[key].append(index)

    clusters: dict[int, list[ProcessedSample]] = defaultdict(list)
    for index, sample in enumerate(samples):
        clusters[union_find.find(index)].append(sample)

    kept: list[ProcessedSample] = []
    rejected: list[RejectedSample] = []
    for cluster_index, group in clusters.items():
        representative = _choose_representative(group)
        representative.duplicate_cluster = f"cluster-near-{cluster_index:06d}"
        kept.append(representative)
        for duplicate in group:
            if duplicate is representative:
                continue
            rejected.append(
                RejectedSample(
                    sample_id=duplicate.sample_id,
                    source_name=duplicate.source_name,
                    family=duplicate.family,
                    stage="dedup_near",
                    reason="near_duplicate",
                    detail=f"duplicate_of={representative.sample_id}",
                    origin_path=duplicate.origin_path,
                    preview=preview_text(duplicate.formatted_text),
                    metadata={"duplicate_cluster": representative.duplicate_cluster},
                )
            )
    return kept, rejected, len(clusters)


def _choose_representative(group: list[ProcessedSample]) -> ProcessedSample:
    return max(
        group,
        key=lambda sample: (
            sample.quality_score,
            sample.quality_metrics.get("char_count", 0),
            sample.sample_id,
        ),
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _simhash(tokens: list[str]) -> int:
    features = [" ".join(tokens[index : index + 3]) for index in range(max(1, len(tokens) - 2))]
    weights = [0] * 64
    for feature in features:
        digest = int.from_bytes(hashlib.sha1(feature.encode("utf-8")).digest()[:8], "big", signed=False)
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _length_ratio(left: ProcessedSample, right: ProcessedSample) -> float:
    left_length = max(1, int(left.quality_metrics.get("char_count", len(left.normalized_text))))
    right_length = max(1, int(right.quality_metrics.get("char_count", len(right.normalized_text))))
    return max(left_length, right_length) / min(left_length, right_length)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1
