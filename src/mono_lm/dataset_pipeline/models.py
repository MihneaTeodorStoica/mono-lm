from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DialogueTurn:
    role: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text}


@dataclass(slots=True)
class RawSample:
    sample_id: str
    source_name: str
    family: str
    layout: str
    local_id: str
    origin_path: str
    split_group: str
    title: str | None = None
    text: str | None = None
    context: str | None = None
    question: str | None = None
    answer: str | None = None
    turns: list[DialogueTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_name": self.source_name,
            "family": self.family,
            "layout": self.layout,
            "local_id": self.local_id,
            "origin_path": self.origin_path,
            "split_group": self.split_group,
            "title": self.title,
            "text": self.text,
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ProcessedSample:
    sample_id: str
    source_name: str
    family: str
    layout: str
    local_id: str
    origin_path: str
    split_group: str
    title: str | None = None
    text: str | None = None
    context: str | None = None
    question: str | None = None
    answer: str | None = None
    turns: list[DialogueTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_text: str = ""
    formatted_text: str = ""
    quality_metrics: dict[str, float | int] = field(default_factory=dict)
    quality_score: float = 0.0
    exact_hash: str = ""
    simhash: int | None = None
    duplicate_cluster: str = ""
    selected_for_final: bool = False
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_name": self.source_name,
            "family": self.family,
            "layout": self.layout,
            "local_id": self.local_id,
            "origin_path": self.origin_path,
            "split_group": self.split_group,
            "title": self.title,
            "text": self.text,
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata,
            "normalized_text": self.normalized_text,
            "formatted_text": self.formatted_text,
            "quality_metrics": self.quality_metrics,
            "quality_score": round(self.quality_score, 6),
            "exact_hash": self.exact_hash,
            "simhash": self.simhash,
            "duplicate_cluster": self.duplicate_cluster,
            "selected_for_final": self.selected_for_final,
            "split": self.split,
        }


@dataclass(slots=True)
class RejectedSample:
    sample_id: str
    source_name: str
    family: str
    stage: str
    reason: str
    detail: str
    origin_path: str
    preview: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_name": self.source_name,
            "family": self.family,
            "stage": self.stage,
            "reason": self.reason,
            "detail": self.detail,
            "origin_path": self.origin_path,
            "preview": self.preview,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PipelineOutputs:
    output_dir: Path
    manifest_path: Path
    report_json_path: Path
    report_markdown_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
