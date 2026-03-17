from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


FAMILY_ALIASES = {
    "article": "expository",
    "document": "expository",
    "discussion": "qa",
    "conversation": "dialogue",
    "story": "prose",
}


def _normalize_family(value: str) -> str:
    normalized = value.strip().lower()
    return FAMILY_ALIASES.get(normalized, normalized)


def _default_layout_for_family(family: str) -> str:
    if family in {"expository", "prose"}:
        return "document"
    if family == "qa":
        return "qa"
    if family == "dialogue":
        return "dialogue"
    raise ValueError(f"Unsupported source family: {family}")


@dataclass(slots=True)
class PipelineConfig:
    name: str
    seed: int
    output_dir: Path
    target_total_chars: int | None = None
    preview_chars: int = 320
    inspection_samples_per_family: int = 4


@dataclass(slots=True)
class CleaningConfig:
    normalize_unicode: bool = True
    strip_html: bool = True
    strip_wiki_markup: bool = True
    strip_zero_width: bool = True
    collapse_internal_whitespace: bool = True
    collapse_blank_lines_to: int = 2
    trim_line_edges: bool = True
    decode_html_entities: bool = True


@dataclass(slots=True)
class QualityConfig:
    min_chars: int = 120
    max_chars: int = 15000
    min_unique_chars: int = 20
    min_alpha_ratio: float = 0.55
    max_symbol_ratio: float = 0.22
    max_digit_ratio: float = 0.25
    max_nonprintable_ratio: float = 0.01
    max_line_length: int = 500
    max_duplicate_line_ratio: float = 0.45
    max_repeated_ngram_ratio: float = 0.25
    allow_urls: bool = False


@dataclass(slots=True)
class DedupConfig:
    enabled: bool = True
    exact: bool = True
    near: bool = True
    simhash_threshold: int = 6
    minimum_tokens_for_near_dedup: int = 24


@dataclass(slots=True)
class FormattingConfig:
    include_family_header: bool = True
    include_source_header: bool = False
    include_titles: bool = True
    document_separator: str = "\n\n<|endofsample|>\n\n"
    article_header: str = "Article"
    qa_header: str = "Q&A"
    conversation_header: str = "Conversation"
    story_header: str = "Story"


@dataclass(slots=True)
class SplitConfig:
    train: float = 0.96
    validation: float = 0.02
    test: float = 0.02

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(slots=True)
class MixtureConfig:
    family_weights: dict[str, float] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)
    minimum_samples_per_family: int = 1


@dataclass(slots=True)
class SourceConfig:
    name: str
    family: str
    kind: str
    path: Path
    enabled: bool = True
    layout: str | None = None
    glob: str = "**/*"
    encoding: str = "utf-8"
    delimiter: str = ","
    id_field: str | None = None
    group_field: str | None = None
    title_field: str | None = None
    text_field: str | None = None
    context_field: str | None = None
    question_field: str | None = None
    answer_field: str | None = None
    turns_field: str | None = None
    turn_role_field: str = "role"
    turn_text_field: str = "content"
    metadata_fields: tuple[str, ...] = ()
    max_records: int | None = None
    title_from_first_line: bool = False
    sampling_weight: float = 1.0

    def resolved_layout(self) -> str:
        return self.layout or _default_layout_for_family(self.family)


@dataclass(slots=True)
class BuildConfig:
    root_dir: Path
    config_path: Path
    pipeline: PipelineConfig
    cleaning: CleaningConfig
    quality: QualityConfig
    dedup: DedupConfig
    formatting: FormattingConfig
    split: SplitConfig
    mixture: MixtureConfig
    sources: list[SourceConfig]


def _coerce_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _discover_project_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return config_path.parent


def _load_source(root: Path, payload: dict[str, Any]) -> SourceConfig:
    family = _normalize_family(str(payload["family"]))
    kind = str(payload["kind"]).strip().lower()
    metadata_fields = tuple(payload.get("metadata_fields", []))
    source = SourceConfig(
        name=str(payload["name"]),
        family=family,
        kind=kind,
        path=_coerce_path(root, str(payload["path"])),
        enabled=bool(payload.get("enabled", True)),
        layout=payload.get("layout"),
        glob=str(payload.get("glob", "**/*")),
        encoding=str(payload.get("encoding", "utf-8")),
        delimiter=str(payload.get("delimiter", ",")),
        id_field=payload.get("id_field"),
        group_field=payload.get("group_field"),
        title_field=payload.get("title_field"),
        text_field=payload.get("text_field"),
        context_field=payload.get("context_field"),
        question_field=payload.get("question_field"),
        answer_field=payload.get("answer_field"),
        turns_field=payload.get("turns_field"),
        turn_role_field=str(payload.get("turn_role_field", "role")),
        turn_text_field=str(payload.get("turn_text_field", "content")),
        metadata_fields=metadata_fields,
        max_records=payload.get("max_records"),
        title_from_first_line=bool(payload.get("title_from_first_line", False)),
        sampling_weight=float(payload.get("sampling_weight", 1.0)),
    )
    layout = source.resolved_layout()
    required_map = {
        "document": ("text_field",) if kind in {"jsonl", "csv"} else (),
        "qa": ("question_field", "answer_field"),
        "dialogue": ("turns_field",),
    }
    missing = [name for name in required_map.get(layout, ()) if getattr(source, name) is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Source '{source.name}' is missing required fields for layout '{layout}': {joined}")
    return source


def load_config(path: str | Path) -> BuildConfig:
    config_path = Path(path).resolve()
    root_dir = _discover_project_root(config_path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    pipeline_table = raw.get("pipeline", {})
    pipeline = PipelineConfig(
        name=str(pipeline_table.get("name", "mono-lm-dataset")),
        seed=int(pipeline_table.get("seed", 13)),
        output_dir=_coerce_path(root_dir, str(pipeline_table.get("output_dir", "data/artifacts/default"))),
        target_total_chars=pipeline_table.get("target_total_chars"),
        preview_chars=int(pipeline_table.get("preview_chars", 320)),
        inspection_samples_per_family=int(pipeline_table.get("inspection_samples_per_family", 4)),
    )

    cleaning = CleaningConfig(**raw.get("cleaning", {}))
    quality = QualityConfig(**raw.get("quality", {}))
    dedup = DedupConfig(**raw.get("dedup", {}))
    formatting = FormattingConfig(**raw.get("formatting", {}))
    split = SplitConfig(**raw.get("split", {}))

    split_total = split.train + split.validation + split.test
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0; received {split_total:.6f}")

    mixture_table = raw.get("mixture", {})
    family_weights = {
        _normalize_family(name): float(weight)
        for name, weight in mixture_table.get("family_weights", {}).items()
    }
    source_weights = {str(name): float(weight) for name, weight in mixture_table.get("source_weights", {}).items()}
    mixture = MixtureConfig(
        family_weights=family_weights,
        source_weights=source_weights,
        minimum_samples_per_family=int(mixture_table.get("minimum_samples_per_family", 1)),
    )

    raw_sources = raw.get("sources", [])
    if not raw_sources:
        raise ValueError("Config must define at least one [[sources]] entry.")
    sources = [_load_source(root_dir, item) for item in raw_sources]
    return BuildConfig(
        root_dir=root_dir,
        config_path=config_path,
        pipeline=pipeline,
        cleaning=cleaning,
        quality=quality,
        dedup=dedup,
        formatting=formatting,
        split=split,
        mixture=mixture,
        sources=sources,
    )
