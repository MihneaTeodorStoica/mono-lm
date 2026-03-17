from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import BuildConfig, SourceConfig
from .models import DialogueTurn, RawSample, RejectedSample
from .utils import preview_text


def load_sources(config: BuildConfig) -> tuple[list[RawSample], list[RejectedSample]]:
    samples: list[RawSample] = []
    rejected: list[RejectedSample] = []
    for source in config.sources:
        if not source.enabled:
            continue
        loader = {
            "text_dir": _load_text_dir,
            "text_file": _load_text_file,
            "jsonl": _load_jsonl,
            "csv": _load_csv,
        }.get(source.kind)
        if loader is None:
            raise ValueError(f"Unsupported source kind '{source.kind}' for source '{source.name}'")
        loaded_samples, load_rejections = loader(source)
        samples.extend(loaded_samples)
        rejected.extend(load_rejections)
    return samples, rejected


def _sample_id(source: SourceConfig, local_id: str) -> str:
    return f"{source.name}:{local_id}"


def _load_text_dir(source: SourceConfig) -> tuple[list[RawSample], list[RejectedSample]]:
    matched = sorted(path for path in source.path.glob(source.glob) if path.is_file())
    return _load_text_paths(source, matched)


def _load_text_file(source: SourceConfig) -> tuple[list[RawSample], list[RejectedSample]]:
    return _load_text_paths(source, [source.path])


def _load_text_paths(source: SourceConfig, paths: list[Path]) -> tuple[list[RawSample], list[RejectedSample]]:
    samples: list[RawSample] = []
    rejected: list[RejectedSample] = []
    limit = source.max_records or len(paths)
    for path in paths[:limit]:
        try:
            raw_text = path.read_text(encoding=source.encoding)
        except OSError as exc:
            rejected.append(
                RejectedSample(
                    sample_id=_sample_id(source, str(path)),
                    source_name=source.name,
                    family=source.family,
                    stage="ingest",
                    reason="read_error",
                    detail=str(exc),
                    origin_path=str(path),
                    preview="",
                    metadata={},
                )
            )
            continue
        title: str | None = None
        text = raw_text
        if source.title_from_first_line:
            title, text = _split_title_from_first_line(raw_text)
        relative = path.relative_to(source.path if source.path.is_dir() else source.path.parent)
        local_id = str(relative)
        samples.append(
            RawSample(
                sample_id=_sample_id(source, local_id),
                source_name=source.name,
                family=source.family,
                layout=source.resolved_layout(),
                local_id=local_id,
                origin_path=str(path),
                split_group=_group_id(source, {"__path__": local_id}, local_id),
                title=title,
                text=text,
                metadata={"path": local_id, "sampling_weight": source.sampling_weight},
            )
        )
    return samples, rejected


def _split_title_from_first_line(raw_text: str) -> tuple[str | None, str]:
    lines = raw_text.splitlines()
    first_non_empty = next((idx for idx, line in enumerate(lines) if line.strip()), None)
    if first_non_empty is None:
        return None, raw_text
    title = lines[first_non_empty].strip().lstrip("#").strip()
    body = "\n".join(lines[first_non_empty + 1 :]).strip()
    if not title:
        return None, raw_text
    return title, body or title


def _load_jsonl(source: SourceConfig) -> tuple[list[RawSample], list[RejectedSample]]:
    samples: list[RawSample] = []
    rejected: list[RejectedSample] = []
    with source.path.open("r", encoding=source.encoding) as handle:
        for index, line in enumerate(handle, start=1):
            if source.max_records is not None and len(samples) >= source.max_records:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(
                    RejectedSample(
                        sample_id=_sample_id(source, f"line-{index}"),
                        source_name=source.name,
                        family=source.family,
                        stage="ingest",
                        reason="json_decode_error",
                        detail=str(exc),
                        origin_path=str(source.path),
                        preview=preview_text(line),
                        metadata={"line": index},
                    )
                )
                continue
            sample, rejection = _structured_record_to_sample(source, record, index, str(source.path))
            if sample is not None:
                samples.append(sample)
            if rejection is not None:
                rejected.append(rejection)
    return samples, rejected


def _load_csv(source: SourceConfig) -> tuple[list[RawSample], list[RejectedSample]]:
    samples: list[RawSample] = []
    rejected: list[RejectedSample] = []
    with source.path.open("r", encoding=source.encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=source.delimiter)
        for index, record in enumerate(reader, start=1):
            if source.max_records is not None and len(samples) >= source.max_records:
                break
            sample, rejection = _structured_record_to_sample(source, record, index, str(source.path))
            if sample is not None:
                samples.append(sample)
            if rejection is not None:
                rejected.append(rejection)
    return samples, rejected


def _structured_record_to_sample(
    source: SourceConfig,
    record: dict[str, Any],
    index: int,
    origin_path: str,
) -> tuple[RawSample | None, RejectedSample | None]:
    layout = source.resolved_layout()
    local_id = str(record.get(source.id_field)) if source.id_field else f"row-{index}"
    metadata = {field: record.get(field) for field in source.metadata_fields}
    metadata["sampling_weight"] = source.sampling_weight
    split_group = _group_id(source, record, local_id)

    try:
        if layout == "document":
            text = _require_text(record, source.text_field, "text_field")
            title = record.get(source.title_field) if source.title_field else None
            sample = RawSample(
                sample_id=_sample_id(source, local_id),
                source_name=source.name,
                family=source.family,
                layout=layout,
                local_id=local_id,
                origin_path=origin_path,
                split_group=split_group,
                title=str(title) if title is not None else None,
                text=str(text),
                metadata=metadata,
            )
            return sample, None
        if layout == "qa":
            question = _require_text(record, source.question_field, "question_field")
            answer = _require_text(record, source.answer_field, "answer_field")
            context = record.get(source.context_field) if source.context_field else None
            sample = RawSample(
                sample_id=_sample_id(source, local_id),
                source_name=source.name,
                family=source.family,
                layout=layout,
                local_id=local_id,
                origin_path=origin_path,
                split_group=split_group,
                context=str(context) if context is not None else None,
                question=str(question),
                answer=str(answer),
                metadata=metadata,
            )
            return sample, None
        if layout == "dialogue":
            turns_raw = record.get(source.turns_field or "")
            if not isinstance(turns_raw, list):
                raise ValueError("dialogue turns must be a list of objects")
            turns = []
            for turn in turns_raw:
                if not isinstance(turn, dict):
                    raise ValueError("dialogue turns must contain objects")
                role = str(turn.get(source.turn_role_field, "")).strip()
                text = str(turn.get(source.turn_text_field, "")).strip()
                if role and text:
                    turns.append(DialogueTurn(role=role, text=text))
            if not turns:
                raise ValueError("dialogue contains no usable turns")
            sample = RawSample(
                sample_id=_sample_id(source, local_id),
                source_name=source.name,
                family=source.family,
                layout=layout,
                local_id=local_id,
                origin_path=origin_path,
                split_group=split_group,
                turns=turns,
                metadata=metadata,
            )
            return sample, None
    except ValueError as exc:
        preview_parts = [str(value) for value in record.values() if value not in {None, ""}]
        rejection = RejectedSample(
            sample_id=_sample_id(source, local_id),
            source_name=source.name,
            family=source.family,
            stage="ingest",
            reason="record_validation_error",
            detail=str(exc),
            origin_path=origin_path,
            preview=preview_text(" | ".join(preview_parts)),
            metadata=metadata,
        )
        return None, rejection
    raise ValueError(f"Unsupported layout '{layout}'")


def _require_text(record: dict[str, Any], field_name: str | None, label: str) -> str:
    if field_name is None:
        raise ValueError(f"{label} was not configured")
    value = record.get(field_name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required field '{field_name}'")
    return str(value)


def _group_id(source: SourceConfig, record: dict[str, Any], fallback: str) -> str:
    if source.group_field:
        value = record.get(source.group_field)
        if value not in {None, ""}:
            return f"{source.name}:{value}"
    return f"{source.name}:{fallback}"
