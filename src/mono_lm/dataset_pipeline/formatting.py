from __future__ import annotations

from .config import FormattingConfig
from .models import DialogueTurn, ProcessedSample


ROLE_ALIASES = {
    "assistant": "Assistant",
    "bot": "Assistant",
    "model": "Assistant",
    "ai": "Assistant",
    "user": "User",
    "human": "User",
    "system": "System",
    "narrator": "Narrator",
}


def normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized in ROLE_ALIASES:
        return ROLE_ALIASES[normalized]
    return role.strip().title() or "Speaker"


def canonical_text(sample: ProcessedSample) -> str:
    parts: list[str] = []
    if sample.layout == "document":
        if sample.title:
            parts.append(sample.title)
        if sample.text:
            parts.append(sample.text)
    elif sample.layout == "qa":
        if sample.context:
            parts.append(f"Context: {sample.context}")
        if sample.question:
            parts.append(f"Question: {sample.question}")
        if sample.answer:
            parts.append(f"Answer: {sample.answer}")
    elif sample.layout == "dialogue":
        for turn in sample.turns:
            parts.append(f"{turn.role}: {turn.text}")
    return "\n\n".join(part for part in parts if part).strip()


def format_sample(sample: ProcessedSample, config: FormattingConfig) -> str:
    blocks: list[str] = []
    family_header = {
        "expository": config.article_header,
        "qa": config.qa_header,
        "dialogue": config.conversation_header,
        "prose": config.story_header,
    }.get(sample.family)
    if config.include_family_header and family_header:
        blocks.append(family_header)
    if config.include_source_header:
        blocks.append(f"Source: {sample.source_name}")

    if sample.layout == "document":
        if config.include_titles and sample.title:
            blocks.append(f"Title: {sample.title}")
        if sample.text:
            blocks.append(sample.text)
    elif sample.layout == "qa":
        if sample.context:
            blocks.append(f"Context: {sample.context}")
        if sample.question:
            blocks.append(f"Question: {sample.question}")
        if sample.answer:
            blocks.append(f"Answer: {sample.answer}")
    elif sample.layout == "dialogue":
        conversation_lines = [f"{turn.role}: {turn.text}" for turn in sample.turns]
        if conversation_lines:
            blocks.append("\n".join(conversation_lines))
    return "\n\n".join(block for block in blocks if block).strip()


def merge_turns(turns: list[DialogueTurn]) -> list[DialogueTurn]:
    merged: list[DialogueTurn] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            merged[-1].text = f"{merged[-1].text}\n\n{turn.text}"
        else:
            merged.append(DialogueTurn(role=turn.role, text=turn.text))
    return merged
