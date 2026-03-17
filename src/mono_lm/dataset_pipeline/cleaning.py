from __future__ import annotations

import html
import re
import unicodedata

from .config import CleaningConfig, FormattingConfig
from .formatting import canonical_text, format_sample, merge_turns, normalize_role
from .models import DialogueTurn, ProcessedSample, RawSample


ZERO_WIDTH_CHARS = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
}


def normalize_raw_sample(
    sample: RawSample,
    cleaning: CleaningConfig,
    formatting: FormattingConfig,
) -> ProcessedSample:
    title = _clean_title(sample.title, cleaning) if sample.title else None
    text = _clean_block(sample.text, cleaning) if sample.text else None
    context = _clean_block(sample.context, cleaning) if sample.context else None
    question = _clean_block(sample.question, cleaning) if sample.question else None
    answer = _clean_block(sample.answer, cleaning) if sample.answer else None
    turns = _clean_turns(sample.turns, cleaning)

    processed = ProcessedSample(
        sample_id=sample.sample_id,
        source_name=sample.source_name,
        family=sample.family,
        layout=sample.layout,
        local_id=sample.local_id,
        origin_path=sample.origin_path,
        split_group=sample.split_group,
        title=title,
        text=text,
        context=context,
        question=question,
        answer=answer,
        turns=turns,
        metadata=sample.metadata,
    )
    processed.normalized_text = canonical_text(processed)
    processed.formatted_text = format_sample(processed, formatting)
    return processed


def _clean_turns(turns: list[DialogueTurn], config: CleaningConfig) -> list[DialogueTurn]:
    cleaned: list[DialogueTurn] = []
    for turn in turns:
        text = _clean_block(turn.text, config)
        role = normalize_role(turn.role)
        if text:
            cleaned.append(DialogueTurn(role=role, text=text))
    return merge_turns(cleaned)


def _clean_title(value: str, config: CleaningConfig) -> str:
    cleaned = _clean_inline(value, config)
    return cleaned[:200].strip()


def _clean_block(value: str, config: CleaningConfig) -> str:
    text = value
    if config.decode_html_entities:
        text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if config.strip_zero_width:
        for char in ZERO_WIDTH_CHARS:
            text = text.replace(char, "")
    if config.normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    if config.strip_html:
        text = re.sub(r"<[^>\n]+>", " ", text)
    if config.strip_wiki_markup:
        text = _strip_wiki_markup(text)
    lines = [_clean_line(line, config) for line in text.split("\n")]
    text = "\n".join(lines)
    max_blank_lines = max(1, config.collapse_blank_lines_to)
    text = re.sub(rf"\n{{{max_blank_lines + 1},}}", "\n" * max_blank_lines, text)
    return text.strip()


def _clean_inline(value: str, config: CleaningConfig) -> str:
    text = _clean_block(value, config)
    return re.sub(r"\s+", " ", text).strip()


def _clean_line(line: str, config: CleaningConfig) -> str:
    text = line
    if config.collapse_internal_whitespace:
        text = re.sub(r"[^\S\n]+", " ", text)
    if config.trim_line_edges:
        text = text.strip()
    text = re.sub(r" ?([,;:.!?])", r"\1", text)
    text = re.sub(r"([(\[{]) ", r"\1", text)
    text = re.sub(r" ([)\]}])", r"\1", text)
    return text


def _strip_wiki_markup(text: str) -> str:
    text = re.sub(r"__[^_]+__", " ", text)
    text = re.sub(r"\[\[Category:[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^{}]+\}\}", " ", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)
    text = re.sub(r"^\s*(Category|References|External links)\s*:.*$", " ", text, flags=re.IGNORECASE | re.MULTILINE)
    return text
