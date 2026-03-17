from __future__ import annotations

from dataclasses import dataclass
import re

from .config import QualityConfig
from .models import ProcessedSample


@dataclass(slots=True)
class QualityOutcome:
    accepted: bool
    reason: str | None
    detail: str
    metrics: dict[str, float | int]
    quality_score: float


def assess_sample(sample: ProcessedSample, config: QualityConfig) -> QualityOutcome:
    text = sample.normalized_text.strip()
    char_count = len(text)
    unique_chars = len(set(text))
    non_whitespace_chars = [char for char in text if not char.isspace()]
    denominator = max(1, len(non_whitespace_chars))
    alpha_ratio = sum(char.isalpha() for char in non_whitespace_chars) / denominator
    digit_ratio = sum(char.isdigit() for char in non_whitespace_chars) / denominator
    symbol_ratio = sum((not char.isalnum()) for char in non_whitespace_chars) / denominator
    nonprintable_ratio = sum((not char.isprintable()) and (not char.isspace()) for char in text) / max(1, len(text))
    line_lengths = [len(line) for line in text.splitlines()] or [0]
    max_line_length = max(line_lengths)
    duplicate_line_ratio = _duplicate_line_ratio(text)
    repeated_ngram_ratio = _repeated_ngram_ratio(text, n=3)
    url_count = len(re.findall(r"https?://|www\.", text))

    metrics = {
        "char_count": char_count,
        "unique_chars": unique_chars,
        "alpha_ratio": round(alpha_ratio, 6),
        "digit_ratio": round(digit_ratio, 6),
        "symbol_ratio": round(symbol_ratio, 6),
        "nonprintable_ratio": round(nonprintable_ratio, 6),
        "max_line_length": max_line_length,
        "duplicate_line_ratio": round(duplicate_line_ratio, 6),
        "repeated_ngram_ratio": round(repeated_ngram_ratio, 6),
        "url_count": url_count,
    }

    checks = [
        (char_count < config.min_chars, "too_short", f"char_count={char_count} < {config.min_chars}"),
        (char_count > config.max_chars, "too_long", f"char_count={char_count} > {config.max_chars}"),
        (unique_chars < config.min_unique_chars, "low_character_variety", f"unique_chars={unique_chars} < {config.min_unique_chars}"),
        (alpha_ratio < config.min_alpha_ratio, "low_alpha_ratio", f"alpha_ratio={alpha_ratio:.3f} < {config.min_alpha_ratio:.3f}"),
        (digit_ratio > config.max_digit_ratio, "digit_heavy", f"digit_ratio={digit_ratio:.3f} > {config.max_digit_ratio:.3f}"),
        (symbol_ratio > config.max_symbol_ratio, "symbol_heavy", f"symbol_ratio={symbol_ratio:.3f} > {config.max_symbol_ratio:.3f}"),
        (nonprintable_ratio > config.max_nonprintable_ratio, "nonprintable_noise", f"nonprintable_ratio={nonprintable_ratio:.3f} > {config.max_nonprintable_ratio:.3f}"),
        (max_line_length > config.max_line_length, "line_too_long", f"max_line_length={max_line_length} > {config.max_line_length}"),
        (
            duplicate_line_ratio > config.max_duplicate_line_ratio,
            "duplicate_lines",
            f"duplicate_line_ratio={duplicate_line_ratio:.3f} > {config.max_duplicate_line_ratio:.3f}",
        ),
        (
            repeated_ngram_ratio > config.max_repeated_ngram_ratio,
            "repetition_heavy",
            f"repeated_ngram_ratio={repeated_ngram_ratio:.3f} > {config.max_repeated_ngram_ratio:.3f}",
        ),
        ((not config.allow_urls) and url_count > 0, "contains_url", f"url_count={url_count}"),
    ]
    for condition, reason, detail in checks:
        if condition:
            return QualityOutcome(False, reason, detail, metrics, 0.0)

    length_bonus = min(1.0, char_count / 1200)
    quality_score = (
        (alpha_ratio * 2.5)
        + (1.0 - digit_ratio)
        + (1.0 - symbol_ratio)
        + (1.0 - duplicate_line_ratio)
        + (1.0 - repeated_ngram_ratio)
        + length_bonus
    ) / 7.5
    return QualityOutcome(True, None, "", metrics, quality_score)


def _duplicate_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    unique_count = len(set(lines))
    return max(0.0, (len(lines) - unique_count) / len(lines))


def _repeated_ngram_ratio(text: str, n: int) -> float:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    unique_count = len(set(ngrams))
    return max(0.0, (len(ngrams) - unique_count) / len(ngrams))
