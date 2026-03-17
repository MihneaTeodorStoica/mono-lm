from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CharacterVocabulary:
    chars: list[str]
    stoi: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("Character vocabulary contains duplicate entries.")
        self.stoi = {char: index for index, char in enumerate(self.chars)}

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            missing = str(exc.args[0])
            raise ValueError(f"Character {missing!r} is not present in the vocabulary.") from exc

    def encode_array(self, text: str) -> np.ndarray:
        return np.asarray(self.encode(text), dtype=dtype_for_vocab_size(self.size))

    def decode(self, indices: list[int] | np.ndarray) -> str:
        return "".join(self.chars[int(index)] for index in indices)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "size": self.size,
            "chars": self.chars,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CharacterVocabulary:
        chars = payload.get("chars")
        if not isinstance(chars, list) or not all(isinstance(char, str) for char in chars):
            raise ValueError("Vocabulary payload must contain a string list under 'chars'.")
        return cls(chars=list(chars))

    @classmethod
    def load(cls, path: Path) -> CharacterVocabulary:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_character_vocabulary(texts: dict[str, str]) -> CharacterVocabulary:
    unique_chars = sorted({char for text in texts.values() for char in text})
    return CharacterVocabulary(chars=unique_chars)


def dtype_for_vocab_size(size: int) -> np.dtype:
    if size <= np.iinfo(np.uint16).max:
        return np.uint16
    if size <= np.iinfo(np.uint32).max:
        return np.uint32
    raise ValueError(f"Vocabulary is too large for this baseline implementation: {size}")
