from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrainingConfig
from .vocabulary import CharacterVocabulary, build_character_vocabulary


@dataclass(slots=True)
class PreparedCorpus:
    prepared_dir: Path
    manifest_path: Path
    vocab_path: Path
    split_paths: dict[str, Path]
    manifest: dict[str, Any]


def prepare_corpus(config: TrainingConfig, force: bool = False) -> PreparedCorpus:
    prepared_dir = config.data.prepared_dir
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared_dir / "corpus_manifest.json"
    vocab_path = prepared_dir / "vocab.json"

    if config.data.reuse_prepared and manifest_path.exists() and vocab_path.exists() and not force:
        return load_prepared_corpus(prepared_dir)

    input_paths = config.resolve_text_paths()
    texts: dict[str, str] = {}
    split_paths: dict[str, Path] = {}
    for split, path in input_paths.items():
        if path is None:
            continue
        if not path.exists():
            if split == "test":
                continue
            raise FileNotFoundError(f"Expected {split} text file at {path}")
        texts[split] = path.read_text(encoding="utf-8")

    if "train" not in texts or "validation" not in texts:
        raise ValueError("Prepared corpus requires both train and validation text.")

    vocabulary = build_character_vocabulary(texts)
    vocabulary.save(vocab_path)

    split_stats: dict[str, dict[str, Any]] = {}
    for split, text in texts.items():
        encoded = vocabulary.encode_array(text)
        split_path = prepared_dir / f"{split}.npy"
        np.save(split_path, encoded, allow_pickle=False)
        split_paths[split] = split_path
        split_stats[split] = {
            "text_path": str(input_paths[split]),
            "encoded_path": str(split_path),
            "characters": len(text),
            "tokens": int(encoded.shape[0]),
        }

    manifest = {
        "config_path": str(config.config_path),
        "prepared_dir": str(prepared_dir),
        "vocab_path": str(vocab_path),
        "vocab_size": vocabulary.size,
        "splits": split_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreparedCorpus(
        prepared_dir=prepared_dir,
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split_paths=split_paths,
        manifest=manifest,
    )


def load_prepared_corpus(prepared_dir: str | Path) -> PreparedCorpus:
    base_dir = Path(prepared_dir).resolve()
    manifest_path = base_dir / "corpus_manifest.json"
    vocab_path = base_dir / "vocab.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_paths = {
        split: Path(stats["encoded_path"]).resolve()
        for split, stats in manifest.get("splits", {}).items()
    }
    return PreparedCorpus(
        prepared_dir=base_dir,
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        split_paths=split_paths,
        manifest=manifest,
    )


def load_encoded_split(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_vocabulary(prepared: PreparedCorpus) -> CharacterVocabulary:
    return CharacterVocabulary.load(prepared.vocab_path)
