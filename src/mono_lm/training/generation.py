from __future__ import annotations

from pathlib import Path

import torch

from .model import MonoLMModel, TransformerConfig
from .vocabulary import CharacterVocabulary


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def generate_text_from_model(
    model: MonoLMModel,
    vocabulary: CharacterVocabulary,
    prompt: str,
    max_new_chars: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    if not prompt:
        raise ValueError("Prompt must be non-empty for generation.")
    prompt_tokens = torch.tensor([vocabulary.encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_chars,
        temperature=temperature,
        top_k=top_k,
    )
    return vocabulary.decode(output[0].tolist())


def load_checkpoint_bundle(checkpoint_path: str | Path, device: str | torch.device = "auto") -> tuple[MonoLMModel, CharacterVocabulary, dict, torch.device]:
    resolved_device = select_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(Path(checkpoint_path).resolve(), map_location=resolved_device, weights_only=False)
    model_config = TransformerConfig(**checkpoint["model_config"])
    model = MonoLMModel(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    vocabulary = CharacterVocabulary.from_dict(checkpoint["vocabulary"])
    return model, vocabulary, checkpoint, resolved_device


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    prompt: str,
    max_new_chars: int,
    temperature: float,
    top_k: int | None,
    device: str = "auto",
) -> str:
    model, vocabulary, _, resolved_device = load_checkpoint_bundle(checkpoint_path, device=device)
    return generate_text_from_model(
        model=model,
        vocabulary=vocabulary,
        prompt=prompt,
        max_new_chars=max_new_chars,
        temperature=temperature,
        top_k=top_k,
        device=resolved_device,
    )


def latest_checkpoint_path(run_dir: str | Path) -> Path:
    checkpoints_dir = Path(run_dir).resolve() / "checkpoints"
    latest = checkpoints_dir / "latest.pt"
    if latest.exists():
        return latest

    candidates = sorted(checkpoints_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {checkpoints_dir}")
    return candidates[-1]
