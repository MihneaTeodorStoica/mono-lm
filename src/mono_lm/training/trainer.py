from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .corpus import load_encoded_split, load_vocabulary, prepare_corpus
from .generation import generate_text_from_model, latest_checkpoint_path, select_device
from .model import MonoLMModel, TransformerConfig


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path
    prepared_dir: Path
    final_step: int
    best_validation_loss: float


def train_model(
    config: TrainingConfig,
    resume: str | Path | None = None,
    force_prepare: bool = False,
) -> TrainingResult:
    prepared = prepare_corpus(config, force=force_prepare)
    vocabulary = load_vocabulary(prepared)
    train_data = load_encoded_split(prepared.split_paths["train"])
    validation_data = load_encoded_split(prepared.split_paths["validation"])
    _ensure_split_lengths(train_data, validation_data, config.model.context_length)

    run_dir = config.run.output_dir
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config_path, run_dir / "config.used.toml")

    device = select_device(config.run.device)
    torch.manual_seed(config.run.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.run.seed)
    rng = np.random.default_rng(config.run.seed)

    model_config = TransformerConfig(
        vocab_size=vocabulary.size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        ffw_multiplier=config.model.ffw_multiplier,
        dropout=config.model.dropout,
        bias=config.model.bias,
    )
    base_model = MonoLMModel(model_config).to(device)
    training_model: nn.Module = base_model
    if config.run.compile and hasattr(torch, "compile"):
        training_model = torch.compile(base_model)

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    start_step = 0
    best_validation_loss = float("inf")
    if resume is not None:
        checkpoint_path = _resolve_resume_checkpoint(resume, run_dir)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        _restore_rng_state(checkpoint, rng, device)

    metrics_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = checkpoints_dir / "best.pt"
    latest_path = checkpoints_dir / "latest.pt"
    start_time = time.time()

    for step in range(start_step + 1, config.training.max_steps + 1):
        training_model.train()
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = _sample_batch(
            train_data,
            batch_size=config.training.batch_size,
            context_length=config.model.context_length,
            rng=rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = training_model(inputs, targets)
        if loss is None:
            raise RuntimeError("Training loss unexpectedly resolved to None.")
        loss.backward()
        if config.optimizer.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.optimizer.grad_clip_norm)
        optimizer.step()

        if step == 1 or step % config.training.log_interval == 0:
            elapsed = max(1e-6, time.time() - start_time)
            chars_processed = step * config.training.batch_size * config.model.context_length
            chars_per_second = chars_processed / elapsed
            print(
                f"step={step} train_loss={loss.item():.4f} "
                f"lr={learning_rate:.6f} chars_per_sec={chars_per_second:.1f}"
            )

        should_evaluate = step == config.training.max_steps or step % config.training.eval_interval == 0
        should_checkpoint = step == config.training.max_steps or step % config.training.checkpoint_interval == 0
        should_sample = step == config.training.max_steps or step % config.training.sample_interval == 0

        if should_evaluate:
            metrics = evaluate_model(
                model=training_model,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=config.training.batch_size,
                context_length=config.model.context_length,
                eval_batches=config.training.eval_batches,
                rng=rng,
                device=device,
            )
            metrics_row = {
                "step": step,
                "learning_rate": learning_rate,
                **metrics,
            }
            _append_jsonl(metrics_path, metrics_row)
            _write_run_summary(
                run_dir / "run_summary.json",
                config=config,
                prepared_dir=prepared.prepared_dir,
                latest_checkpoint_path=latest_path,
                best_checkpoint_path=best_checkpoint_path,
                metrics=metrics_row,
            )
            print(
                f"eval step={step} train_loss={metrics['train_loss']:.4f} "
                f"validation_loss={metrics['validation_loss']:.4f} "
                f"validation_bpc={metrics['validation_bpc']:.4f}"
            )

            if metrics["validation_loss"] < best_validation_loss:
                best_validation_loss = float(metrics["validation_loss"])
                _save_checkpoint(
                    path=best_checkpoint_path,
                    step=step,
                    best_validation_loss=best_validation_loss,
                    base_model=base_model,
                    optimizer=optimizer,
                    vocabulary=vocabulary,
                    training_config=config,
                    model_config=model_config,
                    rng=rng,
                    device=device,
                )

            _save_checkpoint(
                path=latest_path,
                step=step,
                best_validation_loss=best_validation_loss,
                base_model=base_model,
                optimizer=optimizer,
                vocabulary=vocabulary,
                training_config=config,
                model_config=model_config,
                rng=rng,
                device=device,
            )

        if should_checkpoint:
            checkpoint_path = checkpoints_dir / f"step_{step:07d}.pt"
            _save_checkpoint(
                path=checkpoint_path,
                step=step,
                best_validation_loss=best_validation_loss,
                base_model=base_model,
                optimizer=optimizer,
                vocabulary=vocabulary,
                training_config=config,
                model_config=model_config,
                rng=rng,
                device=device,
            )
            _save_checkpoint(
                path=latest_path,
                step=step,
                best_validation_loss=best_validation_loss,
                base_model=base_model,
                optimizer=optimizer,
                vocabulary=vocabulary,
                training_config=config,
                model_config=model_config,
                rng=rng,
                device=device,
            )

        if should_sample:
            base_model.eval()
            sample_text = generate_text_from_model(
                model=base_model,
                vocabulary=vocabulary,
                prompt=config.generation.prompt,
                max_new_chars=config.generation.max_new_chars,
                temperature=config.generation.temperature,
                top_k=config.generation.top_k,
                device=device,
            )
            sample_path = samples_dir / f"step_{step:07d}.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            print(f"sample step={step}: {sample_path}")

    return TrainingResult(
        run_dir=run_dir,
        latest_checkpoint_path=latest_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        prepared_dir=prepared.prepared_dir,
        final_step=config.training.max_steps,
        best_validation_loss=best_validation_loss,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    train_data: np.ndarray,
    validation_data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {"train": [], "validation": []}
    for split_name, data in {"train": train_data, "validation": validation_data}.items():
        for _ in range(eval_batches):
            inputs, targets = _sample_batch(
                data,
                batch_size=batch_size,
                context_length=context_length,
                rng=rng,
                device=device,
            )
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Evaluation loss unexpectedly resolved to None.")
            losses[split_name].append(float(loss.item()))

    train_loss = float(np.mean(losses["train"]))
    validation_loss = float(np.mean(losses["validation"]))
    return {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_bpc": train_loss / math.log(2),
        "validation_bpc": validation_loss / math.log(2),
    }


def _sample_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = int(data.shape[0]) - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"Split length {int(data.shape[0])} is shorter than required context length {context_length + 1}."
        )
    starts = rng.integers(0, max_start + 1, size=batch_size)
    batch_inputs = np.stack([data[start : start + context_length] for start in starts]).astype(np.int64)
    batch_targets = np.stack([data[start + 1 : start + context_length + 1] for start in starts]).astype(np.int64)
    inputs = torch.from_numpy(batch_inputs).to(device)
    targets = torch.from_numpy(batch_targets).to(device)
    return inputs, targets


def _learning_rate(step: int, config: TrainingConfig) -> float:
    max_lr = config.optimizer.learning_rate
    min_lr = config.optimizer.min_learning_rate
    warmup_steps = max(1, config.training.warmup_steps)
    if step <= warmup_steps:
        return max_lr * step / warmup_steps
    if config.training.max_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, config.training.max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + cosine * (max_lr - min_lr)


def _save_checkpoint(
    path: Path,
    step: int,
    best_validation_loss: float,
    base_model: MonoLMModel,
    optimizer: torch.optim.Optimizer,
    vocabulary,
    training_config: TrainingConfig,
    model_config: TransformerConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    payload = {
        "step": step,
        "best_validation_loss": best_validation_loss,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocabulary": vocabulary.to_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": rng.bit_generator.state,
        },
    }
    if device.type == "cuda":
        payload["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _restore_rng_state(checkpoint: dict[str, Any], rng: np.random.Generator, device: torch.device) -> None:
    rng_state = checkpoint.get("rng_state", {})
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if "numpy" in rng_state:
        rng.bit_generator.state = rng_state["numpy"]
    if device.type == "cuda" and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def _write_run_summary(
    path: Path,
    config: TrainingConfig,
    prepared_dir: Path,
    latest_checkpoint_path: Path,
    best_checkpoint_path: Path,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "run_name": config.run.name,
        "config_path": str(config.config_path),
        "prepared_dir": str(prepared_dir),
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "latest_metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_split_lengths(train_data: np.ndarray, validation_data: np.ndarray, context_length: int) -> None:
    minimum_length = context_length + 1
    if int(train_data.shape[0]) < minimum_length:
        raise ValueError(f"Train split must contain at least {minimum_length} characters.")
    if int(validation_data.shape[0]) < minimum_length:
        raise ValueError(f"Validation split must contain at least {minimum_length} characters.")


def _resolve_resume_checkpoint(resume: str | Path, run_dir: Path) -> Path:
    if str(resume) == "latest":
        return latest_checkpoint_path(run_dir)
    return Path(resume).resolve()
