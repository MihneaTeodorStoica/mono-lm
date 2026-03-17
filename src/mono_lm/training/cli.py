from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_training_config
from .corpus import prepare_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mono-lm character-level training workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="prepare encoded character data and vocabulary")
    prepare_parser.add_argument("--config", required=True, help="Path to a training TOML config")
    prepare_parser.add_argument("--force", action="store_true", help="Rebuild prepared artifacts even if they exist")

    train_parser = subparsers.add_parser("train", help="train a character-level mono-lm baseline")
    train_parser.add_argument("--config", required=True, help="Path to a training TOML config")
    train_parser.add_argument(
        "--resume",
        help="Checkpoint path to resume from, or 'latest' to resume from the run directory",
    )
    train_parser.add_argument("--force-prepare", action="store_true", help="Rebuild prepared data before training")

    sample_parser = subparsers.add_parser("sample", help="generate text from a trained checkpoint")
    sample_parser.add_argument("--checkpoint", help="Path to a checkpoint file")
    sample_parser.add_argument("--run-dir", help="Run directory; resolves to the latest checkpoint automatically")
    sample_parser.add_argument("--prompt", required=True, help="Prompt text to continue from")
    sample_parser.add_argument("--max-new-chars", type=int, default=400, help="Maximum number of generated chars")
    sample_parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    sample_parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff")
    sample_parser.add_argument("--device", default="auto", help="Device override such as cpu, cuda, or auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _run_prepare(args.config, force=args.force)
    if args.command == "train":
        return _run_train(args.config, resume=args.resume, force_prepare=args.force_prepare)
    if args.command == "sample":
        return _run_sample(
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            prompt=args.prompt,
            max_new_chars=args.max_new_chars,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
    return 1


def _run_prepare(config_path: str, force: bool) -> int:
    config = load_training_config(config_path)
    prepared = prepare_corpus(config, force=force)
    print(f"Prepared corpus in {prepared.prepared_dir}")
    print(f"Manifest: {prepared.manifest_path}")
    print(f"Vocabulary: {prepared.vocab_path}")
    for split, path in prepared.split_paths.items():
        print(f"{split.title()} encoded: {path}")
    return 0


def _run_train(config_path: str, resume: str | None, force_prepare: bool) -> int:
    try:
        from .trainer import train_model
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Training requires PyTorch. Install a platform-appropriate torch build in .venv before running this command."
            ) from exc
        raise

    config = load_training_config(config_path)
    result = train_model(config, resume=resume, force_prepare=force_prepare)
    print(f"Run directory: {result.run_dir}")
    print(f"Latest checkpoint: {result.latest_checkpoint_path}")
    print(f"Best checkpoint: {result.best_checkpoint_path}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _run_sample(
    checkpoint: str | None,
    run_dir: str | None,
    prompt: str,
    max_new_chars: int,
    temperature: float,
    top_k: int,
    device: str,
) -> int:
    try:
        from .generation import generate_from_checkpoint, latest_checkpoint_path
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Sampling requires PyTorch. Install a platform-appropriate torch build in .venv before running this command."
            ) from exc
        raise

    if not checkpoint and not run_dir:
        raise SystemExit("sample requires either --checkpoint or --run-dir")
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else latest_checkpoint_path(run_dir or "")
    text = generate_from_checkpoint(
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        max_new_chars=max_new_chars,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
