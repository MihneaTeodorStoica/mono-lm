from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import tomllib


def _discover_project_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return config_path.parent


def _coerce_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


@dataclass(slots=True)
class RunConfig:
    name: str
    output_dir: Path
    seed: int = 13
    device: str = "auto"
    compile: bool = False


@dataclass(slots=True)
class DataConfig:
    prepared_dir: Path
    artifact_dir: Path | None = None
    train_text_path: Path | None = None
    validation_text_path: Path | None = None
    test_text_path: Path | None = None
    reuse_prepared: bool = True


@dataclass(slots=True)
class ModelConfig:
    context_length: int = 256
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffw_multiplier: float = 4.0
    dropout: float = 0.1
    bias: bool = True


@dataclass(slots=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0


@dataclass(slots=True)
class TrainingLoopConfig:
    batch_size: int = 32
    max_steps: int = 5000
    warmup_steps: int = 200
    eval_interval: int = 200
    eval_batches: int = 20
    log_interval: int = 20
    checkpoint_interval: int = 500
    sample_interval: int = 200


@dataclass(slots=True)
class GenerationConfig:
    prompt: str = "Question: Why does"
    max_new_chars: int = 400
    temperature: float = 0.9
    top_k: int = 40


@dataclass(slots=True)
class TrainingConfig:
    root_dir: Path
    config_path: Path
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingLoopConfig
    generation: GenerationConfig

    def resolve_text_paths(self) -> dict[str, Path | None]:
        if self.data.artifact_dir is not None:
            base_dir = self.data.artifact_dir / "final"
            train_path = self.data.train_text_path or (base_dir / "train.txt")
            validation_path = self.data.validation_text_path or (base_dir / "validation.txt")
            test_path = self.data.test_text_path or (base_dir / "test.txt")
        else:
            train_path = self.data.train_text_path
            validation_path = self.data.validation_text_path
            test_path = self.data.test_text_path

        if train_path is None or validation_path is None:
            raise ValueError("Training config must define either data.artifact_dir or explicit train/validation text paths.")
        return {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _stringify_paths(payload)


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).resolve()
    root_dir = _discover_project_root(config_path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    run_table = raw.get("run", {})
    data_table = raw.get("data", {})

    run = RunConfig(
        name=str(run_table.get("name", config_path.stem)),
        output_dir=_coerce_path(root_dir, str(run_table.get("output_dir", f"data/runs/{config_path.stem}"))),
        seed=int(run_table.get("seed", 13)),
        device=str(run_table.get("device", "auto")),
        compile=bool(run_table.get("compile", False)),
    )
    data = DataConfig(
        prepared_dir=_coerce_path(root_dir, str(data_table.get("prepared_dir", f"data/prepared/{config_path.stem}"))),
        artifact_dir=_coerce_path(root_dir, data_table.get("artifact_dir")),
        train_text_path=_coerce_path(root_dir, data_table.get("train_text_path")),
        validation_text_path=_coerce_path(root_dir, data_table.get("validation_text_path")),
        test_text_path=_coerce_path(root_dir, data_table.get("test_text_path")),
        reuse_prepared=bool(data_table.get("reuse_prepared", True)),
    )
    model = ModelConfig(**raw.get("model", {}))
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))
    training = TrainingLoopConfig(**raw.get("training", {}))
    generation = GenerationConfig(**raw.get("generation", {}))

    if model.context_length < 2:
        raise ValueError("model.context_length must be at least 2.")
    if model.d_model % model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads.")
    if training.batch_size < 1:
        raise ValueError("training.batch_size must be positive.")
    if training.max_steps < 1:
        raise ValueError("training.max_steps must be positive.")
    if training.eval_interval < 1 or training.eval_batches < 1:
        raise ValueError("training.eval_interval and training.eval_batches must be positive.")
    if training.log_interval < 1 or training.checkpoint_interval < 1 or training.sample_interval < 1:
        raise ValueError("training log/checkpoint/sample intervals must be positive.")

    return TrainingConfig(
        root_dir=root_dir,
        config_path=config_path,
        run=run,
        data=data,
        model=model,
        optimizer=optimizer,
        training=training,
        generation=generation,
    )


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_paths(item) for item in value]
    return value
