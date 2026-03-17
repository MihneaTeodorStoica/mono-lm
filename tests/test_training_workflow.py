from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mono_lm.dataset_pipeline.config import load_config
from mono_lm.dataset_pipeline.pipeline import DatasetPipeline
from mono_lm.training.config import load_training_config
from mono_lm.training.corpus import prepare_corpus
from mono_lm.training.generation import generate_from_checkpoint
from mono_lm.training.trainer import train_model


class TrainingWorkflowTests(unittest.TestCase):
    def test_prepare_train_and_generate(self) -> None:
        dataset_config = load_config(REPO_ROOT / "configs/dataset/demo.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            dataset_config.pipeline.output_dir = temp_root / "artifacts"
            DatasetPipeline(dataset_config).build()

            training_config_path = temp_root / "tiny_training.toml"
            training_config_path.write_text(
                textwrap.dedent(
                    f"""
                    [run]
                    name = "test-run"
                    output_dir = "{(temp_root / 'runs' / 'test-run').as_posix()}"
                    seed = 5
                    device = "cpu"
                    compile = false

                    [data]
                    artifact_dir = "{(temp_root / 'artifacts').as_posix()}"
                    prepared_dir = "{(temp_root / 'prepared').as_posix()}"
                    reuse_prepared = true

                    [model]
                    context_length = 48
                    d_model = 48
                    num_layers = 2
                    num_heads = 4
                    ffw_multiplier = 2.0
                    dropout = 0.0
                    bias = true

                    [optimizer]
                    learning_rate = 0.001
                    min_learning_rate = 0.0001
                    weight_decay = 0.0
                    beta1 = 0.9
                    beta2 = 0.95
                    grad_clip_norm = 1.0

                    [training]
                    batch_size = 4
                    max_steps = 4
                    warmup_steps = 1
                    eval_interval = 2
                    eval_batches = 1
                    log_interval = 1
                    checkpoint_interval = 2
                    sample_interval = 2

                    [generation]
                    prompt = "Question: Why"
                    max_new_chars = 24
                    temperature = 0.8
                    top_k = 8
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            training_config = load_training_config(training_config_path)
            prepared = prepare_corpus(training_config, force=True)
            self.assertTrue(prepared.vocab_path.exists())
            self.assertTrue(prepared.split_paths["train"].exists())
            self.assertTrue(prepared.split_paths["validation"].exists())

            result = train_model(training_config)
            self.assertTrue(result.latest_checkpoint_path.exists())
            self.assertTrue(result.best_checkpoint_path.exists())
            self.assertTrue(result.metrics_path.exists())
            sample_files = sorted((result.run_dir / "samples").glob("*.txt"))
            self.assertTrue(sample_files)

            generated = generate_from_checkpoint(
                checkpoint_path=result.best_checkpoint_path,
                prompt="Question: Why",
                max_new_chars=24,
                temperature=0.8,
                top_k=8,
                device="cpu",
            )
            self.assertTrue(generated.startswith("Question: Why"))
            self.assertGreater(len(generated), len("Question: Why"))


if __name__ == "__main__":
    unittest.main()
