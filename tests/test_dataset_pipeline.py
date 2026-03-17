from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mono_lm.dataset_pipeline.cleaning import normalize_raw_sample
from mono_lm.dataset_pipeline.config import FormattingConfig, load_config
from mono_lm.dataset_pipeline.filters import assess_sample
from mono_lm.dataset_pipeline.models import DialogueTurn, RawSample
from mono_lm.dataset_pipeline.pipeline import DatasetPipeline


class CleaningTests(unittest.TestCase):
    def test_dialogue_roles_and_markup_are_normalized(self) -> None:
        raw = RawSample(
            sample_id="sample-1",
            source_name="demo",
            family="dialogue",
            layout="dialogue",
            local_id="1",
            origin_path="memory",
            split_group="demo:1",
            turns=[
                DialogueTurn(role="human", text="Hi <b>there</b>"),
                DialogueTurn(role="assistant", text="Use [[gentle|gentle]] wording."),
            ],
        )
        processed = normalize_raw_sample(raw, load_config(REPO_ROOT / "configs/dataset/demo.toml").cleaning, FormattingConfig())
        self.assertEqual(processed.turns[0].role, "User")
        self.assertIn("gentle", processed.turns[1].text)
        self.assertNotIn("<b>", processed.formatted_text)


class QualityTests(unittest.TestCase):
    def test_repetition_is_rejected(self) -> None:
        config = load_config(REPO_ROOT / "configs/dataset/demo.toml")
        raw = RawSample(
            sample_id="sample-2",
            source_name="demo",
            family="dialogue",
            layout="dialogue",
            local_id="2",
            origin_path="memory",
            split_group="demo:2",
            turns=[
                DialogueTurn(role="user", text="okay okay okay okay okay okay okay okay"),
                DialogueTurn(role="assistant", text="yes yes yes yes yes yes yes yes"),
            ],
        )
        processed = normalize_raw_sample(raw, config.cleaning, config.formatting)
        outcome = assess_sample(processed, config.quality)
        self.assertFalse(outcome.accepted)
        self.assertIn(outcome.reason, {"repetition_heavy", "too_short"})


class PipelineBuildTests(unittest.TestCase):
    def test_demo_pipeline_builds_artifacts(self) -> None:
        config = load_config(REPO_ROOT / "configs/dataset/demo.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            config.pipeline.output_dir = Path(tmpdir) / "artifacts"
            outputs = DatasetPipeline(config).build()
            self.assertTrue(outputs.report_json_path.exists())
            self.assertTrue(outputs.report_markdown_path.exists())
            self.assertTrue(outputs.train_path.exists())
            report_text = outputs.report_markdown_path.read_text(encoding="utf-8")
            self.assertIn("Stage counts", report_text)
            inspection_path = config.pipeline.output_dir / "reports" / "inspection.md"
            self.assertTrue(inspection_path.exists())


if __name__ == "__main__":
    unittest.main()
