# mono-lm

`mono-lm` now includes both sides of the first project workflow:

- a local corpus-building pipeline for character-level language model data,
- a first end-to-end character-level Transformer training stack.

The project stays intentionally local-first. The data tooling is built around reproducible corpus assembly from local files, not scraping, and the training layer consumes the exported corpus artifacts directly.

## What The Repo Can Do

### Corpus building

The dataset pipeline supports:

- cleaning and normalization,
- low-signal filtering,
- exact and near-duplicate removal,
- configurable family/source mixtures,
- deterministic train/validation/test splits,
- modular source catalogs for larger local inventories,
- source/family composition reports with character-share breakdowns,
- dominance warnings when one family or source takes too much of the final corpus,
- representative cleaned sample inspection before training.

### Training

The training workflow supports:

- character vocabulary building and persistence,
- encoded corpus preparation from exported dataset artifacts,
- a decoder-only autoregressive Transformer baseline,
- training with validation during the run,
- checkpoints for resume/restart,
- periodic sample generation,
- standalone text generation from saved checkpoints.

## Project Layout

```text
configs/
  dataset/
    demo.toml
    demo_catalog.toml
    default.toml
    large_local.toml
    sources/
  training/
    tiny.toml
    baseline.toml
examples/raw/
src/mono_lm/
  dataset_pipeline/
  training/
tests/
```

## Installation

Install into the existing virtual environment:

```bash
./.venv/bin/pip install -e .
```

For the training commands, install a platform-appropriate PyTorch build into the same environment. For a CPU-only Linux setup, for example:

```bash
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Quick Start

Build the runnable demo corpus:

```bash
./.venv/bin/mono-lm-dataset build --config configs/dataset/demo.toml
```

Inspect the configured demo inventory through the catalog-based workflow:

```bash
./.venv/bin/mono-lm-dataset inventory --config configs/dataset/demo_catalog.toml
```

Inspect cleaned selected samples:

```bash
./.venv/bin/mono-lm-dataset inspect --artifact-dir data/artifacts/demo --stage selected --limit 6
```

Prepare encoded training data:

```bash
./.venv/bin/mono-lm-train prepare --config configs/training/tiny.toml
```

Train the first baseline:

```bash
./.venv/bin/mono-lm-train train --config configs/training/tiny.toml
```

Generate from the latest checkpoint:

```bash
./.venv/bin/mono-lm-train sample --run-dir data/runs/tiny-demo --prompt "Question: Why does a battery get warm?"
```

## Dataset Workflow

### Supported source kinds

- `text_dir`: directory of local text/markdown-like files
- `text_file`: single local text file
- `jsonl`: one record per line
- `csv`: tabular local corpus

### Supported families

- `expository`
- `qa`
- `dialogue`
- `prose`

### Supported layouts

- `document`
- `qa`
- `dialogue`

### Config structure

Dataset configs are TOML files with these main sections:

- `source_files`: optional list of TOML source catalogs to include
- `[pipeline]`: output location, target character budget, preview sizing
- `[cleaning]`: normalization and markup cleanup
- `[quality]`: rejection thresholds
- `[dedup]`: exact/near duplicate controls
- `[formatting]`: final sample structure and separators
- `[split]`: train/validation/test ratios
- `[mixture]`: family weights, source weights, minimum family presence
- `[reporting]`: source dominance thresholds and preview settings
- `[[sources]]`: inline source entries when you want everything in one file

### Modular source catalogs

For larger local builds, keep the top-level recipe small and move source lists into `source_files`.

The repo includes:

- `configs/dataset/demo_catalog.toml`: runnable catalog-based example using the bundled demo corpora
- `configs/dataset/large_local.toml`: larger local-build template
- `configs/dataset/sources/*.toml`: example source catalog fragments by family

This makes it easy to grow from a handful of demo datasets to dozens of local corpora without turning one config into an audit nightmare.

### Reports and artifacts

A dataset build writes:

```text
data/artifacts/<run-name>/
  config.used.toml
  manifest.json
  final/
    train.txt
    validation.txt
    test.txt
  intermediate/
    raw_samples.jsonl
    normalized_samples.jsonl
    deduped_samples.jsonl
    selected_samples.jsonl
    rejected_samples.jsonl
  splits/
    train.jsonl
    validation.jsonl
    test.jsonl
  reports/
    report.json
    report.md
    character_inventory.tsv
    family_breakdown.tsv
    source_breakdown.tsv
    inspection.md
```

The reporting layer is meant to answer the practical corpus questions quickly:

- what actually made it into the final corpus,
- which families dominate the character count,
- which individual sources dominate,
- what the cleaned samples look like,
- why records were rejected.

## Adding More Local Corpora

1. Put the corpus somewhere under your local `data/raw/` tree.
2. Add or edit a source catalog in `configs/dataset/sources/`.
3. Point a build config at that catalog through `source_files`.
4. Run `mono-lm-dataset inventory` to confirm the source mix.
5. Run `mono-lm-dataset build`.
6. Review `report.md`, `source_breakdown.tsv`, and `inspection.md` before training.

If you need a new file convention, extend the loaders in [sources.py](/home/mihnea/Programming/GitHub/mono-lm/src/mono_lm/dataset_pipeline/sources.py) and keep the current `RawSample -> ProcessedSample -> selected split artifacts` flow intact.

## Training Workflow

The first model stack is intentionally a clean baseline:

- character-level vocabulary
- decoder-only Transformer
- next-character training objective
- validation loss and bits-per-character tracking
- checkpoints plus resume
- prompt-based sampling

### Training config structure

Training configs are TOML files with:

- `[run]`: run directory, seed, device
- `[data]`: input artifact directory or explicit split paths, prepared output directory
- `[model]`: context length and Transformer size
- `[optimizer]`: AdamW and learning-rate schedule settings
- `[training]`: batch size, steps, eval/checkpoint/sample cadence
- `[generation]`: default validation/sample prompt and sampling controls

### Prepared artifacts

`mono-lm-train prepare` writes:

```text
data/prepared/<run-name>/
  corpus_manifest.json
  vocab.json
  train.npy
  validation.npy
  test.npy
```

### Training outputs

`mono-lm-train train` writes:

```text
data/runs/<run-name>/
  config.used.toml
  metrics.jsonl
  run_summary.json
  checkpoints/
    latest.pt
    best.pt
    step_0000500.pt
  samples/
    step_0000500.txt
```

### Included training presets

- `configs/training/tiny.toml`: fast sanity-check run on the demo corpus
- `configs/training/baseline.toml`: stronger baseline intended for a larger local build

## Tests

Run the test suite with:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Notes

- `data/raw/`, `data/artifacts/`, `data/prepared/`, and `data/runs/` are ignored so you can iterate locally without polluting the repo.
- If you rebuild a corpus and want to regenerate encoded training arrays, rerun `mono-lm-train prepare --force`.
