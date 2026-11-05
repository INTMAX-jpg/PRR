# PRR paper materials

Materials accompanying **Surface Recovery Is Not State Recovery: Pressure–Recovery–Relapse in Multi-Turn LLMs**.

This repository provides the CW generator and behavior runners, final-rebuttal probability-readout and intervention implementations, and the numerical artifacts listed in [MATERIALS.csv](MATERIALS.csv). That inventory identifies the files and validation supplied for each experiment group.

## Quick start: verify the supplied results

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e .
python scripts/verify_results.py --out outputs/verification.json
```

This command checks file hashes and recomputes held-out readout fidelity, natural-recovery quantities, GSM8K paired intervention effects, pressure-family estimates, and newer-model filtering counts. It uses the supplied files and runs on CPU.

## CW dataset and behavior

The companion CW dataset contains 2,000 base items and 10,000 matched trajectory templates. The generator reproduces the frozen construction and record order:

```bash
python scripts/generate_cw.py --out outputs/cw_2000.jsonl
```

Use `--verify-against PATH_TO_CW/original/PRR-ClosedWorld-2K.jsonl` to check content and order against the companion dataset. The configurations in `configs/cw_2000/` specify 2,000 base items and all five trajectory types. They are prepared from the reported paper settings. The behavior runner processes the supplied records, reports their actual counts, and calculates outputs from model predictions.

## Experiment code

Install GPU dependencies when running language models:

```bash
python -m pip install -r requirements-gpu.txt
python scripts/run_behavior.py --config configs/cw_2000/qwen7b.json --dataset outputs/cw_2000.jsonl --outdir runs/cw_qwen7b
python scripts/analyze_behavior.py --metadata runs/cw_qwen7b/metadata.csv --out outputs/cw_qwen7b_metrics.json --rv-weighting directed-pair
```

The final probability readouts translate intermediate activations into answer-candidate probabilities. GSM8K diagnostics use L22 replacement and L24 readout. Weak-only preparation and paired analysis are included. The pressure-template runner computes surface behavior and preserves each saved T0–T4 history when branching at T5.

See [REPRODUCING.md](docs/REPRODUCING.md) for model acquisition, source datasets, complete command sequences, and environment notes. Workflows print their commands by default; `--execute` runs them in a new output directory.

New GSM behavior summaries use deterministic, sorted-item resampling and versioned filenames. The paper-reported intervals remain in the frozen result files; see [GSM_BEHAVIOR_INTERVALS.md](docs/GSM_BEHAVIOR_INTERVALS.md).

## Figures

`figures/` contains the two supplied rebuttal figures. To render the corresponding statistics:

```bash
python scripts/plot_rebuttal.py --outdir outputs/rebuttal_figures
```

The plotting code requires Times New Roman. Supply `--font-file /path/to/times.ttf` when the font is not installed. Fonts are supplied by the user’s operating system or font installation.

## Organization

- `src/prr_behavior/`: behavior runtime and the original CW generator.
- `src/latent_pressure/`: final probability-readout and intervention support.
- `scripts/`: preparation, execution, analysis, verification, and plotting.
- `configs/`: portable model and experiment settings.
- `data/`: frozen synthetic inputs, cohort information, and causal inputs.
- `results/`: the artifacts identified in `MATERIALS.csv`.

## Validation and citation

The submitted draft is checked using CPU artifact recomputation, dataset construction checks, a source/dependency audit, and behavior-runner parity checks. Model inference is a separate execution step; the packaging checks do not assert a fresh GPU reproduction of every reported estimate.

See [CITATION.cff](CITATION.cff). License selection is pending confirmation by the author team; see [LICENSE_STATUS.txt](LICENSE_STATUS.txt).
