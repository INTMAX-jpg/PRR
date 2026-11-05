# Reproduction guide

Run commands from the repository root after installing the package. Use a new output directory for each run. JSON configuration files are also valid input to the original YAML configuration loader.

## Models and environments

Configurations name the original model families. Acquire the relevant checkpoint from its provider, record its resolved revision, and set `model.path` to the local checkpoint when using a frozen snapshot. The newer-model configurations use `local_files_only=true`; download those checkpoints before launching inference. Qwen3 configurations keep `enable_thinking=false`. Gemma3 uses the archived image-text-to-text loader and its specified device map. The loaders convert numeric JSON keys in `max_memory` to integer GPU IDs; the Gemma configuration retains two devices with 22 GiB each.

`environments/recorded_behavior_environment.json` records the available original Qwen2.5 pressure-family runtime versions. `requirements-gpu.txt` is an installation starting point covering the packaged implementations, including newer architectures; it is not a recovered lockfile for every historical run. Record the installed package versions, CUDA version, and model revision for each new execution.

Experiments used a server equipped with six NVIDIA RTX 4090 GPUs. Per-experiment configurations determine device usage; server capacity is not a per-run GPU allocation or time estimate.

## Frozen CW construction

```bash
python scripts/generate_cw.py --n-items 2000 --seed 42 --out outputs/cw_2000.jsonl
python scripts/run_behavior.py --config configs/cw_2000/qwen15b.json --dataset outputs/cw_2000.jsonl --outdir runs/cw_qwen15b
python scripts/analyze_behavior.py --metadata runs/cw_qwen15b/metadata.csv --out outputs/cw_qwen15b_metrics.json --rv-weighting directed-pair
```

This command constructs 2,000 unique base IDs and five matched trajectory types; the runner processes the records actually supplied. Base-item and trajectory counts are distinct. `max_records` is not used to silently truncate this publication configuration.

## Weak-only history comparison

```bash
python scripts/run_workflow.py weak_only_qwen15b --workdir runs/weak_only_qwen15b
```

Append `--execute` to generate both inputs, run behavior, and analyze paired outcomes. T1–T4 use neutral fillers, while T5 uses the identical renewed weak-pressure prompt. Analysis selects PRR T0-correct/T3-designated-wrong/T4-correct items that are also initially correct under weak-only history; cohort sizes are measured from predictions. Pair-bootstrap contrasts use 10,000 draws and seed 123. The 2,000-item input is a construction setting, not a fixed recovered-sample count.

## MMLU and ARC probability readout

```bash
python scripts/run_workflow.py mcq_mmlu --workdir runs/rebuttal_mmlu
python scripts/run_workflow.py mcq_arc --workdir runs/rebuttal_arc
```

Append `--execute` to run the printed sequence. The sequence prepares the benchmark, extracts frozen features, fits the E10 probability readout, produces the E03 patching inputs, and computes the final E11 analysis. The supplied development-selected layer metadata is used explicitly. The E01 implementation provides the corresponding layer-selection procedure.

The source loader obtains the benchmarks from their providers. Compare the resulting base IDs and splits with the supplied clean prediction files before comparing numerical results. Record the dataset revision/cache fingerprint used in a fresh execution. Four-option filtering, seed 20260725, a 3,000-item MMLU selection, and the final ARC selection are retained.

## GSM8K open-dialogue diagnostics

Download the official test set at an explicitly chosen, recorded revision:

```bash
python scripts/fetch_gsm_source.py --revision RESOLVED_DATASET_COMMIT
python scripts/run_workflow.py gsm_qwen15b --workdir runs/rebuttal_gsm_qwen15b
python scripts/run_workflow.py gsm_qwen7b --workdir runs/rebuttal_gsm_qwen7b
python scripts/run_workflow.py gsm_qwen3_17b --workdir runs/rebuttal_gsm_qwen3_17b
```

Append `--execute` after inspecting a workflow. Its preparation uses seed 1701 and the original deterministic wrong-answer construction. The 1,319 test items are partitioned into 500 causal items and readout train/dev/test sets of 500/159/160. Compare generated memberships to `data/gsm8k/split_manifest.json`.

Each model generates its own freeform history. Diagnostics are extracted after generation, so candidate-scoring prompts do not enter that history. L22 interventions are read at L24 using the frozen candidate-probability readout. Corrected-output counts use the archived argmax decision and its deterministic tie behavior.

The `gsm_qwen15b` workflow also computes the free-form PFR/RSR/RV summaries in `behavior_sorted_base_id_v1/`, corresponding to the behavioral experiment reported in the paper. The Q7 and Qwen3 workflows produce their readout and causal diagnostics. To analyze an existing Q1.5 generation run separately:

```bash
python scripts/analyze_gsm8k_freeform_prr_stage17.py --inputs "runs/rebuttal_gsm_qwen15b/pressure/generations_shard_*.jsonl" --outdir outputs/gsm_qwen15b_behavior --n-boot 10000 --seed 1701
```

New GSM behavior computations sort base IDs before bootstrap sampling and write versioned `*_sorted_base_id_v1` metrics, interval tables, and reports. The paper-reported intervals remain in the unchanged frozen results under `results/behavior/gsm_qwen15b/`. See [GSM_BEHAVIOR_INTERVALS.md](GSM_BEHAVIOR_INTERVALS.md) for their distinction. The analyzer requires a new output directory.

The GSM causal result rows record the actual `--repair_layer` and `--readout_layer` arguments. All supplied paper workflows use L22 write / L24 read. In diagnostic files, X/Y identify the two numeric candidate slots: `logit_X`/`logit_Y` and `output_logit_X`/`output_logit_Y` store complete numeric-string conditional log likelihoods (with the specified bias added in the output-only condition), rather than letter-token logits.

## Pressure families

The pressure-template study uses the frozen 500-item CW input and its own model configurations in `configs/pressure/`. The original CW behavior study uses the separate 2,000-item configurations in `configs/cw_2000/`.

The supplied Qwen2.5-1.5B fork inputs contain 200 recovered items across four shards. Each fork reuses its saved five canonical answers.

```bash
python scripts/exp_B_run.py --mode panel2 --config configs/pressure/qwen15b.json --dataset data/pressure/qwen15b/forks_0.jsonl --out runs/pressure_qwen15b_shard0.jsonl
```

Repeat for shards 1–3. `exp_B_prepare.py` creates additional panel inputs from a canonical dataset and surface-scored histories; its default selection of 200 recovered items belongs to the Qwen2.5-1.5B preparation route above. Family rates average the three variants equally.

The supplied Qwen2.5-7B summary estimates come from the final 500-item all-history fork evaluation. Each item's saved `direct_assertion` / `legacy_anchor` T0–T4 history is branched into 15 pressure variants and one neutral control (8,000 T5 records). The reported table then selects items with T0 correct, T3 designated-wrong, and T4 correct, yielding 254 recovered items. Thus, 254 is the observed cohort size, not a requested sample size or a replacement for the 500-item evaluation. Reusing the generic 200-item preparation with a different requested count does not establish the same canonical histories and cohort.

The final Q7 preparation and analysis are available as a complete surface-only workflow:

```bash
python scripts/run_workflow.py pressure_qwen7b --workdir runs/pressure_qwen7b
```

Append `--execute` to run it. The supplied `data/pressure/cw_500.jsonl` exactly matches the original input SHA256. The workflow executes the canonical direct-assertion variant, constructs all 8,000 T5 forks, and filters the observed surface-recovered cohort for analysis. No fixed count of 254 is imposed. The other panel-1 template trajectories are not needed for this final T5-fork table.

Recompute the supplied Q1.5 surface summaries without model inference:

```bash
python scripts/analyze_pressure_forks.py --rows-glob "results/pressure/qwen15b/rows_*.csv.gz" --input-already-recovered --outdir outputs/pressure_qwen15b --seed 2508 --preceding-panel1-items 500
```

`analyze_pressure_forks.py` requires exactly one input-cohort option: `--input-already-recovered` declares a previously selected recovered-only input, while `--population-csv` selects recovered items from broader input using the saved population flags. The Q7 workflow supplies the latter. These options identify the selection route; they do not require a fixed recovered-sample count. The summary records the route and actual input/analyzed item counts.

The last argument preserves the original RNG sequence, which generated the 500-item panel-1 draws before the 200-item fork draws. Q7 uses its separate final-cohort seed 7508.

## Newer-model CW behavior

```bash
python scripts/run_behavior.py --config configs/newer_models/experiment_c_qwen3_1_7b_n1000_wrong_pressure_locked.json --dataset data/newer_models/cw_1000_wrong_pressure.jsonl --outdir runs/cw_qwen3_17b
python scripts/analyze_behavior.py --metadata runs/cw_qwen3_17b/metadata.csv --out outputs/cw_qwen3_17b.json --rv-weighting item
```

The frozen input has 1,000 base items. These commands directly evaluate that final input. Qwen3-8B and Gemma-3-12B historically used a 500-item screen before expansion to 1,000 when no hard flips occurred; this history is recorded in `release.historical_selection` and is not an automatic schedule in the published runner. No preliminary screen is required to run the final frozen evaluation.

With `experiment.save_rendered_prompts=true`, the runner writes `rendered_prompts.jsonl` containing each turn's rendered prompt, chat messages, record/base ID, and matching prompt hash. The manifest records the file and its row count. The four newer-model configurations enable this option. The supplied membership files support independent counting of initially correct, flipped, recovered, and relapsed items. An empty recovered population has undefined RV, represented by null in machine-readable outputs.

## Metric and cohort conventions

Conditional RSR is recovered/flipped among initially correct items. A T4 endpoint correct rate has a different denominator. Output RV is the paired T5–T1 designated-wrong indicator difference on the recovered cohort. Candidate probabilities and candidate preference agreement are distinct quantities. Resampling summaries retain their stated sampling unit and duplicate-handling rule. The CPU verifier checks the frozen interval files by hash and recomputes the listed point estimates; it does not regenerate every historical resampling draw.

`analyze_behavior.py` emits only the selected RV estimate and its corresponding uncertainty summary. Newer-model CW uses `--rv-weighting item` (the default), with an item-bootstrap CI. Original CW uses `--rv-weighting directed-pair`, with the historical label-pair resampling range. The latter uses 1,000 draws, pair seed 124, and merges duplicate sampled pairs before averaging; it is not a standard bootstrap CI. PFR and conditional RSR are separately labeled item-weighted rates. See IMPLEMENTATION_NOTES.md. All estimates are calculated from the supplied predictions.
