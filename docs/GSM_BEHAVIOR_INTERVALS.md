# GSM8K behavior intervals

The Qwen2.5-1.5B free-form behavior results reported in the paper are in `results/behavior/gsm_qwen15b/`:

- `scored_turns.csv`: per-item, per-turn scored outputs (500 items).
- `stage17_metrics_sorted_base_id_v1.json` and `stage17_bootstrap_ci_sorted_base_id_v1.csv`: the metrics and 95% intervals reported in the paper.

The evaluation contains 500 items, 325 initially correct items, 207 hard-flipped items, and 144 recovered items. The reported values are PFR 0.637 [0.585, 0.689], RSR 0.696 [0.633, 0.758], and RV 0.3125 [0.236, 0.389].

## Bootstrap procedure

Intervals are percentile bootstrap intervals over base items with 10,000 draws and seed 1701. Within each metric cohort, base IDs are sorted lexicographically before sampling, and a single random stream is shared in the order PFR → RSR → RV. `analyze_gsm8k_freeform_prr_stage17.py` implements this procedure and labels its output `computation_version: sorted_base_id_v1`. The paper-reported files were computed from `scored_turns.csv` with the same bootstrap routine; `scripts/verify_results.py` recomputes them on CPU.

## Earlier computation

`stage17_metrics.json` and `stage17_bootstrap_ci.csv` come from an earlier version of the analyzer that iterated over unordered sets before sampling, so the seed alone did not fix the sampling order. Counts and point estimates are identical; only the RV upper endpoint differs (0.396 vs. 0.389). These files are retained for provenance and are not used in the paper.

Fresh model runs additionally depend on the model, tokenizer, dataset revision, and numerical environment; record these when comparing new generations with the supplied scored turns.
