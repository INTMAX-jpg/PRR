# GSM8K behavior: reported and newly computed intervals

The paper-reported Qwen2.5-1.5B behavior results are preserved in:

- `results/behavior/gsm_qwen15b/stage17_metrics.json`
- `results/behavior/gsm_qwen15b/stage17_bootstrap_ci.csv`
- `results/behavior/gsm_qwen15b/scored_turns.csv`

These are the frozen reported artifacts. They contain 500 evaluated items, 325 initially correct items, 207 hard-flipped items, and 144 recovered items. The reported RV is 0.3125 with interval [0.2361111111111111, 0.3958333333333333]; the reported RSR is 144/207 with interval [0.6328502415458938, 0.7584541062801933]. The paper continues to use these reported values.

## New computation version

`analyze_gsm8k_freeform_prr_stage17.py` now sorts the base IDs lexicographically within each metric cohort before sampling. It retains the item bootstrap, percentile calculation, default 10,000 draws, seed 1701, and the shared random-stream order PFR → RSR → RV. Cohort definitions, answer parsing, and point-estimate formulas are unchanged.

New output is identified by `result_origin: new_computation` and `computation_version: sorted_base_id_v1`. Metrics and interval files are named `stage17_metrics_sorted_base_id_v1.json` and `stage17_bootstrap_ci_sorted_base_id_v1.csv`; the generated report also bears the version suffix. The analyzer writes into a new output directory.

The historical analyzer iterated over Python sets before sampling. Its recorded random seed alone therefore did not specify the input order for finite bootstrap draws. A sorted recomputation can produce slightly different interval endpoints on identical scored rows. Such endpoints are newly computed intervals, rather than replacements for the frozen paper-reported intervals. A matching point estimate does not require identical finite-resampling endpoints.

The workflow and plotting paths retain the frozen reported artifacts as the reference for paper figures and tables. Fresh model runs additionally depend on their model, tokenizer, dataset revision, and numerical environment; record these separately when comparing fresh predictions.
