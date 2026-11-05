# Implementation notes

The probability-readout and causal scripts preserve the final-rebuttal implementation and their local helper dependencies. Portable path substitutions adapt the original execution environment to this repository.

The behavior-only pressure runner retains the original prompt rendering, option-logit scoring, deterministic argmax, T0 reuse, sequential history updates, and matched T5 forks. It emits surface outcomes and logits. The CW runner uses the same scoring operations and records the actual number of supplied base items and trajectories. Experiment sizes are set by the input construction commands.

CW 2,000-item configurations are prepared from the paper's reported settings. Results are calculated from actual model outputs; no published filtering counts are embedded into the runner. Fitted probability readouts and the numerical artifacts in MATERIALS.csv retain their own experimental cohorts.

The synthetic CW dataset preserves the original association between queried field and correct answer label. The companion dataset card describes this potential shortcut. Reconstructing this paper version does not rebalance or silently alter it.

The Qwen2.5-14B behavior configuration uses `device_map: auto`, matching its generated runtime configuration. `cuda:0` remains the input/default device; automatic placement can distribute model layers.

The weak-only implementation copies the original preparation logic and retains its paired surface-analysis formulas. It uses neutral fillers at T1–T4 and the exact PRR renewed weak prompt at T5. The comparison changes the preceding history and does not isolate T4.

MMLU and ARC causal samples are 200-item subsets of each held-out test split, disjoint from training and development. In these tasks L24 was selected by the largest development T0–T3 true-minus-wrong margin drop in the preceding corrected-lens analysis. Clean development data selects regularization; the probability translator is frozen before test evaluation. The output-correction denominators use baseline designated-wrong outputs that become correct (156 and 169).

`cw_pair_resampling.py` preserves the original CW interval calculation: draw directed label pairs with replacement, merge duplicate selected pairs, and average the remaining distinct pair estimates equally. Its 2.5th and 97.5th percentiles are descriptive resampling ranges, not standard bootstrap confidence intervals. Weak-only contrasts instead retain duplicate pair draws. Pressure-fork summaries share the same item draws across template variants and neutral control.

The behavior analyzer emits one RV weighting per invocation: item weighting for newer-model CW and directed-label-pair weighting for original CW. The pressure analysis requires an explicit declaration of preselected recovered input or a population file for filtering. Pressure model configurations describe the 500-item source study separately from the original CW 2,000-item configurations; both runners retain their model and prompt settings.

Numeric JSON device keys are normalized to integers before passing `max_memory` to the model loader. The GSM causal writer records its actual repair/readout layer arguments, with the paper defaults retained at L22/L24. Behavior runs optionally save complete rendered prompts when configured.

New GSM behavior bootstrap computations use sorted base IDs and versioned output files; the reported intervals and numerical artifacts remain unchanged. See GSM_BEHAVIOR_INTERVALS.md. Historical newer-model screening rules are provenance metadata; the published runner directly evaluates the supplied final 1,000-item input.
