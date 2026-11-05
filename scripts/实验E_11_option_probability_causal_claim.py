#!/usr/bin/env python3
"""Experiment E_11: probability-scale causal test of the dissociation claim.

This script combines the frozen E_10 option-probability readout with E_03
interventions.  Every output probability is recomputed by applying softmax to
exactly output_logit_A/B/C/D.  Every readout probability comes from E_10's
four-way affine-softmax probe.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


METHOD = "option_probability_affine_softmax"
MODES = [
    "target_baseline",
    "output_logit_bias",
    "clean_state_restoration",
]


def option_softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - array.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.lower().map(mapping)
    if converted.isna().any():
        raise ValueError(
            f"Cannot parse boolean values: {series[converted.isna()].unique()}"
        )
    return converted.astype(bool)


def safe_correlation(
    first: Iterable[float], second: Iterable[float], *, rank: bool = False
) -> float:
    x = np.asarray(list(first), dtype=np.float64)
    y = np.asarray(list(second), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return math.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan
    if rank:
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean(
    values: Iterable[float], n_bootstrap: int, seed: int
) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        draws[index] = float(rng.choice(array, size=len(array), replace=True).mean())
    return {
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def readout_lookup(
    all_scores: pd.DataFrame,
    pressure_trajectory: str,
    clean_turn: int,
    target_turn: int,
) -> Dict[tuple[str, int], Dict[str, Any]]:
    frame = all_scores[
        all_scores["split"].astype(str).eq("test")
        & all_scores["trajectory_type"].astype(str).eq(pressure_trajectory)
        & all_scores["turn_index"].astype(int).isin([clean_turn, target_turn])
    ].copy()
    result: Dict[tuple[str, int], Dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row["base_id"]), int(row["turn_index"]))
        if key in result:
            raise ValueError(f"Duplicate E_10 score row for {key}")
        result[key] = row.to_dict()
    return result


def build_panel(
    *,
    all_scores: pd.DataFrame,
    e03_rows: pd.DataFrame,
    labels: Sequence[str],
    pressure_trajectory: str,
    clean_turn: int,
    target_turn: int,
) -> pd.DataFrame:
    lookup = readout_lookup(
        all_scores, pressure_trajectory, clean_turn, target_turn
    )
    e03 = e03_rows[e03_rows["mode"].astype(str).isin(MODES)].copy()
    e03["output_correct"] = parse_bool(e03["output_correct"])
    e03["output_designated_wrong"] = parse_bool(e03["output_designated_wrong"])
    raw_columns = [f"output_logit_{label}" for label in labels]
    missing = [column for column in raw_columns if column not in e03.columns]
    if missing:
        raise ValueError(f"E_03 is missing intervention output logits: {missing}")
    output_probability = option_softmax(
        e03[raw_columns].to_numpy(dtype=np.float64)
    )
    for index, label in enumerate(labels):
        e03[f"recomputed_output_probability_{label}"] = output_probability[:, index]
    recomputed_label = np.asarray(labels)[output_probability.argmax(axis=1)]
    if not np.array_equal(recomputed_label, e03["output_pred_label"].astype(str)):
        mismatch = int(
            (recomputed_label != e03["output_pred_label"].astype(str).to_numpy()).sum()
        )
        raise ValueError(
            f"E_03 output label disagrees with four-option argmax on {mismatch} rows"
        )
    output_rows = {
        (str(row["base_id"]), str(row["mode"])): row.to_dict()
        for _, row in e03.iterrows()
    }
    label_index = {str(label): index for index, label in enumerate(labels)}
    panel_rows: List[Dict[str, Any]] = []
    for base_id in sorted(e03["base_id"].astype(str).unique()):
        clean = lookup.get((base_id, clean_turn))
        target = lookup.get((base_id, target_turn))
        if clean is None or target is None:
            continue
        for mode in MODES:
            output = output_rows.get((base_id, mode))
            if output is None:
                continue
            readout_source = clean if mode == "clean_state_restoration" else target
            true_label = str(output["true_label"])
            wrong_label = str(output["wrong_label"])
            true_index = label_index[true_label]
            wrong_index = label_index[wrong_label]
            output_probs = np.asarray(
                [
                    output[f"recomputed_output_probability_{label}"]
                    for label in labels
                ],
                dtype=np.float64,
            )
            readout_probs = np.asarray(
                [
                    readout_source[f"readout_option_probability_{label}"]
                    for label in labels
                ],
                dtype=np.float64,
            )
            if abs(output_probs.sum() - 1.0) > 1e-10:
                raise AssertionError("E_03 option probabilities do not sum to one")
            if abs(readout_probs.sum() - 1.0) > 1e-10:
                raise AssertionError("E_10 readout probabilities do not sum to one")
            readout_prediction = str(labels[int(readout_probs.argmax())])
            row = {
                "base_id": base_id,
                "method": METHOD,
                "mode": mode,
                "true_label": true_label,
                "wrong_label": wrong_label,
                "output_pred_label": str(output["output_pred_label"]),
                "output_correct": bool(output["output_correct"]),
                "output_designated_wrong": bool(
                    output["output_designated_wrong"]
                ),
                "output_gold_probability": float(output_probs[true_index]),
                "output_wrong_probability": float(output_probs[wrong_index]),
                "output_true_vs_wrong_probability_gap": float(
                    output_probs[true_index] - output_probs[wrong_index]
                ),
                "readout_activation_source": (
                    f"T{clean_turn}_clean"
                    if mode == "clean_state_restoration"
                    else f"T{target_turn}_pressure_target"
                ),
                "readout_pred_label": readout_prediction,
                "readout_correct": readout_prediction == true_label,
                "readout_designated_wrong": readout_prediction == wrong_label,
                "readout_gold_probability": float(readout_probs[true_index]),
                "readout_wrong_probability": float(readout_probs[wrong_index]),
                "readout_true_vs_wrong_probability_gap": float(
                    readout_probs[true_index] - readout_probs[wrong_index]
                ),
                "readout_pairwise_wrong_aligned": bool(
                    readout_probs[true_index] < readout_probs[wrong_index]
                ),
            }
            for index, label in enumerate(labels):
                row[f"output_option_probability_{label}"] = float(
                    output_probs[index]
                )
                row[f"readout_option_probability_{label}"] = float(
                    readout_probs[index]
                )
            panel_rows.append(row)
    return pd.DataFrame(panel_rows)


def mode_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for mode, group in panel.groupby("mode", sort=True):
        rows.append(
            {
                "method": METHOD,
                "mode": mode,
                "n": int(group["base_id"].nunique()),
                "output_correct_rate": float(group["output_correct"].mean()),
                "output_designated_wrong_rate": float(
                    group["output_designated_wrong"].mean()
                ),
                "output_gold_probability_mean": float(
                    group["output_gold_probability"].mean()
                ),
                "output_probability_gap_mean": float(
                    group["output_true_vs_wrong_probability_gap"].mean()
                ),
                "readout_correct_rate": float(group["readout_correct"].mean()),
                "readout_designated_wrong_rate": float(
                    group["readout_designated_wrong"].mean()
                ),
                "readout_pairwise_wrong_alignment_rate": float(
                    group["readout_pairwise_wrong_aligned"].mean()
                ),
                "readout_gold_probability_mean": float(
                    group["readout_gold_probability"].mean()
                ),
                "readout_probability_gap_mean": float(
                    group["readout_true_vs_wrong_probability_gap"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_frame(panel: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "output_correct",
        "output_designated_wrong",
        "output_gold_probability",
        "output_true_vs_wrong_probability_gap",
        "readout_correct",
        "readout_designated_wrong",
        "readout_pairwise_wrong_aligned",
        "readout_gold_probability",
        "readout_true_vs_wrong_probability_gap",
        "readout_option_probability_A",
        "readout_option_probability_B",
        "readout_option_probability_C",
        "readout_option_probability_D",
    ]
    pieces = []
    for mode in MODES:
        piece = panel[panel["mode"].eq(mode)][["base_id", *value_columns]].copy()
        pieces.append(piece.set_index("base_id").add_prefix(f"{mode}__"))
    return pd.concat(pieces, axis=1, join="inner").reset_index()


def causal_analysis(
    *,
    panel: pd.DataFrame,
    clean_metrics: pd.DataFrame,
    labels: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    paired = paired_frame(panel)
    baseline = "target_baseline"
    bias = "output_logit_bias"
    restoration = "clean_state_restoration"
    effect_rows: List[Dict[str, Any]] = []
    delta_cache: Dict[str, tuple[pd.Series, pd.Series]] = {}
    for comparison_index, (name, mode) in enumerate(
        [
            ("output_logit_bias_minus_baseline", bias),
            ("clean_state_restoration_minus_baseline", restoration),
        ]
    ):
        output_gold_delta = (
            paired[f"{mode}__output_gold_probability"].astype(float)
            - paired[f"{baseline}__output_gold_probability"].astype(float)
        )
        readout_gold_delta = (
            paired[f"{mode}__readout_gold_probability"].astype(float)
            - paired[f"{baseline}__readout_gold_probability"].astype(float)
        )
        output_gap_delta = (
            paired[f"{mode}__output_true_vs_wrong_probability_gap"].astype(float)
            - paired[
                f"{baseline}__output_true_vs_wrong_probability_gap"
            ].astype(float)
        )
        readout_gap_delta = (
            paired[f"{mode}__readout_true_vs_wrong_probability_gap"].astype(float)
            - paired[
                f"{baseline}__readout_true_vs_wrong_probability_gap"
            ].astype(float)
        )
        delta_cache[name] = (output_gap_delta, readout_gap_delta)
        row: Dict[str, Any] = {
            "method": METHOD,
            "comparison": name,
            "n": int(len(paired)),
        }
        for offset, (prefix, values) in enumerate(
            [
                ("output_gold_probability_delta", output_gold_delta),
                ("readout_gold_probability_delta", readout_gold_delta),
                ("output_probability_gap_delta", output_gap_delta),
                ("readout_probability_gap_delta", readout_gap_delta),
            ]
        ):
            result = bootstrap_mean(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed + comparison_index * 20 + offset,
            )
            row[prefix] = result["estimate"]
            row[f"{prefix}_ci_low"] = result["ci_low"]
            row[f"{prefix}_ci_high"] = result["ci_high"]
        effect_rows.append(row)

    bias_output_delta, bias_readout_delta = delta_cache[
        "output_logit_bias_minus_baseline"
    ]
    restoration_output_delta, restoration_readout_delta = delta_cache[
        "clean_state_restoration_minus_baseline"
    ]
    surface_repaired = paired[
        paired[f"{baseline}__output_designated_wrong"].astype(bool)
        & paired[f"{bias}__output_correct"].astype(bool)
    ].copy()
    pair_wrong_ci = bootstrap_mean(
        surface_repaired[f"{bias}__readout_pairwise_wrong_aligned"].astype(float),
        n_bootstrap=n_bootstrap,
        seed=seed + 500,
    )
    exact_wrong_ci = bootstrap_mean(
        surface_repaired[f"{bias}__readout_designated_wrong"].astype(float),
        n_bootstrap=n_bootstrap,
        seed=seed + 501,
    )
    readout_probability_delta_max_abs = 0.0
    for label in labels:
        difference = (
            paired[f"{bias}__readout_option_probability_{label}"].astype(float)
            - paired[f"{baseline}__readout_option_probability_{label}"].astype(float)
        )
        readout_probability_delta_max_abs = max(
            readout_probability_delta_max_abs,
            float(np.max(np.abs(difference))),
        )

    test_metric = clean_metrics[
        clean_metrics["method"].astype(str).eq(METHOD)
        & clean_metrics["split"].astype(str).eq("test")
    ]
    if len(test_metric) != 1:
        raise ValueError("Expected exactly one E_10 clean-test metric row")
    test_metric = test_metric.iloc[0]
    reliability = {
        "clean_test_preference_agreement": float(
            test_metric["preference_agreement"]
        ),
        "clean_test_balanced_agreement": float(
            test_metric["preference_balanced_agreement"]
        ),
        "clean_test_option_probability_kl": float(
            test_metric["option_probability_kl"]
        ),
        "clean_test_probability_gap_r2": float(
            test_metric["probability_gap_r2"]
        ),
        "restoration_delta_spearman": safe_correlation(
            restoration_readout_delta, restoration_output_delta, rank=True
        ),
        "restoration_delta_pearson": safe_correlation(
            restoration_readout_delta, restoration_output_delta
        ),
        "restoration_delta_sign_agreement": float(
            (
                np.signbit(restoration_readout_delta)
                == np.signbit(restoration_output_delta)
            ).mean()
        ),
        "n_output_bias_surface_repairs": int(len(surface_repaired)),
        "output_bias_readout_pairwise_wrong_rate": pair_wrong_ci["estimate"],
        "output_bias_readout_pairwise_wrong_ci_low": pair_wrong_ci["ci_low"],
        "output_bias_readout_pairwise_wrong_ci_high": pair_wrong_ci["ci_high"],
        "output_bias_readout_exact_wrong_rate": exact_wrong_ci["estimate"],
        "output_bias_readout_exact_wrong_ci_low": exact_wrong_ci["ci_low"],
        "output_bias_readout_exact_wrong_ci_high": exact_wrong_ci["ci_high"],
        "output_bias_readout_probability_delta_max_abs": (
            readout_probability_delta_max_abs
        ),
        "output_bias_output_probability_gap_delta_mean": float(
            bias_output_delta.mean()
        ),
        "output_bias_readout_probability_gap_delta_mean": float(
            bias_readout_delta.mean()
        ),
    }
    gates = {
        "clean_test_preference_agreement_gte_0p80": (
            reliability["clean_test_preference_agreement"] >= 0.80
        ),
        "clean_test_balanced_agreement_gte_0p75": (
            reliability["clean_test_balanced_agreement"] >= 0.75
        ),
        "clean_test_probability_gap_r2_gte_0p50": (
            reliability["clean_test_probability_gap_r2"] >= 0.50
        ),
        "restoration_spearman_gte_0p50": (
            reliability["restoration_delta_spearman"] >= 0.50
        ),
        "restoration_sign_agreement_gte_0p75": (
            reliability["restoration_delta_sign_agreement"] >= 0.75
        ),
        "at_least_20_output_bias_surface_repairs": (
            reliability["n_output_bias_surface_repairs"] >= 20
        ),
        "output_bias_readout_distribution_is_unchanged": (
            reliability["output_bias_readout_probability_delta_max_abs"] <= 1e-12
        ),
        "majority_of_surface_repairs_remain_readout_wrong_aligned": (
            reliability["output_bias_readout_pairwise_wrong_rate"] > 0.50
            and reliability["output_bias_readout_pairwise_wrong_ci_low"] > 0.50
        ),
    }
    decision = {
        "method": METHOD,
        **reliability,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }
    return pd.DataFrame(effect_rows), pd.DataFrame([reliability]), decision


def write_report(
    *,
    path: Path,
    dataset_name: str,
    clean_metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    effects: pd.DataFrame,
    decision: Dict[str, Any],
) -> None:
    primary = clean_metrics[
        clean_metrics["method"].eq(METHOD)
        & clean_metrics["split"].eq("test")
    ].iloc[0]
    wording = (
        "SUPPORTED with the preregistered operational limitation"
        if decision["all_gates_pass"]
        else "NOT SUPPORTED by all preregistered gates"
    )
    report = f"""# 实验 E_11：{dataset_name} 四选项概率 readout 因果检验

## 结论

**{wording}**

所有目标和预测均为仅在 A/B/C/D 内归一化的概率分布，每行严格加和为 1。

允许检验的窄 claim：

> Surface answer correction can be dissociated from repair of a
> behaviorally validated, readout-defined intermediate option-preference
> distribution.

## Clean held-out fidelity

- preference agreement: {float(primary['preference_agreement']):.4f}
- balanced agreement: {float(primary['preference_balanced_agreement']):.4f}
- option-probability KL: {float(primary['option_probability_kl']):.6f}
- option-probability R²: {float(primary['option_probability_r2']):.4f}
- true-vs-wrong probability-gap R²: {float(primary['probability_gap_r2']):.4f}

## Causal fidelity and separation

- clean-restoration predicted/actual probability-gap delta Spearman:
  {float(decision['restoration_delta_spearman']):.4f}
- delta sign agreement:
  {float(decision['restoration_delta_sign_agreement']):.4f}
- output-bias surface-repaired items:
  {int(decision['n_output_bias_surface_repairs'])}
- among those, unchanged readout pairwise-wrong rate:
  {float(decision['output_bias_readout_pairwise_wrong_rate']):.4f}
  [{float(decision['output_bias_readout_pairwise_wrong_ci_low']):.4f},
   {float(decision['output_bias_readout_pairwise_wrong_ci_high']):.4f}]
- maximum A/B/C/D readout-probability change under output-only bias:
  {float(decision['output_bias_readout_probability_delta_max_abs']):.3e}

## Gate decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## Mode summary

{summaries.to_markdown(index=False)}

## Paired effects

{effects.to_markdown(index=False)}

## Interpretation boundary

该结果证明的是干预层面的可分离性：后置 output-only intervention 可以修正
surface answer 而不改变较早的 L24 option-preference distribution。它不把该分布
等同于 universal belief state，也不单凭这一实验断言自然 T4 recovery 使用独立机制。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e10_metrics", required=True)
    parser.add_argument("--e10_all_scores", required=True)
    parser.add_argument("--e03_rows", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset_name", default="dataset")
    parser.add_argument("--labels", default="A,B,C,D")
    parser.add_argument("--pressure_trajectory", default="wrong_pressure")
    parser.add_argument("--clean_turn", type=int, default=0)
    parser.add_argument("--target_turn", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if len(labels) != 4:
        raise ValueError(f"Expected four labels, got {labels}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean_metrics = pd.read_csv(args.e10_metrics)
    all_scores = pd.read_csv(args.e10_all_scores)
    e03_rows = pd.read_csv(args.e03_rows)
    panel = build_panel(
        all_scores=all_scores,
        e03_rows=e03_rows,
        labels=labels,
        pressure_trajectory=args.pressure_trajectory,
        clean_turn=args.clean_turn,
        target_turn=args.target_turn,
    )
    if panel.empty:
        raise ValueError("No joined E_10/E_03 rows")
    summaries = mode_summary(panel)
    effects, gates, decision = causal_analysis(
        panel=panel,
        clean_metrics=clean_metrics,
        labels=labels,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    paths = {
        "panel": outdir / "实验E_11_probability_causal_panel.csv.gz",
        "mode_summary": outdir / "实验E_11_mode_summary.csv",
        "paired_effects": outdir / "实验E_11_paired_effects.csv",
        "claim_gates": outdir / "实验E_11_claim_gates.csv",
        "decision": outdir / "实验E_11_claim_decision.json",
        "report": outdir / "实验E_11_report.md",
        "manifest": outdir / "实验E_11_manifest.json",
    }
    panel.to_csv(paths["panel"], index=False, compression="gzip")
    summaries.to_csv(paths["mode_summary"], index=False)
    effects.to_csv(paths["paired_effects"], index=False)
    gates.to_csv(paths["claim_gates"], index=False)
    paths["decision"].write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        path=paths["report"],
        dataset_name=args.dataset_name,
        clean_metrics=clean_metrics,
        summaries=summaries,
        effects=effects,
        decision=decision,
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "experiment": "E_11_option_probability_causal_claim",
                "dataset_name": args.dataset_name,
                "labels": labels,
                "probability_definition": (
                    "softmax over exactly A/B/C/D for E_03 outputs and "
                    "E_10 affine-softmax probabilities for readout"
                ),
                "readout_training_data": (
                    "E_10 clean T0 train only; E_11 does not refit"
                ),
                "surface_repair_selection": (
                    "actual E_03 target-baseline designated-wrong and "
                    "output-logit-bias correct; readout not used for selection"
                ),
                "outputs": {
                    name: str(output.resolve()) for name, output in paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"[OK] {paths['report']}")


if __name__ == "__main__":
    main()
