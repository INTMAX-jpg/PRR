#!/usr/bin/env python3
"""Experiment E_12: independent validation of E_10/E_11 probability results.

This validator intentionally does not import either analysis script.  It
recomputes normalization, clean held-out metrics, natural-T4 selection and the
primary causal statistics from saved row-level outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    converted = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if converted.isna().any():
        raise ValueError("Unparseable boolean values")
    return converted.astype(bool)


def correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if rank:
        first = pd.Series(first).rank(method="average").to_numpy(dtype=float)
        second = pd.Series(second).rank(method="average").to_numpy(dtype=float)
    if len(first) < 3 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def pair_gap(
    probability: np.ndarray, true_index: np.ndarray, wrong_index: np.ndarray
) -> np.ndarray:
    rows = np.arange(len(probability))
    return probability[rows, true_index] - probability[rows, wrong_index]


def compare(
    checks: list[Dict[str, Any]],
    *,
    name: str,
    recomputed: float,
    saved: float,
    tolerance: float,
) -> None:
    if math.isnan(recomputed) and math.isnan(saved):
        error = 0.0
        passed = True
    else:
        error = abs(float(recomputed) - float(saved))
        passed = bool(error <= tolerance)
    checks.append(
        {
            "name": name,
            "recomputed": float(recomputed),
            "saved": float(saved),
            "absolute_error": float(error),
            "tolerance": float(tolerance),
            "pass": passed,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e10_metrics", required=True)
    parser.add_argument("--e10_all_scores", required=True)
    parser.add_argument("--e10_natural_summary", required=True)
    parser.add_argument("--e11_panel", required=True)
    parser.add_argument("--e11_decision", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--labels", default="A,B,C,D")
    parser.add_argument("--pressure_trajectory", default="wrong_pressure")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    label_to_index = {label: index for index, label in enumerate(labels)}
    output_columns = [f"output_option_probability_{label}" for label in labels]
    readout_columns = [f"readout_option_probability_{label}" for label in labels]
    scores = pd.read_csv(args.e10_all_scores)
    metrics = pd.read_csv(args.e10_metrics)
    natural_saved = pd.read_csv(args.e10_natural_summary).iloc[0]
    panel = pd.read_csv(args.e11_panel)
    decision = json.loads(Path(args.e11_decision).read_text(encoding="utf-8"))
    tolerance = float(args.tolerance)
    checks: list[Dict[str, Any]] = []

    output_probability = scores[output_columns].to_numpy(dtype=np.float64)
    readout_probability = scores[readout_columns].to_numpy(dtype=np.float64)
    normalization = {
        "output_probability_sum_max_abs_error": float(
            np.max(np.abs(output_probability.sum(axis=1) - 1.0))
        ),
        "readout_probability_sum_max_abs_error": float(
            np.max(np.abs(readout_probability.sum(axis=1) - 1.0))
        ),
        "output_probability_min": float(output_probability.min()),
        "output_probability_max": float(output_probability.max()),
        "readout_probability_min": float(readout_probability.min()),
        "readout_probability_max": float(readout_probability.max()),
    }
    normalization["pass"] = bool(
        normalization["output_probability_sum_max_abs_error"] <= tolerance
        and normalization["readout_probability_sum_max_abs_error"] <= tolerance
        and normalization["output_probability_min"] >= -tolerance
        and normalization["output_probability_max"] <= 1.0 + tolerance
        and normalization["readout_probability_min"] >= -tolerance
        and normalization["readout_probability_max"] <= 1.0 + tolerance
    )

    clean = scores[
        scores["split"].astype(str).eq("test")
        & scores["trajectory_type"].astype(str).eq(args.pressure_trajectory)
        & scores["turn_index"].astype(int).eq(0)
    ].copy()
    q = clean[output_columns].to_numpy(dtype=np.float64)
    p = clean[readout_columns].to_numpy(dtype=np.float64)
    actual_preference = q.argmax(axis=1)
    predicted_preference = p.argmax(axis=1)
    gold = clean["true_label"].astype(str).map(label_to_index).to_numpy(dtype=int)
    wrong = clean["wrong_label"].astype(str).map(label_to_index).to_numpy(dtype=int)
    actual_gap = pair_gap(q, gold, wrong)
    predicted_gap = pair_gap(p, gold, wrong)
    clean_recomputed = {
        "preference_agreement": float(
            (actual_preference == predicted_preference).mean()
        ),
        "preference_balanced_agreement": float(
            balanced_accuracy_score(actual_preference, predicted_preference)
        ),
        "option_probability_kl": float(
            np.mean(
                np.sum(
                    np.clip(q, 1e-12, 1.0)
                    * (
                        np.log(np.clip(q, 1e-12, 1.0))
                        - np.log(np.clip(p, 1e-12, 1.0))
                    ),
                    axis=1,
                )
            )
        ),
        "option_probability_r2": float(
            r2_score(q.reshape(-1), p.reshape(-1))
        ),
        "probability_gap_r2": float(r2_score(actual_gap, predicted_gap)),
    }
    clean_saved = metrics[metrics["split"].astype(str).eq("test")].iloc[0]
    for name, value in clean_recomputed.items():
        compare(
            checks,
            name=f"clean_test__{name}",
            recomputed=value,
            saved=float(clean_saved[name]),
            tolerance=tolerance,
        )

    natural_frame = scores[
        scores["split"].astype(str).eq("test")
        & scores["trajectory_type"].astype(str).eq(args.pressure_trajectory)
        & scores["turn_index"].astype(int).isin([0, 3, 4])
    ].copy()
    records = []
    for base_id, group in natural_frame.groupby("base_id", sort=True):
        by_turn = {int(row["turn_index"]): row for _, row in group.iterrows()}
        if not all(turn in by_turn for turn in [0, 3, 4]):
            continue
        t0, t3, t4 = by_turn[0], by_turn[3], by_turn[4]
        if (
            str(t0["model_pred_label"]) == str(t0["true_label"])
            and str(t3["model_pred_label"]) == str(t3["wrong_label"])
            and str(t4["model_pred_label"]) == str(t4["true_label"])
        ):
            latent_gold_decline = float(
                t0["readout_gold_probability"] - t4["readout_gold_probability"]
            )
            output_gold_decline = float(
                t0["output_gold_probability"] - t4["output_gold_probability"]
            )
            latent_gap_decline = float(
                t0["readout_true_vs_wrong_probability_gap"]
                - t4["readout_true_vs_wrong_probability_gap"]
            )
            output_gap_decline = float(
                t0["output_true_vs_wrong_probability_gap"]
                - t4["output_true_vs_wrong_probability_gap"]
            )
            records.append(
                {
                    "latent_gold": latent_gold_decline,
                    "output_gold": output_gold_decline,
                    "gold_gap": latent_gold_decline - output_gold_decline,
                    "latent_pair": latent_gap_decline,
                    "output_pair": output_gap_decline,
                    "pair_gap": latent_gap_decline - output_gap_decline,
                    "pair_wrong": float(
                        t4["readout_true_vs_wrong_probability_gap"] < 0
                    ),
                    "exact_wrong": float(
                        str(t4["readout_pred_label"]) == str(t4["wrong_label"])
                    ),
                }
            )
    natural = pd.DataFrame(records)
    natural_recomputed = {
        "n_surface_recovered": float(len(natural)),
        "latent_gold_probability_decline_T0_minus_T4": float(
            natural["latent_gold"].mean()
        ),
        "output_gold_probability_decline_T0_minus_T4": float(
            natural["output_gold"].mean()
        ),
        "gold_probability_dissociation_gap": float(natural["gold_gap"].mean()),
        "latent_probability_gap_decline_T0_minus_T4": float(
            natural["latent_pair"].mean()
        ),
        "output_probability_gap_decline_T0_minus_T4": float(
            natural["output_pair"].mean()
        ),
        "probability_gap_dissociation_gap": float(natural["pair_gap"].mean()),
        "T4_readout_pairwise_wrong_alignment_rate": float(
            natural["pair_wrong"].mean()
        ),
        "T4_readout_exact_designated_wrong_rate": float(
            natural["exact_wrong"].mean()
        ),
    }
    for name, value in natural_recomputed.items():
        compare(
            checks,
            name=f"natural_T4__{name}",
            recomputed=value,
            saved=float(natural_saved[name]),
            tolerance=tolerance,
        )

    for column in ["output_correct", "output_designated_wrong"]:
        panel[column] = parse_bool(panel[column])
    for column in [
        "readout_pairwise_wrong_aligned",
        "readout_designated_wrong",
    ]:
        panel[column] = parse_bool(panel[column])
    pieces = []
    value_columns = [
        "output_correct",
        "output_designated_wrong",
        "output_true_vs_wrong_probability_gap",
        "readout_designated_wrong",
        "readout_pairwise_wrong_aligned",
        "readout_true_vs_wrong_probability_gap",
        *readout_columns,
    ]
    for mode in [
        "target_baseline",
        "output_logit_bias",
        "clean_state_restoration",
    ]:
        piece = panel[panel["mode"].astype(str).eq(mode)][
            ["base_id", *value_columns]
        ].copy()
        pieces.append(piece.set_index("base_id").add_prefix(f"{mode}__"))
    paired = pd.concat(pieces, axis=1, join="inner").reset_index()
    baseline = "target_baseline"
    bias = "output_logit_bias"
    restoration = "clean_state_restoration"
    repaired = paired[
        paired[f"{baseline}__output_designated_wrong"].astype(bool)
        & paired[f"{bias}__output_correct"].astype(bool)
    ]
    restoration_readout_delta = (
        paired[
            f"{restoration}__readout_true_vs_wrong_probability_gap"
        ].astype(float)
        - paired[f"{baseline}__readout_true_vs_wrong_probability_gap"].astype(float)
    )
    restoration_output_delta = (
        paired[
            f"{restoration}__output_true_vs_wrong_probability_gap"
        ].astype(float)
        - paired[f"{baseline}__output_true_vs_wrong_probability_gap"].astype(float)
    )
    max_bias_probability_change = 0.0
    for column in readout_columns:
        max_bias_probability_change = max(
            max_bias_probability_change,
            float(
                np.max(
                    np.abs(
                        paired[f"{bias}__{column}"].astype(float)
                        - paired[f"{baseline}__{column}"].astype(float)
                    )
                )
            ),
        )
    causal_recomputed = {
        "n_output_bias_surface_repairs": float(len(repaired)),
        "output_bias_readout_pairwise_wrong_rate": float(
            repaired[f"{bias}__readout_pairwise_wrong_aligned"].astype(float).mean()
        ),
        "output_bias_readout_exact_wrong_rate": float(
            repaired[f"{bias}__readout_designated_wrong"].astype(float).mean()
        ),
        "output_bias_readout_probability_delta_max_abs": (
            max_bias_probability_change
        ),
        "restoration_delta_spearman": correlation(
            restoration_readout_delta.to_numpy(),
            restoration_output_delta.to_numpy(),
            rank=True,
        ),
        "restoration_delta_pearson": correlation(
            restoration_readout_delta.to_numpy(),
            restoration_output_delta.to_numpy(),
        ),
        "restoration_delta_sign_agreement": float(
            (
                np.signbit(restoration_readout_delta)
                == np.signbit(restoration_output_delta)
            ).mean()
        ),
    }
    for name, value in causal_recomputed.items():
        compare(
            checks,
            name=f"causal__{name}",
            recomputed=value,
            saved=float(decision[name]),
            tolerance=tolerance,
        )

    max_saved_error = max(item["absolute_error"] for item in checks)
    result = {
        "experiment": "E_12_probability_readout_independent_validation",
        "probability_normalization": normalization,
        "checks": checks,
        "n_checks": int(len(checks)),
        "max_saved_stat_absolute_error": float(max_saved_error),
        "tolerance": tolerance,
        "all_checks_pass": bool(
            normalization["pass"] and all(item["pass"] for item in checks)
        ),
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "实验E_12_independent_validation.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
