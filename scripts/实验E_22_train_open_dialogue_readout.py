#!/usr/bin/env python3
"""Train a clean-only L24 readout for the open-dialogue X/Y diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from 实验E_00_common import (  # noqa: E402
    command_record,
    file_fingerprint,
    json_dump,
    json_load,
)
from 实验E_10_option_probability_readout import (  # noqa: E402
    bootstrap_mean,
    fit_and_select,
    pair_gap,
    predict_probability,
    probability_metrics,
)


LABELS = ["X", "Y"]
MAPPING = {"X": 0, "Y": 1}


def parse_grid(value: str) -> List[float]:
    result = [float(item) for item in str(value).split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("Invalid L2 grid")
    return result


def add_bootstrap(
    destination: Dict[str, Any],
    name: str,
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> None:
    result = bootstrap_mean(values, n_bootstrap=samples, seed=seed)
    destination[name] = result["estimate"]
    destination[f"{name}_ci_low"] = result["ci_low"]
    destination[f"{name}_ci_high"] = result["ci_high"]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return centre - radius, centre + radius


def bootstrap_metric_interval(
    *,
    target: np.ndarray,
    predicted: np.ndarray,
    metric,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    if len(target) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(int(samples), dtype=np.float64)
    for bootstrap_index in range(int(samples)):
        indices = rng.integers(0, len(target), size=len(target))
        try:
            values[bootstrap_index] = float(
                metric(target[indices], predicted[indices])
            )
        except Exception:
            values[bootstrap_index] = math.nan
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.nan, math.nan
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def add_uncertainty(
    row: Dict[str, Any],
    *,
    predicted_probability: np.ndarray,
    target_probability: np.ndarray,
    target_preference: np.ndarray,
    gold: np.ndarray,
    samples: int,
    seed: int,
) -> None:
    predicted_probability = np.asarray(predicted_probability, dtype=np.float64)
    target_probability = np.asarray(target_probability, dtype=np.float64)
    prediction = predicted_probability.argmax(axis=1)
    preference_success = int((prediction == target_preference).sum())
    gold_success = int((prediction == gold).sum())
    (
        row["preference_agreement_ci_low"],
        row["preference_agreement_ci_high"],
    ) = wilson_interval(preference_success, len(prediction))
    row["gold_accuracy_ci_low"], row["gold_accuracy_ci_high"] = wilson_interval(
        gold_success, len(prediction)
    )
    (
        row["preference_balanced_agreement_ci_low"],
        row["preference_balanced_agreement_ci_high"],
    ) = bootstrap_metric_interval(
        target=np.asarray(target_preference, dtype=int),
        predicted=np.asarray(prediction, dtype=int),
        metric=balanced_accuracy_score,
        samples=samples,
        seed=seed,
    )
    (
        row["option_probability_r2_ci_low"],
        row["option_probability_r2_ci_high"],
    ) = bootstrap_metric_interval(
        target=target_probability,
        predicted=predicted_probability,
        metric=lambda actual, estimate: r2_score(
            actual.reshape(-1), estimate.reshape(-1)
        ),
        samples=samples,
        seed=seed + 1,
    )
    clipped_target = np.clip(target_probability, 1e-12, 1.0)
    clipped_predicted = np.clip(predicted_probability, 1e-12, 1.0)
    row_kl = np.sum(
        clipped_target
        * (np.log(clipped_target) - np.log(clipped_predicted)),
        axis=1,
    )
    kl_result = bootstrap_mean(
        row_kl, n_bootstrap=samples, seed=seed + 2
    )
    row["option_probability_kl_ci_low"] = kl_result["ci_low"]
    row["option_probability_kl_ci_high"] = kl_result["ci_high"]


def natural_recovery_analysis(
    predictions: pd.DataFrame, *, samples: int, seed: int
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    causal = predictions[
        predictions["experiment_split"].astype(str).eq("causal_eval")
    ].copy()
    rows: List[Dict[str, Any]] = []
    for base_id, group in causal.groupby("base_id", sort=True):
        by_turn = {int(row["turn_index"]): row for _, row in group.iterrows()}
        if not all(turn in by_turn for turn in range(6)):
            continue
        t0, t1, t3, t4, t5 = (
            by_turn[0],
            by_turn[1],
            by_turn[3],
            by_turn[4],
            by_turn[5],
        )
        surface_recovered = (
            str(t0["freeform_outcome"]) == "gold"
            and str(t3["freeform_outcome"]) == "designated_wrong"
            and str(t4["freeform_outcome"]) == "gold"
        )
        if not surface_recovered:
            continue
        rows.append(
            {
                "base_id": str(base_id),
                "T0_target_gold_probability": float(
                    t0["target_gold_probability"]
                ),
                "T4_target_gold_probability": float(
                    t4["target_gold_probability"]
                ),
                "T0_readout_gold_probability": float(
                    t0["readout_gold_probability"]
                ),
                "T4_readout_gold_probability": float(
                    t4["readout_gold_probability"]
                ),
                "T0_target_gap": float(t0["target_gold_vs_wrong_gap"]),
                "T4_target_gap": float(t4["target_gold_vs_wrong_gap"]),
                "T0_readout_gap": float(t0["readout_gold_vs_wrong_gap"]),
                "T4_readout_gap": float(t4["readout_gold_vs_wrong_gap"]),
                "T1_target_wrong_probability": float(
                    t1["target_wrong_probability"]
                ),
                "T5_target_wrong_probability": float(
                    t5["target_wrong_probability"]
                ),
                "T1_readout_wrong_probability": float(
                    t1["readout_wrong_probability"]
                ),
                "T5_readout_wrong_probability": float(
                    t5["readout_wrong_probability"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary: Dict[str, Any] = {
        "n_surface_recovered_freeform": int(len(frame))
    }
    if len(frame):
        add_bootstrap(
            summary,
            "target_T0_minus_T4_gold_probability",
            (
                frame["T0_target_gold_probability"]
                - frame["T4_target_gold_probability"]
            ).to_numpy(),
            samples=samples,
            seed=seed + 1,
        )
        add_bootstrap(
            summary,
            "readout_T0_minus_T4_gold_probability",
            (
                frame["T0_readout_gold_probability"]
                - frame["T4_readout_gold_probability"]
            ).to_numpy(),
            samples=samples,
            seed=seed + 2,
        )
        add_bootstrap(
            summary,
            "target_T5_minus_T1_wrong_probability",
            (
                frame["T5_target_wrong_probability"]
                - frame["T1_target_wrong_probability"]
            ).to_numpy(),
            samples=samples,
            seed=seed + 3,
        )
        add_bootstrap(
            summary,
            "readout_T5_minus_T1_wrong_probability",
            (
                frame["T5_readout_wrong_probability"]
                - frame["T1_readout_wrong_probability"]
            ).to_numpy(),
            samples=samples,
            seed=seed + 4,
        )
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--readout_layer", type=int, default=24)
    parser.add_argument(
        "--l2_grid",
        default=(
            "0,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,"
            "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1"
        ),
    )
    parser.add_argument("--max_iter", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split_document = json_load(Path(args.split_manifest))
    split = split_document["split"]
    split_sets = {name: set(map(str, values)) for name, values in split.items()}
    run_dir = Path(args.run_dir)
    manifest = json_load(run_dir / "manifest.json")
    metadata = pd.read_csv(run_dir / "metadata.csv")
    activation_path = (
        run_dir / "activations" / f"resid_pre_L{args.readout_layer}.npy"
    )
    activation = np.load(activation_path, mmap_mode="r")
    if len(metadata) != len(activation):
        raise ValueError("Metadata/activation length mismatch")

    clean = metadata[
        ~metadata["experiment_split"].astype(str).eq("causal_eval")
    ].copy()
    if len(clean) != 819 or clean["base_id"].astype(str).duplicated().any():
        raise ValueError("Expected one clean row for each of 819 readout items")
    expected_clean = set().union(
        split_sets["readout_train"],
        split_sets["readout_dev"],
        split_sets["readout_test"],
    )
    if set(clean["base_id"].astype(str)) != expected_clean:
        raise ValueError("Clean readout pool does not match E19 split")
    clean["probe_split"] = clean["experiment_split"].astype(str).str.replace(
        "readout_", "", regex=False
    )
    clean = clean.sort_values(["probe_split", "base_id"]).reset_index(drop=True)
    clean_x = np.asarray(
        activation[clean["row_idx"].astype(int).to_numpy()], dtype=np.float32
    )
    target_probability = clean[["probability_X", "probability_Y"]].to_numpy(
        dtype=np.float64
    )
    target_preference = target_probability.argmax(axis=1)
    stored_preference = clean["diagnostic_pred_label"].map(MAPPING).to_numpy()
    if not np.array_equal(target_preference, stored_preference):
        raise ValueError("Stored diagnostic preference mismatch")
    gold = clean["gold_option_label"].map(MAPPING).to_numpy(dtype=int)
    wrong = clean["wrong_option_label"].map(MAPPING).to_numpy(dtype=int)
    indices = {
        name: clean.index[clean["probe_split"].eq(name)].to_numpy()
        for name in ["train", "dev", "test"]
    }
    counts = {name: int(len(value)) for name, value in indices.items()}
    if counts != {"train": 500, "dev": 159, "test": 160}:
        raise ValueError(f"Unexpected readout counts: {counts}")

    bundle, tuning = fit_and_select(
        x_train=clean_x[indices["train"]],
        train_probability=target_probability[indices["train"]],
        x_dev=clean_x[indices["dev"]],
        dev_probability=target_probability[indices["dev"]],
        dev_preference=target_preference[indices["dev"]],
        dev_gold=gold[indices["dev"]],
        dev_wrong=wrong[indices["dev"]],
        l2_grid=parse_grid(args.l2_grid),
        max_iter=int(args.max_iter),
        threads=int(args.threads),
    )
    bundle.update(
        {
            "experiment": "E_22_open_dialogue_probability_readout",
            "model_name": args.model_name,
            "labels": LABELS,
            "readout_layer": int(args.readout_layer),
            "feature_key": f"resid_pre_L{args.readout_layer}",
            "training_base_ids": sorted(split_sets["readout_train"]),
            "dev_base_ids": sorted(split_sets["readout_dev"]),
            "test_base_ids": sorted(split_sets["readout_test"]),
            "causal_eval_base_ids_excluded": sorted(split_sets["causal_eval"]),
        }
    )

    metrics = []
    clean_predictions = clean.copy()
    for split_name, split_index in indices.items():
        predicted = predict_probability(
            bundle["model"],
            bundle["scaler"].transform(clean_x[split_index]),
        )
        metric_row = {
            "model_name": args.model_name,
            "selected_l2": bundle["selected_l2"],
            **probability_metrics(
                split=split_name,
                predicted_probability=predicted,
                target_probability=target_probability[split_index],
                target_preference=target_preference[split_index],
                gold=gold[split_index],
                wrong=wrong[split_index],
            ),
        }
        add_uncertainty(
            metric_row,
            predicted_probability=predicted,
            target_probability=target_probability[split_index],
            target_preference=target_preference[split_index],
            gold=gold[split_index],
            samples=int(args.bootstrap_samples),
            seed=int(args.seed) + ["train", "dev", "test"].index(split_name) * 10,
        )
        metrics.append(metric_row)
        clean_predictions.loc[split_index, "readout_probability_X"] = predicted[
            :, 0
        ]
        clean_predictions.loc[split_index, "readout_probability_Y"] = predicted[
            :, 1
        ]

    all_x = np.asarray(activation, dtype=np.float32)
    all_predicted = predict_probability(
        bundle["model"], bundle["scaler"].transform(all_x)
    )
    predictions = metadata.copy()
    predictions["readout_probability_X"] = all_predicted[:, 0]
    predictions["readout_probability_Y"] = all_predicted[:, 1]
    gold_index = predictions["gold_option_label"].map(MAPPING).to_numpy(dtype=int)
    wrong_index = predictions["wrong_option_label"].map(MAPPING).to_numpy(dtype=int)
    target_all = predictions[["probability_X", "probability_Y"]].to_numpy(
        dtype=np.float64
    )
    predictions["target_gold_probability"] = target_all[
        np.arange(len(predictions)), gold_index
    ]
    predictions["target_wrong_probability"] = target_all[
        np.arange(len(predictions)), wrong_index
    ]
    predictions["readout_gold_probability"] = all_predicted[
        np.arange(len(predictions)), gold_index
    ]
    predictions["readout_wrong_probability"] = all_predicted[
        np.arange(len(predictions)), wrong_index
    ]
    predictions["target_gold_vs_wrong_gap"] = pair_gap(
        target_all, gold_index, wrong_index
    )
    predictions["readout_gold_vs_wrong_gap"] = pair_gap(
        all_predicted, gold_index, wrong_index
    )
    predictions["readout_pred_label"] = [
        LABELS[index] for index in all_predicted.argmax(axis=1)
    ]
    predictions["readout_gold_preferred"] = (
        all_predicted.argmax(axis=1) == gold_index
    )
    natural_rows, natural_summary = natural_recovery_analysis(
        predictions,
        samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": outdir / "实验E_22_clean_heldout_metrics.csv",
        "tuning": outdir / "实验E_22_dev_tuning.csv",
        "predictions": outdir / "实验E_22_all_diagnostic_predictions.csv.gz",
        "model": outdir / "实验E_22_readout.joblib",
        "natural_rows": outdir / "实验E_22_natural_surface_recovery.csv",
        "natural_summary": outdir / "实验E_22_natural_summary.json",
        "manifest": outdir / "实验E_22_manifest.json",
    }
    pd.DataFrame(metrics).to_csv(paths["metrics"], index=False)
    tuning.to_csv(paths["tuning"], index=False)
    predictions.to_csv(paths["predictions"], index=False, compression="gzip")
    natural_rows.to_csv(paths["natural_rows"], index=False)
    paths["natural_summary"].write_text(
        json.dumps(natural_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(bundle, paths["model"])
    json_dump(
        {
            "experiment": "E_22_open_dialogue_probability_readout",
            "command": command_record(),
            "model_name": args.model_name,
            "run_manifest": file_fingerprint(run_dir / "manifest.json"),
            "metadata": file_fingerprint(run_dir / "metadata.csv"),
            "activation": {
                **file_fingerprint(activation_path),
                "shape": list(activation.shape),
                "dtype": str(activation.dtype),
            },
            "split_manifest": file_fingerprint(Path(args.split_manifest)),
            "split_counts": counts,
            "target": (
                "model final X/Y probability at a randomized numeric-candidate "
                "diagnostic appended after unrestricted free-form dialogue"
            ),
            "training": "clean T0 only; soft-label CE/KL; dev-only L2 selection",
            "selected_l2": bundle["selected_l2"],
            "natural_summary": natural_summary,
            "compute": {"device": "cpu", "threads": int(args.threads)},
            "outputs": {
                name: str(path.resolve()) for name, path in paths.items()
            },
        },
        paths["manifest"],
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(natural_summary, ensure_ascii=False, indent=2))
    print(f"[OK] {paths['manifest']}")


if __name__ == "__main__":
    main()
