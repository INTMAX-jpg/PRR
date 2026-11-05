#!/usr/bin/env python3
"""Experiment E_10: clean-only four-option probability readout.

The target is the model's final distribution restricted to A/B/C/D:

    q = softmax([logit_A, logit_B, logit_C, logit_D])

The probe is an affine map from the fixed hidden state followed by a four-way
softmax.  It is trained with soft-label cross entropy (equivalently KL up to a
target-only constant), so both the target and prediction are valid
four-option probability distributions whose rows sum to one.

Training uses every clean T0 row in the train split, without a
model-correctness or future-T4-recovery filter.  Hyperparameters are selected
on clean dev only and the frozen probe is then evaluated on test and applied
to all stored turns.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_pressure.config import load_config  # noqa: E402
from 实验E_00_common import (  # noqa: E402
    assert_row_alignment,
    assert_split_disjoint,
    clean_t0_frame,
    command_record,
    feature_key,
    file_fingerprint,
    json_dump,
    json_load,
    label_to_index,
    labels_as_indices,
    resolve_feature_path,
)


METHOD = "option_probability_affine_softmax"


def parse_grid(value: str) -> List[float]:
    result = [float(item) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("Hyperparameter grid must not be empty")
    if any(item < 0 for item in result):
        raise ValueError("L2 grid must be non-negative")
    return result


def option_softmax(values: np.ndarray) -> np.ndarray:
    """Softmax over the four candidate-option logits only."""
    array = np.asarray(values, dtype=np.float64)
    shifted = array - array.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    result = exponent / exponent.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite option probability")
    return result


def pair_gap(
    probabilities: np.ndarray,
    true_index: np.ndarray,
    wrong_index: np.ndarray,
) -> np.ndarray:
    row = np.arange(len(probabilities))
    return (
        np.asarray(probabilities)[row, np.asarray(true_index, dtype=int)]
        - np.asarray(probabilities)[row, np.asarray(wrong_index, dtype=int)]
    )


def selected_probability(
    probabilities: np.ndarray, selected_index: np.ndarray
) -> np.ndarray:
    return np.asarray(probabilities)[
        np.arange(len(probabilities)), np.asarray(selected_index, dtype=int)
    ]


def mean_kl(target: np.ndarray, predicted: np.ndarray) -> float:
    q = np.clip(np.asarray(target, dtype=np.float64), 1e-12, 1.0)
    p = np.clip(np.asarray(predicted, dtype=np.float64), 1e-12, 1.0)
    return float(np.mean(np.sum(q * (np.log(q) - np.log(p)), axis=1)))


def mean_js(target: np.ndarray, predicted: np.ndarray) -> float:
    q = np.clip(np.asarray(target, dtype=np.float64), 1e-12, 1.0)
    p = np.clip(np.asarray(predicted, dtype=np.float64), 1e-12, 1.0)
    midpoint = 0.5 * (q + p)
    return float(
        0.5
        * np.mean(
            np.sum(q * (np.log(q) - np.log(midpoint)), axis=1)
            + np.sum(p * (np.log(p) - np.log(midpoint)), axis=1)
        )
    )


def safe_correlation(
    first: np.ndarray, second: np.ndarray, *, rank: bool = False
) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
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


def fit_affine_softmax(
    *,
    x_train: np.ndarray,
    target_probability: np.ndarray,
    l2: float,
    max_iter: int,
    threads: int,
) -> Dict[str, Any]:
    torch.set_num_threads(max(1, int(threads)))
    x_tensor = torch.as_tensor(
        np.asarray(x_train, dtype=np.float64), dtype=torch.float64
    )
    q_tensor = torch.as_tensor(
        np.asarray(target_probability, dtype=np.float64), dtype=torch.float64
    )
    n_features = int(x_tensor.shape[1])
    n_labels = int(q_tensor.shape[1])
    weight = torch.nn.Parameter(
        torch.zeros((n_features, n_labels), dtype=torch.float64)
    )
    bias = torch.nn.Parameter(torch.zeros(n_labels, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=int(max_iter),
        max_eval=int(max_iter * 2),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = x_tensor @ weight + bias
        cross_entropy = -(q_tensor * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        regularization = 0.5 * float(l2) * weight.square().sum()
        loss = cross_entropy + regularization
        loss.backward()
        return loss

    final_loss = float(optimizer.step(closure).detach().cpu())
    with torch.no_grad():
        logits = x_tensor @ weight + bias
        data_loss = float(
            (-(q_tensor * F.log_softmax(logits, dim=1)).sum(dim=1).mean())
            .detach()
            .cpu()
        )
    optimizer.zero_grad(set_to_none=True)
    logits = x_tensor @ weight + bias
    objective_check = (
        -(q_tensor * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        + 0.5 * float(l2) * weight.square().sum()
    )
    objective_check.backward()
    max_abs_gradient = float(
        max(
            weight.grad.detach().abs().max().cpu(),
            bias.grad.detach().abs().max().cpu(),
        )
    )
    return {
        "coef": weight.detach().cpu().numpy(),
        "bias": bias.detach().cpu().numpy(),
        "l2": float(l2),
        "optimizer_returned_loss": final_loss,
        "train_soft_cross_entropy": data_loss,
        "max_abs_gradient": max_abs_gradient,
        "objective_check": float(objective_check.detach().cpu()),
    }


def predict_probability(model: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    scores = (
        np.asarray(x, dtype=np.float64) @ np.asarray(model["coef"], dtype=np.float64)
        + np.asarray(model["bias"], dtype=np.float64)
    )
    return option_softmax(scores)


def probability_metrics(
    *,
    split: str,
    predicted_probability: np.ndarray,
    target_probability: np.ndarray,
    target_preference: np.ndarray,
    gold: np.ndarray,
    wrong: np.ndarray,
) -> Dict[str, Any]:
    predicted = np.asarray(predicted_probability, dtype=np.float64)
    target = np.asarray(target_probability, dtype=np.float64)
    prediction = predicted.argmax(axis=1)
    actual_gap = pair_gap(target, gold, wrong)
    predicted_gap = pair_gap(predicted, gold, wrong)
    row: Dict[str, Any] = {
        "method": METHOD,
        "split": split,
        "n_rows": int(len(predicted)),
        "preference_agreement": float((prediction == target_preference).mean()),
        "preference_balanced_agreement": float(
            balanced_accuracy_score(target_preference, prediction)
        ),
        "gold_accuracy": float((prediction == gold).mean()),
        "option_probability_kl": mean_kl(target, predicted),
        "option_probability_js": mean_js(target, predicted),
        "option_probability_mae": float(np.mean(np.abs(target - predicted))),
        "option_probability_rmse": float(np.sqrt(np.mean((target - predicted) ** 2))),
        "option_probability_brier": float(
            np.mean(np.sum((target - predicted) ** 2, axis=1))
        ),
        "target_probability_sum_max_abs_error": float(
            np.max(np.abs(target.sum(axis=1) - 1.0))
        ),
        "predicted_probability_sum_max_abs_error": float(
            np.max(np.abs(predicted.sum(axis=1) - 1.0))
        ),
        "probability_gap_sign_agreement": float(
            (np.signbit(predicted_gap) == np.signbit(actual_gap)).mean()
        ),
        "probability_gap_pearson": safe_correlation(actual_gap, predicted_gap),
        "probability_gap_spearman": safe_correlation(
            actual_gap, predicted_gap, rank=True
        ),
    }
    try:
        row["option_probability_r2"] = float(
            r2_score(target.reshape(-1), predicted.reshape(-1))
        )
        row["probability_gap_r2"] = float(r2_score(actual_gap, predicted_gap))
    except Exception:
        row["option_probability_r2"] = math.nan
        row["probability_gap_r2"] = math.nan
    return row


def fit_and_select(
    *,
    x_train: np.ndarray,
    train_probability: np.ndarray,
    x_dev: np.ndarray,
    dev_probability: np.ndarray,
    dev_preference: np.ndarray,
    dev_gold: np.ndarray,
    dev_wrong: np.ndarray,
    l2_grid: Sequence[float],
    max_iter: int,
    threads: int,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    scaler = StandardScaler().fit(x_train)
    z_train = scaler.transform(x_train)
    z_dev = scaler.transform(x_dev)
    models: Dict[float, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for l2 in l2_grid:
        model = fit_affine_softmax(
            x_train=z_train,
            target_probability=train_probability,
            l2=float(l2),
            max_iter=max_iter,
            threads=threads,
        )
        predicted = predict_probability(model, z_dev)
        metrics = probability_metrics(
            split="dev",
            predicted_probability=predicted,
            target_probability=dev_probability,
            target_preference=dev_preference,
            gold=dev_gold,
            wrong=dev_wrong,
        )
        rows.append(
            {
                "method": METHOD,
                "parameter_name": "l2",
                "parameter": float(l2),
                "train_soft_cross_entropy": model["train_soft_cross_entropy"],
                "max_abs_gradient": model["max_abs_gradient"],
                **{
                    f"dev_{key}": value
                    for key, value in metrics.items()
                    if key not in {"method", "split", "n_rows"}
                },
            }
        )
        models[float(l2)] = model
    tuning = pd.DataFrame(rows).sort_values(
        [
            "dev_option_probability_kl",
            "dev_preference_balanced_agreement",
            "dev_preference_agreement",
            "parameter",
        ],
        ascending=[True, False, False, False],
    )
    selected_l2 = float(tuning.iloc[0]["parameter"])
    tuning["selected"] = tuning["parameter"].eq(selected_l2)
    return {
        "scaler": scaler,
        "model": models[selected_l2],
        "selected_l2": selected_l2,
    }, tuning.sort_values("parameter").reset_index(drop=True)


def add_bootstrap_fields(
    destination: Dict[str, Any],
    *,
    prefix: str,
    values: Iterable[float],
    n_bootstrap: int,
    seed: int,
) -> None:
    result = bootstrap_mean(values, n_bootstrap=n_bootstrap, seed=seed)
    destination[prefix] = result["estimate"]
    destination[f"{prefix}_ci_low"] = result["ci_low"]
    destination[f"{prefix}_ci_high"] = result["ci_high"]


def natural_t4_analysis(
    *,
    scores: pd.DataFrame,
    pressure_trajectory: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    frame = scores[
        scores["split"].astype(str).eq("test")
        & scores["trajectory_type"].astype(str).eq(pressure_trajectory)
        & scores["turn_index"].astype(int).isin([0, 3, 4])
    ].copy()
    records: List[Dict[str, Any]] = []
    for base_id, group in frame.groupby("base_id", sort=True):
        by_turn = {int(row["turn_index"]): row for _, row in group.iterrows()}
        if not all(turn in by_turn for turn in [0, 3, 4]):
            continue
        t0, t3, t4 = by_turn[0], by_turn[3], by_turn[4]
        if not (
            str(t0["model_pred_label"]) == str(t0["true_label"])
            and str(t3["model_pred_label"]) == str(t3["wrong_label"])
            and str(t4["model_pred_label"]) == str(t4["true_label"])
        ):
            continue
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
                "base_id": str(base_id),
                "true_label": str(t4["true_label"]),
                "wrong_label": str(t4["wrong_label"]),
                "T0_readout_pred_label": str(t0["readout_pred_label"]),
                "T3_readout_pred_label": str(t3["readout_pred_label"]),
                "T4_readout_pred_label": str(t4["readout_pred_label"]),
                "T0_readout_gold_probability": float(
                    t0["readout_gold_probability"]
                ),
                "T3_readout_gold_probability": float(
                    t3["readout_gold_probability"]
                ),
                "T4_readout_gold_probability": float(
                    t4["readout_gold_probability"]
                ),
                "T0_output_gold_probability": float(
                    t0["output_gold_probability"]
                ),
                "T3_output_gold_probability": float(
                    t3["output_gold_probability"]
                ),
                "T4_output_gold_probability": float(
                    t4["output_gold_probability"]
                ),
                "T0_readout_probability_gap": float(
                    t0["readout_true_vs_wrong_probability_gap"]
                ),
                "T3_readout_probability_gap": float(
                    t3["readout_true_vs_wrong_probability_gap"]
                ),
                "T4_readout_probability_gap": float(
                    t4["readout_true_vs_wrong_probability_gap"]
                ),
                "T0_output_probability_gap": float(
                    t0["output_true_vs_wrong_probability_gap"]
                ),
                "T3_output_probability_gap": float(
                    t3["output_true_vs_wrong_probability_gap"]
                ),
                "T4_output_probability_gap": float(
                    t4["output_true_vs_wrong_probability_gap"]
                ),
                "latent_gold_probability_decline_T0_minus_T4": (
                    latent_gold_decline
                ),
                "output_gold_probability_decline_T0_minus_T4": (
                    output_gold_decline
                ),
                "gold_probability_dissociation_gap": (
                    latent_gold_decline - output_gold_decline
                ),
                "latent_probability_gap_decline_T0_minus_T4": (
                    latent_gap_decline
                ),
                "output_probability_gap_decline_T0_minus_T4": (
                    output_gap_decline
                ),
                "probability_gap_dissociation_gap": (
                    latent_gap_decline - output_gap_decline
                ),
                "T3_to_T4_readout_gold_probability_recovery": float(
                    t4["readout_gold_probability"]
                    - t3["readout_gold_probability"]
                ),
                "T3_to_T4_output_gold_probability_recovery": float(
                    t4["output_gold_probability"] - t3["output_gold_probability"]
                ),
                "T4_readout_pairwise_wrong_aligned": bool(
                    t4["readout_true_vs_wrong_probability_gap"] < 0
                ),
                "T4_readout_exact_designated_wrong": bool(
                    str(t4["readout_pred_label"]) == str(t4["wrong_label"])
                ),
                "gold_probability_dissociation_positive": bool(
                    latent_gold_decline > output_gold_decline
                ),
                "probability_gap_dissociation_positive": bool(
                    latent_gap_decline > output_gap_decline
                ),
            }
        )
    recovered = pd.DataFrame(records)
    summary: Dict[str, Any] = {
        "method": METHOD,
        "n_surface_recovered": int(len(recovered)),
    }
    if not recovered.empty:
        for offset, (name, column) in enumerate(
            [
                (
                    "latent_gold_probability_decline_T0_minus_T4",
                    "latent_gold_probability_decline_T0_minus_T4",
                ),
                (
                    "output_gold_probability_decline_T0_minus_T4",
                    "output_gold_probability_decline_T0_minus_T4",
                ),
                (
                    "gold_probability_dissociation_gap",
                    "gold_probability_dissociation_gap",
                ),
                (
                    "latent_probability_gap_decline_T0_minus_T4",
                    "latent_probability_gap_decline_T0_minus_T4",
                ),
                (
                    "output_probability_gap_decline_T0_minus_T4",
                    "output_probability_gap_decline_T0_minus_T4",
                ),
                (
                    "probability_gap_dissociation_gap",
                    "probability_gap_dissociation_gap",
                ),
                (
                    "T3_to_T4_readout_gold_probability_recovery",
                    "T3_to_T4_readout_gold_probability_recovery",
                ),
                (
                    "T3_to_T4_output_gold_probability_recovery",
                    "T3_to_T4_output_gold_probability_recovery",
                ),
                (
                    "T4_readout_pairwise_wrong_alignment_rate",
                    "T4_readout_pairwise_wrong_aligned",
                ),
                (
                    "T4_readout_exact_designated_wrong_rate",
                    "T4_readout_exact_designated_wrong",
                ),
                (
                    "gold_probability_dissociation_positive_rate",
                    "gold_probability_dissociation_positive",
                ),
                (
                    "probability_gap_dissociation_positive_rate",
                    "probability_gap_dissociation_positive",
                ),
            ]
        ):
            add_bootstrap_fields(
                summary,
                prefix=name,
                values=recovered[column].astype(float),
                n_bootstrap=n_bootstrap,
                seed=seed + offset,
            )
        summary["T4_readout_pairwise_wrong_alignment_n"] = int(
            recovered["T4_readout_pairwise_wrong_aligned"].sum()
        )
        summary["T4_readout_exact_designated_wrong_n"] = int(
            recovered["T4_readout_exact_designated_wrong"].sum()
        )
        summary["gold_probability_dissociation_positive_n"] = int(
            recovered["gold_probability_dissociation_positive"].sum()
        )
        summary["probability_gap_dissociation_positive_n"] = int(
            recovered["probability_gap_dissociation_positive"].sum()
        )
    population = {
        "n_test_pressure_rows_T0_T3_T4": int(len(frame)),
        "n_surface_recovered": int(len(recovered)),
        "selection_rule": (
            "test wrong_pressure only: actual output T0=gold, "
            "T3=designated wrong, T4=gold; no readout field is used for selection"
        ),
        "training_overlap_rule": (
            "test base IDs, including their T0 rows, are excluded from readout training"
        ),
    }
    return pd.DataFrame([summary]), recovered, population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--selection", default=None)
    parser.add_argument(
        "--layer_field",
        choices=["decoder_layer", "causal_layer"],
        default="causal_layer",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--site", default="resid_pre")
    parser.add_argument("--clean_trajectory", default="wrong_pressure")
    parser.add_argument("--pressure_trajectory", default="wrong_pressure")
    parser.add_argument("--clean_turn", type=int, default=0)
    parser.add_argument(
        "--l2_grid",
        default=(
            "0,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,"
            "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1"
        ),
    )
    parser.add_argument("--max_iter", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(
        args.seed
        if args.seed is not None
        else cfg.get("experiment", {}).get("seed", 42)
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    labels = [str(value) for value in cfg["experiment"].get("labels", "ABCD")]
    if len(labels) != 4:
        raise ValueError(f"Expected four option labels, got {labels}")
    mapping = label_to_index(labels)
    logit_columns = [f"logit_{label}" for label in labels]
    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = json_load(run_dir / "manifest.json")
    metadata = pd.read_csv(run_dir / "metadata.csv")
    missing = [column for column in logit_columns if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing option-logit columns: {missing}")

    if args.layer is not None:
        layer = int(args.layer)
        layer_source = "explicit_cli"
    elif args.selection:
        selection = json_load(Path(args.selection))
        layer = int(selection[args.layer_field])
        layer_source = f"{args.layer_field}_from_dev_only_Experiment_E_01"
    else:
        raise ValueError("Supply --layer or --selection")
    key = feature_key(args.site, layer)
    activation_path = resolve_feature_path(run_dir, manifest, key)
    activation = np.load(activation_path, mmap_mode="r")
    assert_row_alignment(metadata, len(activation))

    clean = clean_t0_frame(
        metadata,
        trajectory_type=args.clean_trajectory,
        turn_index=args.clean_turn,
    ).sort_values("row_idx").reset_index(drop=True)
    assert_split_disjoint(clean)
    for required_split in ["train", "dev", "test"]:
        if not clean["split"].astype(str).eq(required_split).any():
            raise ValueError(f"Missing clean split={required_split}")

    clean_logits = clean[logit_columns].to_numpy(dtype=np.float64)
    clean_probability = option_softmax(clean_logits)
    actual_preference = clean_probability.argmax(axis=1)
    stored_preference = labels_as_indices(clean["pred_label"], mapping)
    if not np.array_equal(actual_preference, stored_preference):
        mismatch = int((actual_preference != stored_preference).sum())
        raise ValueError(
            f"Stored pred_label disagrees with option-only argmax on {mismatch} clean rows"
        )
    gold = labels_as_indices(clean["true_label"], mapping)
    wrong = labels_as_indices(clean["wrong_label"], mapping)
    clean_x = np.asarray(
        activation[clean["row_idx"].astype(int).to_numpy()], dtype=np.float32
    )
    split_indices = {
        split: clean.index[clean["split"].astype(str).eq(split)].to_numpy()
        for split in ["train", "dev", "test"]
    }
    train_index = split_indices["train"]
    dev_index = split_indices["dev"]
    bundle, tuning = fit_and_select(
        x_train=clean_x[train_index],
        train_probability=clean_probability[train_index],
        x_dev=clean_x[dev_index],
        dev_probability=clean_probability[dev_index],
        dev_preference=actual_preference[dev_index],
        dev_gold=gold[dev_index],
        dev_wrong=wrong[dev_index],
        l2_grid=parse_grid(args.l2_grid),
        max_iter=args.max_iter,
        threads=args.threads,
    )
    bundle.update(
        {
            "experiment": "E_10_option_probability_readout",
            "feature_key": key,
            "layer_source": layer_source,
            "labels": labels,
            "target_definition": (
                "softmax over final logit_A/logit_B/logit_C/logit_D only"
            ),
            "clean_training_rule": (
                f"one {args.clean_trajectory} T{args.clean_turn} row per train base; "
                "no model-correctness or future-recovery filter"
            ),
        }
    )

    metric_rows: List[Dict[str, Any]] = []
    clean_predictions = clean[
        [
            "row_idx",
            "base_id",
            "split",
            "trajectory_type",
            "turn_index",
            "true_label",
            "wrong_label",
            "pred_label",
            *logit_columns,
        ]
    ].copy()
    for label_index, label in enumerate(labels):
        clean_predictions[f"target_option_probability_{label}"] = clean_probability[
            :, label_index
        ]
    for split, indices in split_indices.items():
        z = bundle["scaler"].transform(clean_x[indices])
        predicted = predict_probability(bundle["model"], z)
        metric_rows.append(
            {
                "feature_key": key,
                "layer_source": layer_source,
                "selected_l2": bundle["selected_l2"],
                **probability_metrics(
                    split=split,
                    predicted_probability=predicted,
                    target_probability=clean_probability[indices],
                    target_preference=actual_preference[indices],
                    gold=gold[indices],
                    wrong=wrong[indices],
                ),
            }
        )
        clean_predictions.loc[indices, "readout_pred_label"] = [
            labels[index] for index in predicted.argmax(axis=1)
        ]
        for label_index, label in enumerate(labels):
            clean_predictions.loc[
                indices, f"readout_option_probability_{label}"
            ] = predicted[:, label_index]
        clean_predictions.loc[
            indices, "target_true_vs_wrong_probability_gap"
        ] = pair_gap(clean_probability[indices], gold[indices], wrong[indices])
        clean_predictions.loc[
            indices, "readout_true_vs_wrong_probability_gap"
        ] = pair_gap(predicted, gold[indices], wrong[indices])

    all_scores = metadata[
        [
            "row_idx",
            "base_id",
            "split",
            "trajectory_type",
            "turn_index",
            "true_label",
            "wrong_label",
            "pred_label",
            *logit_columns,
        ]
    ].copy()
    all_scores = all_scores.rename(columns={"pred_label": "model_pred_label"})
    all_gold = labels_as_indices(metadata["true_label"], mapping)
    all_wrong = labels_as_indices(metadata["wrong_label"], mapping)
    for label in labels:
        all_scores[f"output_option_probability_{label}"] = np.nan
        all_scores[f"readout_option_probability_{label}"] = np.nan
    all_scores["readout_pred_label"] = ""
    all_scores["output_gold_probability"] = np.nan
    all_scores["readout_gold_probability"] = np.nan
    all_scores["output_true_vs_wrong_probability_gap"] = np.nan
    all_scores["readout_true_vs_wrong_probability_gap"] = np.nan

    for start in range(0, len(metadata), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(metadata))
        x_batch = np.asarray(activation[start:stop], dtype=np.float32)
        z_batch = bundle["scaler"].transform(x_batch)
        predicted = predict_probability(bundle["model"], z_batch)
        output_probability = option_softmax(
            metadata.iloc[start:stop][logit_columns].to_numpy(dtype=np.float64)
        )
        for label_index, label in enumerate(labels):
            all_scores.loc[
                start : stop - 1, f"output_option_probability_{label}"
            ] = output_probability[:, label_index]
            all_scores.loc[
                start : stop - 1, f"readout_option_probability_{label}"
            ] = predicted[:, label_index]
        all_scores.loc[start : stop - 1, "readout_pred_label"] = [
            labels[index] for index in predicted.argmax(axis=1)
        ]
        all_scores.loc[
            start : stop - 1, "output_gold_probability"
        ] = selected_probability(output_probability, all_gold[start:stop])
        all_scores.loc[
            start : stop - 1, "readout_gold_probability"
        ] = selected_probability(predicted, all_gold[start:stop])
        all_scores.loc[
            start : stop - 1, "output_true_vs_wrong_probability_gap"
        ] = pair_gap(output_probability, all_gold[start:stop], all_wrong[start:stop])
        all_scores.loc[
            start : stop - 1, "readout_true_vs_wrong_probability_gap"
        ] = pair_gap(predicted, all_gold[start:stop], all_wrong[start:stop])

    target_sum_error = float(
        np.max(
            np.abs(
                all_scores[
                    [f"output_option_probability_{label}" for label in labels]
                ].sum(axis=1)
                - 1.0
            )
        )
    )
    readout_sum_error = float(
        np.max(
            np.abs(
                all_scores[
                    [f"readout_option_probability_{label}" for label in labels]
                ].sum(axis=1)
                - 1.0
            )
        )
    )
    if target_sum_error > 1e-10 or readout_sum_error > 1e-10:
        raise AssertionError(
            f"Probability normalization failed: target={target_sum_error}, "
            f"readout={readout_sum_error}"
        )

    natural_summary, natural_items, natural_population = natural_t4_analysis(
        scores=all_scores,
        pressure_trajectory=args.pressure_trajectory,
        n_bootstrap=args.bootstrap_samples,
        seed=seed,
    )

    paths = {
        "metrics": outdir / "实验E_10_clean_heldout_probability_metrics.csv",
        "tuning": outdir / "实验E_10_dev_tuning.csv",
        "clean_predictions": outdir / "实验E_10_clean_predictions.csv.gz",
        "all_scores": outdir / "实验E_10_all_turn_probability_scores.csv.gz",
        "natural_summary": outdir / "实验E_10_natural_T4_probability_summary.csv",
        "natural_items": outdir / "实验E_10_natural_T4_probability_items.csv.gz",
        "models": outdir / "实验E_10_models.joblib",
        "manifest": outdir / "实验E_10_manifest.json",
    }
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    tuning.to_csv(paths["tuning"], index=False)
    clean_predictions.to_csv(paths["clean_predictions"], index=False, compression="gzip")
    all_scores.to_csv(paths["all_scores"], index=False, compression="gzip")
    natural_summary.to_csv(paths["natural_summary"], index=False)
    natural_items.to_csv(paths["natural_items"], index=False, compression="gzip")
    joblib.dump(bundle, paths["models"])
    json_dump(
        {
            "experiment": "E_10_option_probability_readout",
            "command": command_record(),
            "config": file_fingerprint(Path(args.config)),
            "run_dir": str(run_dir.resolve()),
            "metadata": file_fingerprint(run_dir / "metadata.csv"),
            "activation": {
                **file_fingerprint(activation_path),
                "shape": list(activation.shape),
                "dtype": str(activation.dtype),
            },
            "feature_key": key,
            "layer_source": layer_source,
            "labels": labels,
            "target_definition": (
                "A/B/C/D option-only probabilities: softmax over exactly the "
                "four saved option logits; every row sums to one"
            ),
            "probe_definition": (
                "affine hidden-state translator followed by four-way softmax"
            ),
            "training_objective": (
                "soft-label cross entropy / KL with L2 on the affine weight"
            ),
            "clean_training_rule": (
                f"one {args.clean_trajectory} T{args.clean_turn} row per train base; "
                "all clean train items; no correctness or future-recovery filter"
            ),
            "selection_rule": (
                "L2 selected by minimum clean-dev option-probability KL, then "
                "balanced agreement; test evaluated once"
            ),
            "selected_l2": bundle["selected_l2"],
            "n_train": int(len(train_index)),
            "n_dev": int(len(dev_index)),
            "n_test": int(len(split_indices["test"])),
            "target_probability_sum_max_abs_error_all_rows": target_sum_error,
            "readout_probability_sum_max_abs_error_all_rows": readout_sum_error,
            "natural_t4_population": natural_population,
            "compute": {
                "device": "cpu",
                "threads": int(args.threads),
                "gpu_count_used": 0,
            },
            "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        },
        paths["manifest"],
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(natural_summary.to_string(index=False))
    print(f"[OK] {paths['manifest']}")


if __name__ == "__main__":
    main()
