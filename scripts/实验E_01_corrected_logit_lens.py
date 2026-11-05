#!/usr/bin/env python3
"""Experiment E_01: corrected logit lens on saved external-QA activations.

For each saved resid_pre vector this script computes

    lm_head(final_norm(h_layer))

and, as an explicit implementation check, the uncorrected raw projection
lm_head(h_layer). Layer selection uses dev data only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_pressure.config import load_config  # noqa: E402
from latent_pressure.hf_model import load_tokenizer_and_model  # noqa: E402
from 实验E_00_common import (  # noqa: E402
    assert_row_alignment,
    assert_split_disjoint,
    clean_t0_frame,
    cluster_bootstrap_mean,
    command_record,
    feature_key,
    file_fingerprint,
    gold_margin,
    json_dump,
    json_load,
    label_to_index,
    labels_as_indices,
    multiclass_metrics,
    pair_margin,
    project_label_logits,
    resolve_feature_path,
)


def available_layers(manifest: Dict[str, Any], site: str) -> List[int]:
    prefix = f"{site}_L"
    layers = []
    for key in manifest.get("feature_paths", {}):
        if str(key).startswith(prefix):
            layers.append(int(str(key)[len(prefix) :]))
    return sorted(set(layers))


def project_array(
    model,
    array: np.ndarray,
    label_token_ids: List[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    corrected: List[np.ndarray] = []
    raw: List[np.ndarray] = []
    model_dtype = next(model.parameters()).dtype
    for start in range(0, len(array), int(batch_size)):
        stop = min(start + int(batch_size), len(array))
        hidden = torch.from_numpy(
            np.array(array[start:stop], dtype=np.float32, copy=True)
        ).to(dtype=model_dtype)
        corrected.append(
            project_label_logits(
                model,
                hidden,
                label_token_ids=label_token_ids,
                apply_final_norm=True,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        raw.append(
            project_label_logits(
                model,
                hidden,
                label_token_ids=label_token_ids,
                apply_final_norm=False,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(corrected), np.concatenate(raw)


def evaluate_slice(
    frame: pd.DataFrame,
    scores: np.ndarray,
    labels: List[str],
    label_mapping: Dict[str, int],
    lens_type: str,
    layer: int,
    slice_name: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    row_indices = frame["row_idx"].astype(int).to_numpy()
    slice_scores = scores[row_indices]
    y = labels_as_indices(frame["true_label"], label_mapping)
    wrong = labels_as_indices(frame["wrong_label"], label_mapping)
    pred = slice_scores.argmax(axis=1)
    metrics = multiclass_metrics(y, slice_scores)
    gold_m = gold_margin(slice_scores, y)
    pair_m = pair_margin(slice_scores, y, wrong)
    correct = (pred == y).astype(float)
    wrong_pred = (pred == wrong).astype(float)
    model_pred = labels_as_indices(frame["pred_label"], label_mapping)
    accuracy_ci = cluster_bootstrap_mean(
        correct,
        frame["base_id"].astype(str),
        n_bootstrap=bootstrap_samples,
        seed=seed + layer,
    )
    margin_ci = cluster_bootstrap_mean(
        pair_m,
        frame["base_id"].astype(str),
        n_bootstrap=bootstrap_samples,
        seed=seed + 1000 + layer,
    )
    return {
        "lens_type": lens_type,
        "layer": int(layer),
        "slice": slice_name,
        "split": str(frame["split"].iloc[0]) if frame["split"].nunique() == 1 else "mixed",
        "trajectory_type": (
            str(frame["trajectory_type"].iloc[0])
            if frame["trajectory_type"].nunique() == 1
            else "mixed"
        ),
        "turn_index": (
            int(frame["turn_index"].iloc[0])
            if frame["turn_index"].nunique() == 1
            else -1
        ),
        "n_rows": int(len(frame)),
        "n_base": int(frame["base_id"].nunique()),
        **metrics,
        "accuracy_ci_low": accuracy_ci["ci_low"],
        "accuracy_ci_high": accuracy_ci["ci_high"],
        "gold_margin_mean": float(gold_m.mean()),
        "true_vs_wrong_margin_mean": float(pair_m.mean()),
        "true_vs_wrong_margin_ci_low": margin_ci["ci_low"],
        "true_vs_wrong_margin_ci_high": margin_ci["ci_high"],
        "designated_wrong_prediction_rate": float(wrong_pred.mean()),
        "agreement_with_final_output": float((pred == model_pred).mean()),
        "final_output_accuracy": float((model_pred == y).mean()),
    }


def metric_rows_for_layer(
    meta: pd.DataFrame,
    corrected: np.ndarray,
    raw: np.ndarray,
    labels: List[str],
    label_mapping: Dict[str, int],
    layer: int,
    clean_trajectory: str,
    clean_turn: int,
    bootstrap_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    clean = clean_t0_frame(
        meta,
        trajectory_type=clean_trajectory,
        turn_index=clean_turn,
    )
    slices: List[tuple[str, pd.DataFrame]] = []
    for split, group in clean.groupby("split", sort=True):
        slices.append((f"clean_T0_{split}", group.copy()))
    for keys, group in meta.groupby(
        ["split", "trajectory_type", "turn_index"], sort=True
    ):
        split, trajectory, turn = keys
        if str(split) not in {"dev", "test"}:
            continue
        slices.append((f"{split}_{trajectory}_T{int(turn)}", group.copy()))

    for lens_type, scores in [("corrected", corrected), ("raw_no_final_norm", raw)]:
        for slice_name, frame in slices:
            if frame.empty:
                continue
            rows.append(
                evaluate_slice(
                    frame=frame,
                    scores=scores,
                    labels=labels,
                    label_mapping=label_mapping,
                    lens_type=lens_type,
                    layer=layer,
                    slice_name=slice_name,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--site", default=None)
    parser.add_argument("--layers", default=None, help="Comma-separated override")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    e_cfg = cfg.get("experiment_e", {})
    seed = int(args.seed if args.seed is not None else cfg["experiment"]["seed"])
    site = str(args.site or e_cfg.get("feature_site", "resid_pre"))
    clean_trajectory = str(
        e_cfg.get("clean_trajectory_type", "wrong_pressure")
    )
    clean_turn = int(e_cfg.get("clean_turn", 0))
    pressure_trajectory = str(
        e_cfg.get("pressure_trajectory_type", "wrong_pressure")
    )
    pressure_turn = int(e_cfg.get("pressure_turn", 3))
    batch_size = int(args.batch_size or e_cfg.get("batch_size", 256))
    bootstrap_samples = int(
        args.bootstrap_samples or e_cfg.get("bootstrap_samples", 2000)
    )

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    scores_dir = outdir / "实验E_01_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    manifest = json_load(run_dir / "manifest.json")
    meta = pd.read_csv(run_dir / "metadata.csv")
    assert_split_disjoint(clean_t0_frame(meta, clean_trajectory, clean_turn))
    labels = [str(value) for value in cfg["experiment"].get("labels", "ABCD")]
    mapping = label_to_index(labels)

    layers = available_layers(manifest, site)
    if args.layers:
        requested = [int(value) for value in args.layers.split(",") if value.strip()]
        missing = sorted(set(requested) - set(layers))
        if missing:
            raise ValueError(f"Requested layers not in manifest: {missing}")
        layers = requested
    if not layers:
        raise ValueError(f"No {site} activations found in manifest")

    tokenizer, model = load_tokenizer_and_model(cfg)
    saved_ids = manifest.get("label_token_ids", {})
    label_token_ids: List[int] = []
    label_token_texts: Dict[str, str] = {}
    for label in labels:
        if label in saved_ids:
            token_id = int(saved_ids[label])
        else:
            tokenized = tokenizer.encode(label, add_special_tokens=False)
            if len(tokenized) != 1:
                raise ValueError(f"Label {label!r} is not a single token: {tokenized}")
            token_id = int(tokenized[0])
        label_token_ids.append(token_id)
        label_token_texts[label] = tokenizer.decode([token_id])

    all_metrics: List[Dict[str, Any]] = []
    layer_selection_rows: List[Dict[str, Any]] = []
    activation_files: List[Dict[str, Any]] = []
    for layer in layers:
        key = feature_key(site, layer)
        activation_path = resolve_feature_path(run_dir, manifest, key)
        array = np.load(activation_path, mmap_mode="r")
        assert_row_alignment(meta, len(array))
        print(f"[实验E_01] projecting {key}: shape={array.shape}", flush=True)
        corrected, raw = project_array(
            model,
            array,
            label_token_ids=label_token_ids,
            batch_size=batch_size,
        )
        y_all = labels_as_indices(meta["true_label"], mapping)
        wrong_all = labels_as_indices(meta["wrong_label"], mapping)
        corrected_pred = corrected.argmax(axis=1)
        raw_pred = raw.argmax(axis=1)
        score_frame = pd.DataFrame({"row_idx": meta["row_idx"].astype(int)})
        for label_index, label in enumerate(labels):
            score_frame[f"corrected_logit_{label}"] = corrected[:, label_index]
            score_frame[f"raw_logit_{label}"] = raw[:, label_index]
        score_frame["corrected_pred_label"] = [
            labels[index] for index in corrected_pred
        ]
        score_frame["raw_pred_label"] = [labels[index] for index in raw_pred]
        score_frame["corrected_gold_margin"] = gold_margin(corrected, y_all)
        score_frame["corrected_true_vs_wrong_margin"] = pair_margin(
            corrected, y_all, wrong_all
        )
        score_frame["raw_true_vs_wrong_margin"] = pair_margin(
            raw, y_all, wrong_all
        )
        score_path = scores_dir / f"实验E_01_scores_{key}.csv.gz"
        score_frame.to_csv(score_path, index=False, compression="gzip")

        metrics = metric_rows_for_layer(
            meta=meta,
            corrected=corrected,
            raw=raw,
            labels=labels,
            label_mapping=mapping,
            layer=layer,
            clean_trajectory=clean_trajectory,
            clean_turn=clean_turn,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        all_metrics.extend(metrics)
        corrected_metrics = [row for row in metrics if row["lens_type"] == "corrected"]
        dev_clean = next(
            (row for row in corrected_metrics if row["slice"] == "clean_T0_dev"),
            None,
        )

        dev = meta["split"].astype(str).eq("dev")
        clean_mask = (
            dev
            & meta["trajectory_type"].astype(str).eq(clean_trajectory)
            & meta["turn_index"].astype(int).eq(clean_turn)
        )
        pressure_mask = (
            dev
            & meta["trajectory_type"].astype(str).eq(pressure_trajectory)
            & meta["turn_index"].astype(int).eq(pressure_turn)
        )
        clean_margin = pd.Series(
            pair_margin(
                corrected[clean_mask.to_numpy()],
                y_all[clean_mask.to_numpy()],
                wrong_all[clean_mask.to_numpy()],
            ),
            index=meta.loc[clean_mask, "base_id"].astype(str),
        )
        pressure_margin = pd.Series(
            pair_margin(
                corrected[pressure_mask.to_numpy()],
                y_all[pressure_mask.to_numpy()],
                wrong_all[pressure_mask.to_numpy()],
            ),
            index=meta.loc[pressure_mask, "base_id"].astype(str),
        )
        paired = pd.concat(
            [clean_margin.rename("clean"), pressure_margin.rename("pressure")],
            axis=1,
            join="inner",
        ).dropna()
        layer_selection_rows.append(
            {
                "layer": int(layer),
                "dev_clean_accuracy": (
                    float(dev_clean["accuracy"]) if dev_clean else np.nan
                ),
                "dev_clean_balanced_accuracy": (
                    float(dev_clean["balanced_accuracy"]) if dev_clean else np.nan
                ),
                "dev_pressure_n_pairs": int(len(paired)),
                "dev_clean_margin_mean": (
                    float(paired["clean"].mean()) if len(paired) else np.nan
                ),
                "dev_pressure_margin_mean": (
                    float(paired["pressure"].mean()) if len(paired) else np.nan
                ),
                "dev_pressure_margin_drop": (
                    float((paired["clean"] - paired["pressure"]).mean())
                    if len(paired)
                    else np.nan
                ),
            }
        )
        activation_files.append(
            {
                "feature_key": key,
                "path": str(activation_path),
                "size_bytes": activation_path.stat().st_size,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_path = outdir / "实验E_01_layer_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    selection_frame = pd.DataFrame(layer_selection_rows)
    selection_table_path = outdir / "实验E_01_dev_layer_selection.csv"
    selection_frame.to_csv(selection_table_path, index=False)

    if selection_frame["dev_clean_balanced_accuracy"].isna().all():
        raise ValueError("Cannot select decoder layer: no dev clean metrics")
    decoder_row = selection_frame.sort_values(
        ["dev_clean_balanced_accuracy", "layer"],
        ascending=[False, True],
    ).iloc[0]
    causal_candidates = selection_frame.dropna(subset=["dev_pressure_margin_drop"])
    if causal_candidates.empty:
        raise ValueError("Cannot select causal layer: no paired dev pressure rows")
    causal_row = causal_candidates.sort_values(
        ["dev_pressure_margin_drop", "layer"],
        ascending=[False, True],
    ).iloc[0]
    selection = {
        "selection_data": "dev_only",
        "decoder_selection_rule": "max corrected clean-T0 balanced accuracy; earliest layer breaks ties",
        "decoder_layer": int(decoder_row["layer"]),
        "decoder_dev_clean_balanced_accuracy": float(
            decoder_row["dev_clean_balanced_accuracy"]
        ),
        "causal_selection_rule": (
            "max paired dev corrected-lens true-vs-wrong margin drop "
            f"from {clean_trajectory} T{clean_turn} to "
            f"{pressure_trajectory} T{pressure_turn}; earliest layer breaks ties"
        ),
        "causal_layer": int(causal_row["layer"]),
        "causal_dev_pressure_margin_drop": float(
            causal_row["dev_pressure_margin_drop"]
        ),
    }
    selection_path = outdir / "实验E_01_selected_layers.json"
    json_dump(selection, selection_path)
    json_dump(
        {
            "experiment": "E_01_corrected_logit_lens",
            "command": command_record(),
            "config": file_fingerprint(Path(args.config)),
            "metadata": file_fingerprint(run_dir / "metadata.csv"),
            "run_manifest": file_fingerprint(run_dir / "manifest.json"),
            "run_dir": str(run_dir.resolve()),
            "outdir": str(outdir.resolve()),
            "site": site,
            "layers": layers,
            "labels": labels,
            "label_token_ids": dict(zip(labels, label_token_ids)),
            "label_token_texts": label_token_texts,
            "clean_trajectory_type": clean_trajectory,
            "clean_turn": clean_turn,
            "pressure_trajectory_type": pressure_trajectory,
            "pressure_turn": pressure_turn,
            "batch_size": batch_size,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "activation_files": activation_files,
            "outputs": {
                "metrics": str(metrics_path.resolve()),
                "selection_table": str(selection_table_path.resolve()),
                "selection": str(selection_path.resolve()),
                "scores_dir": str(scores_dir.resolve()),
            },
        },
        outdir / "实验E_01_manifest.json",
    )
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    print(f"[OK] {metrics_path}")


if __name__ == "__main__":
    main()
