#!/usr/bin/env python3
"""Experiment E_03: clean-to-pressure activation restoration with controls.

The causal layer is selected by E_01 on dev. This script evaluates an
outcome-independent sample of test base items and compares:
  * clean-state restoration;
  * an equal-norm orthogonal random direction;
  * a shuffled cross-item direction;
  * output-only correct-label logit bias.

It also checks whether the corrected-lens predicted margin change agrees with
the actual downstream output-margin change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_pressure.config import load_config  # noqa: E402
from latent_pressure.hf_model import (  # noqa: E402
    ActivationTracer,
    first_param_device,
    get_label_token_info,
    load_tokenizer_and_model,
    patch_activation,
)
from latent_pressure.prompts import apply_chat_template, build_messages  # noqa: E402
from latent_pressure.utils import read_jsonl, stable_hash  # noqa: E402
from 实验E_00_common import (  # noqa: E402
    cluster_bootstrap_mean,
    command_record,
    file_fingerprint,
    json_dump,
    json_load,
    label_to_index,
    project_label_logits,
    safe_spearman,
    through_origin_slope,
)


def record_index(dataset_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in read_jsonl(dataset_path):
        key = (str(record["base_id"]), str(record["trajectory_type"]))
        if key in result:
            raise ValueError(f"Duplicate dataset record for {key}")
        result[key] = record
    return result


def prediction_index(meta: pd.DataFrame) -> Dict[Tuple[str, str, int], str]:
    result: Dict[Tuple[str, str, int], str] = {}
    for _, row in meta.iterrows():
        key = (
            str(row["base_id"]),
            str(row["trajectory_type"]),
            int(row["turn_index"]),
        )
        if key in result:
            raise ValueError(f"Duplicate metadata prediction for {key}")
        result[key] = str(row["pred_label"])
    return result


def metadata_row_index(
    meta: pd.DataFrame,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for _, row in meta.iterrows():
        key = (
            str(row["base_id"]),
            str(row["trajectory_type"]),
            int(row["turn_index"]),
        )
        result[key] = row.to_dict()
    return result


def build_exact_prompt(
    tokenizer,
    cfg: Dict[str, Any],
    record: Dict[str, Any],
    turn_index: int,
    predictions: Dict[Tuple[str, str, int], str],
) -> str:
    history = []
    for prior_turn in range(turn_index):
        key = (
            str(record["base_id"]),
            str(record["trajectory_type"]),
            int(prior_turn),
        )
        if key not in predictions:
            raise KeyError(f"Missing recorded model history for {key}")
        history.append(predictions[key])
    messages = build_messages(record, turn_index, history, cfg)
    return apply_chat_template(tokenizer, messages)


def encode_prompt(tokenizer, model, prompt: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = first_param_device(model)
    result = {"input_ids": encoded["input_ids"].to(device)}
    if encoded.get("attention_mask") is not None:
        result["attention_mask"] = encoded["attention_mask"].to(device)
    return result


def forward_capture(
    tokenizer,
    model,
    prompt: str,
    site: str,
    layer: int,
    read_pos: int,
    label_token_ids: List[int],
) -> tuple[np.ndarray, torch.Tensor]:
    encoded = encode_prompt(tokenizer, model, prompt)
    with torch.inference_mode():
        with ActivationTracer(
            model, layers=[layer], sites=[site], read_pos=read_pos
        ) as tracer:
            next_logits = model(**encoded).logits[0, -1, :]
        label_logits = (
            next_logits[label_token_ids].detach().float().cpu().numpy()
        )
        vector = tracer.cache[(site, layer)][0].detach().float().cpu()
    return label_logits, vector


def forward_with_patch(
    tokenizer,
    model,
    prompt: str,
    site: str,
    layer: int,
    read_pos: int,
    patch_vector: torch.Tensor,
    label_token_ids: List[int],
) -> np.ndarray:
    encoded = encode_prompt(tokenizer, model, prompt)
    with torch.inference_mode():
        with patch_activation(
            model,
            site=site,
            layer_idx=layer,
            patch_vec=patch_vector,
            read_pos=read_pos,
        ):
            next_logits = model(**encoded).logits[0, -1, :]
    return next_logits[label_token_ids].detach().float().cpu().numpy()


def lens_scores(
    model,
    vector: torch.Tensor,
    label_token_ids: List[int],
) -> np.ndarray:
    hidden = vector.reshape(1, -1).to(dtype=next(model.parameters()).dtype)
    return (
        project_label_logits(
            model,
            hidden,
            label_token_ids=label_token_ids,
            apply_final_norm=True,
        )[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def equal_norm_orthogonal(
    delta: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    random_direction = torch.randn(
        delta.shape, generator=generator, dtype=torch.float32
    )
    delta = delta.float()
    delta_norm = torch.linalg.vector_norm(delta)
    if float(delta_norm) <= 1e-12:
        return torch.zeros_like(delta)
    random_direction = random_direction - (
        torch.dot(random_direction, delta)
        / torch.clamp(torch.dot(delta, delta), min=1e-12)
    ) * delta
    random_norm = torch.linalg.vector_norm(random_direction)
    if float(random_norm) <= 1e-12:
        raise RuntimeError("Failed to construct orthogonal random direction")
    return random_direction * (delta_norm / random_norm)


def rescale_like(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    direction_norm = torch.linalg.vector_norm(direction.float())
    reference_norm = torch.linalg.vector_norm(reference.float())
    if float(direction_norm) <= 1e-12 or float(reference_norm) <= 1e-12:
        return torch.zeros_like(reference.float())
    return direction.float() * (reference_norm / direction_norm)


def mode_row(
    base: Dict[str, Any],
    mode: str,
    output_scores: np.ndarray,
    lens_score: np.ndarray,
    labels: List[str],
    true_index: int,
    wrong_index: int,
) -> Dict[str, Any]:
    output_pred = int(np.argmax(output_scores))
    lens_pred = int(np.argmax(lens_score))
    return {
        **base,
        "mode": mode,
        "output_pred_label": labels[output_pred],
        "output_correct": bool(output_pred == true_index),
        "output_designated_wrong": bool(output_pred == wrong_index),
        "output_true_vs_wrong_margin": float(
            output_scores[true_index] - output_scores[wrong_index]
        ),
        "lens_pred_label": labels[lens_pred],
        "lens_correct": bool(lens_pred == true_index),
        "lens_designated_wrong": bool(lens_pred == wrong_index),
        "lens_true_vs_wrong_margin": float(
            lens_score[true_index] - lens_score[wrong_index]
        ),
        **{
            f"output_logit_{label}": float(output_scores[index])
            for index, label in enumerate(labels)
        },
        **{
            f"lens_logit_{label}": float(lens_score[index])
            for index, label in enumerate(labels)
        },
    }


def summarize_rows(
    rows: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = rows[rows["mode"].eq("target_baseline")][
        [
            "base_id",
            "output_true_vs_wrong_margin",
            "lens_true_vs_wrong_margin",
            "output_correct",
            "output_designated_wrong",
        ]
    ].rename(
        columns={
            "output_true_vs_wrong_margin": "baseline_output_margin",
            "lens_true_vs_wrong_margin": "baseline_lens_margin",
            "output_correct": "baseline_output_correct",
            "output_designated_wrong": "baseline_output_wrong",
        }
    )
    joined = rows.merge(baseline, on="base_id", validate="many_to_one")
    joined["output_margin_delta"] = (
        joined["output_true_vs_wrong_margin"] - joined["baseline_output_margin"]
    )
    joined["lens_margin_delta"] = (
        joined["lens_true_vs_wrong_margin"] - joined["baseline_lens_margin"]
    )
    summary_rows = []
    fidelity_rows = []
    for mode, group in joined.groupby("mode", sort=True):
        output_delta_ci = cluster_bootstrap_mean(
            group["output_margin_delta"].to_numpy(),
            group["base_id"].astype(str),
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
        lens_delta_ci = cluster_bootstrap_mean(
            group["lens_margin_delta"].to_numpy(),
            group["base_id"].astype(str),
            n_bootstrap=bootstrap_samples,
            seed=seed + 1,
        )
        baseline_wrong = group["baseline_output_wrong"].astype(bool)
        recovered = (
            float(group.loc[baseline_wrong, "output_correct"].mean())
            if baseline_wrong.any()
            else np.nan
        )
        summary_rows.append(
            {
                "mode": mode,
                "n_base": int(group["base_id"].nunique()),
                "output_correct_rate": float(group["output_correct"].mean()),
                "output_designated_wrong_rate": float(
                    group["output_designated_wrong"].mean()
                ),
                "output_margin_mean": float(
                    group["output_true_vs_wrong_margin"].mean()
                ),
                "output_margin_delta_mean": output_delta_ci["estimate"],
                "output_margin_delta_ci_low": output_delta_ci["ci_low"],
                "output_margin_delta_ci_high": output_delta_ci["ci_high"],
                "positive_output_margin_delta_rate": float(
                    (group["output_margin_delta"] > 0).mean()
                ),
                "lens_correct_rate": float(group["lens_correct"].mean()),
                "lens_margin_mean": float(
                    group["lens_true_vs_wrong_margin"].mean()
                ),
                "lens_margin_delta_mean": lens_delta_ci["estimate"],
                "lens_margin_delta_ci_low": lens_delta_ci["ci_low"],
                "lens_margin_delta_ci_high": lens_delta_ci["ci_high"],
                "baseline_wrong_n": int(baseline_wrong.sum()),
                "recovered_among_baseline_wrong_rate": recovered,
            }
        )
        actual = group["output_margin_delta"].to_numpy(dtype=float)
        predicted = group["lens_margin_delta"].to_numpy(dtype=float)
        nonzero = (np.abs(actual) > 1e-9) | (np.abs(predicted) > 1e-9)
        fidelity_rows.append(
            {
                "mode": mode,
                "n_base": int(len(group)),
                "sign_agreement": (
                    float(
                        (
                            np.sign(actual[nonzero])
                            == np.sign(predicted[nonzero])
                        ).mean()
                    )
                    if nonzero.any()
                    else np.nan
                ),
                "spearman_lens_vs_output_delta": safe_spearman(
                    predicted, actual
                ),
                "output_on_lens_through_origin_slope": through_origin_slope(
                    predicted, actual
                ),
            }
        )

    wide = joined.pivot(
        index="base_id", columns="mode", values="output_margin_delta"
    )
    comparison_rows = []
    for control in ["random_orthogonal", "shuffled_cross_item"]:
        if "clean_state_restoration" not in wide or control not in wide:
            continue
        difference = (
            wide["clean_state_restoration"] - wide[control]
        ).dropna()
        ci = cluster_bootstrap_mean(
            difference.to_numpy(),
            difference.index.astype(str),
            n_bootstrap=bootstrap_samples,
            seed=seed + 100,
        )
        comparison_rows.append(
            {
                "comparison": f"clean_state_restoration_minus_{control}",
                "n_paired_base": int(len(difference)),
                "paired_output_margin_delta_difference": ci["estimate"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fidelity_rows),
        pd.DataFrame(comparison_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    e_cfg = cfg["experiment_e"]
    seed = int(args.seed if args.seed is not None else cfg["experiment"]["seed"])
    site = str(e_cfg.get("feature_site", "resid_pre"))
    clean_trajectory = str(
        e_cfg.get("clean_trajectory_type", "wrong_pressure")
    )
    clean_turn = int(e_cfg.get("clean_turn", 0))
    pressure_trajectory = str(
        e_cfg.get("pressure_trajectory_type", "wrong_pressure")
    )
    pressure_turn = int(e_cfg.get("pressure_turn", 3))
    read_pos = int(cfg.get("activations", {}).get("read_pos", -1))
    output_bias = float(e_cfg.get("output_logit_bias", 8.0))
    bootstrap_samples = int(
        args.bootstrap_samples or e_cfg.get("bootstrap_samples", 2000)
    )

    selection_path = Path(args.selection)
    selection = json_load(selection_path)
    if args.layer is None:
        if selection.get("selection_data") != "dev_only":
            raise ValueError("E_01 selection is not marked dev_only")
        layer = int(selection["causal_layer"])
        layer_source = "E_01_dev_only_pressure_selection"
    else:
        layer = int(args.layer)
        layer_source = "explicit_cli_override"

    run_dir = Path(args.run_dir)
    dataset_path = Path(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(run_dir / "metadata.csv")
    records = record_index(dataset_path)
    predictions = prediction_index(meta)
    meta_rows = metadata_row_index(meta)
    labels = [str(value) for value in cfg["experiment"].get("labels", "ABCD")]
    mapping = label_to_index(labels)

    eligible = sorted(
        {
            str(base_id)
            for base_id in meta.loc[
                meta["split"].astype(str).eq("test")
                & meta["trajectory_type"].astype(str).eq(pressure_trajectory),
                "base_id",
            ]
            if (str(base_id), pressure_trajectory) in records
            and (str(base_id), clean_trajectory) in records
        }
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    if args.max_items is not None:
        eligible = eligible[: int(args.max_items)]
    if len(eligible) < 2:
        raise ValueError("Need at least two outcome-independent test base items")

    tokenizer, model = load_tokenizer_and_model(cfg)
    label_info = get_label_token_info(tokenizer, labels)
    label_token_ids = [int(label_info.token_ids[label]) for label in labels]
    captures: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for item_index, base_id in enumerate(eligible):
        try:
            clean_record = records[(base_id, clean_trajectory)]
            pressure_record = records[(base_id, pressure_trajectory)]
            true_label = str(pressure_record["true_label"])
            wrong_label = str(pressure_record["wrong_label"])
            if str(clean_record["true_label"]) != true_label:
                raise ValueError("Clean and pressure records disagree on true label")
            clean_prompt = build_exact_prompt(
                tokenizer, cfg, clean_record, clean_turn, predictions
            )
            pressure_prompt = build_exact_prompt(
                tokenizer, cfg, pressure_record, pressure_turn, predictions
            )
            clean_key = (base_id, clean_trajectory, clean_turn)
            pressure_key = (base_id, pressure_trajectory, pressure_turn)
            clean_expected_hash = str(meta_rows[clean_key]["prompt_hash"])
            pressure_expected_hash = str(meta_rows[pressure_key]["prompt_hash"])
            if stable_hash(clean_prompt) != clean_expected_hash:
                raise ValueError("Clean prompt hash does not match extraction run")
            if stable_hash(pressure_prompt) != pressure_expected_hash:
                raise ValueError("Pressure prompt hash does not match extraction run")

            clean_output, clean_vector = forward_capture(
                tokenizer,
                model,
                clean_prompt,
                site,
                layer,
                read_pos,
                label_token_ids,
            )
            pressure_output, pressure_vector = forward_capture(
                tokenizer,
                model,
                pressure_prompt,
                site,
                layer,
                read_pos,
                label_token_ids,
            )
            saved_pressure_logits = np.array(
                [
                    float(meta_rows[pressure_key][f"logit_{label}"])
                    for label in labels
                ]
            )
            true_index = mapping[true_label]
            wrong_index = mapping[wrong_label]
            recomputed_margin = float(
                pressure_output[true_index] - pressure_output[wrong_index]
            )
            saved_margin = float(
                saved_pressure_logits[true_index]
                - saved_pressure_logits[wrong_index]
            )
            captures.append(
                {
                    "base_id": base_id,
                    "true_label": true_label,
                    "wrong_label": wrong_label,
                    "clean_prompt": clean_prompt,
                    "pressure_prompt": pressure_prompt,
                    "clean_prompt_hash": clean_expected_hash,
                    "pressure_prompt_hash": pressure_expected_hash,
                    "clean_output": clean_output,
                    "pressure_output": pressure_output,
                    "clean_vector": clean_vector,
                    "pressure_vector": pressure_vector,
                    "delta": clean_vector - pressure_vector,
                    "baseline_reproduction_max_abs_logit_error": float(
                        np.max(np.abs(pressure_output - saved_pressure_logits))
                    ),
                    "baseline_reproduction_margin_abs_error": float(
                        abs(recomputed_margin - saved_margin)
                    ),
                    "baseline_reproduction_prediction_match": bool(
                        int(np.argmax(pressure_output))
                        == int(np.argmax(saved_pressure_logits))
                    ),
                    "baseline_reproduction_margin_sign_match": bool(
                        np.sign(recomputed_margin) == np.sign(saved_margin)
                    ),
                }
            )
        except Exception as error:
            errors.append({"base_id": base_id, "error": repr(error)})
        if (item_index + 1) % 25 == 0:
            print(
                f"[实验E_03] captured {item_index + 1}/{len(eligible)}",
                flush=True,
            )

    if len(captures) < 2:
        raise RuntimeError(f"Fewer than two successful captures; errors={errors[:3]}")

    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed + 101)
    result_rows: List[Dict[str, Any]] = []
    for index, capture in enumerate(captures):
        try:
            true_index = mapping[capture["true_label"]]
            wrong_index = mapping[capture["wrong_label"]]
            baseline_vector = capture["pressure_vector"]
            delta = capture["delta"]
            restored_vector = capture["clean_vector"]
            random_vector = baseline_vector + equal_norm_orthogonal(
                delta, torch_generator
            )
            other_delta = captures[(index + 1) % len(captures)]["delta"]
            shuffled_vector = baseline_vector + rescale_like(other_delta, delta)

            restored_output = forward_with_patch(
                tokenizer,
                model,
                capture["pressure_prompt"],
                site,
                layer,
                read_pos,
                restored_vector,
                label_token_ids,
            )
            random_output = forward_with_patch(
                tokenizer,
                model,
                capture["pressure_prompt"],
                site,
                layer,
                read_pos,
                random_vector,
                label_token_ids,
            )
            shuffled_output = forward_with_patch(
                tokenizer,
                model,
                capture["pressure_prompt"],
                site,
                layer,
                read_pos,
                shuffled_vector,
                label_token_ids,
            )
            biased_output = capture["pressure_output"].copy()
            biased_output[true_index] += output_bias

            base = {
                "base_id": capture["base_id"],
                "true_label": capture["true_label"],
                "wrong_label": capture["wrong_label"],
                "site": site,
                "layer": layer,
                "layer_source": layer_source,
                "clean_turn": clean_turn,
                "pressure_turn": pressure_turn,
                "clean_prompt_hash": capture["clean_prompt_hash"],
                "pressure_prompt_hash": capture["pressure_prompt_hash"],
                "restoration_delta_norm": float(
                    torch.linalg.vector_norm(delta.float())
                ),
                "baseline_reproduction_max_abs_logit_error": capture[
                    "baseline_reproduction_max_abs_logit_error"
                ],
                "baseline_reproduction_margin_abs_error": capture[
                    "baseline_reproduction_margin_abs_error"
                ],
                "baseline_reproduction_prediction_match": capture[
                    "baseline_reproduction_prediction_match"
                ],
                "baseline_reproduction_margin_sign_match": capture[
                    "baseline_reproduction_margin_sign_match"
                ],
            }
            result_rows.extend(
                [
                    mode_row(
                        base,
                        "source_clean",
                        capture["clean_output"],
                        lens_scores(model, capture["clean_vector"], label_token_ids),
                        labels,
                        true_index,
                        wrong_index,
                    ),
                    mode_row(
                        base,
                        "target_baseline",
                        capture["pressure_output"],
                        lens_scores(model, baseline_vector, label_token_ids),
                        labels,
                        true_index,
                        wrong_index,
                    ),
                    mode_row(
                        base,
                        "clean_state_restoration",
                        restored_output,
                        lens_scores(model, restored_vector, label_token_ids),
                        labels,
                        true_index,
                        wrong_index,
                    ),
                    mode_row(
                        base,
                        "random_orthogonal",
                        random_output,
                        lens_scores(model, random_vector, label_token_ids),
                        labels,
                        true_index,
                        wrong_index,
                    ),
                    mode_row(
                        base,
                        "shuffled_cross_item",
                        shuffled_output,
                        lens_scores(model, shuffled_vector, label_token_ids),
                        labels,
                        true_index,
                        wrong_index,
                    ),
                    mode_row(
                        base,
                        "output_logit_bias",
                        biased_output,
                        lens_scores(model, baseline_vector, label_token_ids),
                        labels,
                        true_index,
                        wrong_index,
                    ),
                ]
            )
        except Exception as error:
            errors.append({"base_id": capture["base_id"], "error": repr(error)})
        if (index + 1) % 25 == 0:
            print(
                f"[实验E_03] patched {index + 1}/{len(captures)}",
                flush=True,
            )

    rows = pd.DataFrame(result_rows)
    complete_counts = rows.groupby("base_id")["mode"].nunique()
    complete_ids = complete_counts[complete_counts.eq(6)].index
    rows = rows[rows["base_id"].isin(complete_ids)].copy()
    if rows.empty:
        raise RuntimeError("No base item completed all six modes")
    summary, fidelity, comparisons = summarize_rows(
        rows, bootstrap_samples=bootstrap_samples, seed=seed
    )

    paths = {
        "rows": outdir / "实验E_03_patching_rows.csv.gz",
        "summary": outdir / "实验E_03_patching_summary.csv",
        "causal_fidelity": outdir / "实验E_03_causal_fidelity.csv",
        "control_comparisons": outdir / "实验E_03_control_comparisons.csv",
        "errors": outdir / "实验E_03_errors.csv",
    }
    rows.to_csv(paths["rows"], index=False, compression="gzip")
    summary.to_csv(paths["summary"], index=False)
    fidelity.to_csv(paths["causal_fidelity"], index=False)
    comparisons.to_csv(paths["control_comparisons"], index=False)
    pd.DataFrame(errors, columns=["base_id", "error"]).to_csv(
        paths["errors"], index=False
    )

    reproduction_error = float(
        rows["baseline_reproduction_max_abs_logit_error"].max()
    )
    reproduction_by_base = rows.drop_duplicates("base_id")
    reproduction_logit_errors = reproduction_by_base[
        "baseline_reproduction_max_abs_logit_error"
    ].to_numpy(dtype=float)
    reproduction_margin_errors = reproduction_by_base[
        "baseline_reproduction_margin_abs_error"
    ].to_numpy(dtype=float)
    json_dump(
        {
            "experiment": "E_03_clean_state_causal_restoration",
            "command": command_record(),
            "config": file_fingerprint(Path(args.config)),
            "dataset": file_fingerprint(dataset_path),
            "metadata": file_fingerprint(run_dir / "metadata.csv"),
            "selection": file_fingerprint(selection_path),
            "selection_data": selection.get("selection_data"),
            "layer": layer,
            "layer_source": layer_source,
            "site": site,
            "clean_source": f"{clean_trajectory} T{clean_turn}",
            "pressure_target": f"{pressure_trajectory} T{pressure_turn}",
            "test_item_selection": (
                "all eligible test base IDs, shuffled by seed before optional "
                "max_items truncation; no outcome-based filtering"
            ),
            "n_eligible": len(eligible),
            "n_captured": len(captures),
            "n_complete": int(len(complete_ids)),
            "n_errors": len(errors),
            "output_logit_bias": output_bias,
            "baseline_reproduction_max_abs_logit_error": reproduction_error,
            "baseline_reproduction_median_max_abs_logit_error": float(
                np.median(reproduction_logit_errors)
            ),
            "baseline_reproduction_p95_max_abs_logit_error": float(
                np.quantile(reproduction_logit_errors, 0.95)
            ),
            "baseline_reproduction_margin_mae": float(
                np.mean(reproduction_margin_errors)
            ),
            "baseline_reproduction_margin_p95_abs_error": float(
                np.quantile(reproduction_margin_errors, 0.95)
            ),
            "baseline_reproduction_prediction_agreement": float(
                reproduction_by_base[
                    "baseline_reproduction_prediction_match"
                ].mean()
            ),
            "baseline_reproduction_margin_sign_agreement": float(
                reproduction_by_base[
                    "baseline_reproduction_margin_sign_match"
                ].mean()
            ),
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        },
        outdir / "实验E_03_manifest.json",
    )
    print(summary.to_string(index=False))
    print(f"[OK] {paths['summary']}")


if __name__ == "__main__":
    main()
