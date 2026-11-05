#!/usr/bin/env python3
"""L22 restoration -> L24 readout under natural GSM8K dialogue histories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_pressure.config import load_config  # noqa: E402
from latent_pressure.hf_model import (  # noqa: E402
    ActivationTracer,
    first_param_device,
    load_tokenizer_and_model,
    patch_activation,
)
from 实验E_00_common import (  # noqa: E402
    command_record,
    file_fingerprint,
    json_dump,
    json_load,
)
from 实验E_10_option_probability_readout import (  # noqa: E402
    bootstrap_mean,
    option_softmax,
    predict_probability,
    safe_correlation,
)


LABELS = ["X", "Y"]
MAPPING = {"X": 0, "Y": 1}
MODES = [
    "source_clean",
    "target_baseline",
    "clean_state_restoration",
    "random_orthogonal",
    "shuffled_cross_item",
    "output_logit_bias",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def equal_norm_orthogonal(
    delta: torch.Tensor, *, base_id: str, seed: int
) -> torch.Tensor:
    digest = hashlib.sha256(str(base_id).encode("utf-8")).hexdigest()
    item_seed = (int(digest[:12], 16) + int(seed)) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(item_seed)
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
        raise RuntimeError("Failed to construct orthogonal direction")
    return random_direction * (delta_norm / random_norm)


def rescale_like(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    direction_norm = torch.linalg.vector_norm(direction.float())
    reference_norm = torch.linalg.vector_norm(reference.float())
    if float(direction_norm) <= 1e-12 or float(reference_norm) <= 1e-12:
        return torch.zeros_like(reference.float())
    return direction.float() * (reference_norm / direction_norm)


def score_candidate(
    *,
    tokenizer,
    model,
    candidate: str,
    first_logits: torch.Tensor,
    past_key_values,
    prompt_length: int,
    device: torch.device,
) -> float:
    token_ids = tokenizer.encode(str(candidate), add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Empty tokenization for candidate {candidate!r}")
    first_log_probability = torch.log_softmax(
        first_logits.detach().float(), dim=-1
    )
    total = float(first_log_probability[int(token_ids[0])].cpu())
    if len(token_ids) > 1:
        # Qwen3 mutates DynamicCache in-place while continuing a candidate.
        # Clone the prompt cache so X/Y and intervention modes are independent.
        legacy_cache = (
            past_key_values
            if isinstance(past_key_values, tuple)
            else past_key_values.to_legacy_cache()
        )
        cache = DynamicCache.from_legacy_cache(
            tuple(
                tuple(tensor.detach().clone() for tensor in layer_cache)
                for layer_cache in legacy_cache
            )
        )
        continuation = torch.tensor(
            [token_ids[:-1]], dtype=torch.long, device=device
        )
        attention_mask = torch.ones(
            (1, int(prompt_length) + len(token_ids) - 1),
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            continuation_logits = model(
                input_ids=continuation,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=False,
            ).logits[0]
        continuation_log_probability = torch.log_softmax(
            continuation_logits.detach().float(), dim=-1
        )
        for position, token_id in enumerate(token_ids[1:]):
            total += float(
                continuation_log_probability[position, int(token_id)].cpu()
            )
    return total


def forward_capture(
    *,
    tokenizer,
    model,
    prompt: str,
    candidate_x: str,
    candidate_y: str,
    readout_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = first_param_device(model)
    kwargs = {
        "input_ids": encoded["input_ids"].to(device),
        "use_cache": True,
    }
    if encoded.get("attention_mask") is not None:
        kwargs["attention_mask"] = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        with ActivationTracer(
            model,
            layers=[int(readout_layer)],
            sites=["resid_pre"],
            read_pos=-1,
        ) as tracer:
            model_output = model(**kwargs)
    first_logits = model_output.logits[0, -1, :]
    return (
        np.asarray(
            [
                score_candidate(
                    tokenizer=tokenizer,
                    model=model,
                    candidate=candidate_x,
                    first_logits=first_logits,
                    past_key_values=model_output.past_key_values,
                    prompt_length=int(encoded["input_ids"].shape[1]),
                    device=device,
                ),
                score_candidate(
                    tokenizer=tokenizer,
                    model=model,
                    candidate=candidate_y,
                    first_logits=first_logits,
                    past_key_values=model_output.past_key_values,
                    prompt_length=int(encoded["input_ids"].shape[1]),
                    device=device,
                ),
            ],
            dtype=np.float64,
        ),
        tracer.cache[("resid_pre", int(readout_layer))][0].numpy(),
    )


def forward_patch(
    *,
    tokenizer,
    model,
    prompt: str,
    candidate_x: str,
    candidate_y: str,
    patch_vector: torch.Tensor,
    repair_layer: int,
    readout_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = first_param_device(model)
    kwargs = {
        "input_ids": encoded["input_ids"].to(device),
        "use_cache": True,
    }
    if encoded.get("attention_mask") is not None:
        kwargs["attention_mask"] = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        with patch_activation(
            model,
            site="resid_pre",
            layer_idx=int(repair_layer),
            patch_vec=patch_vector,
            read_pos=-1,
        ):
            with ActivationTracer(
                model,
                layers=[int(readout_layer)],
                sites=["resid_pre"],
                read_pos=-1,
            ) as tracer:
                model_output = model(**kwargs)
    first_logits = model_output.logits[0, -1, :]
    return (
        np.asarray(
            [
                score_candidate(
                    tokenizer=tokenizer,
                    model=model,
                    candidate=candidate_x,
                    first_logits=first_logits,
                    past_key_values=model_output.past_key_values,
                    prompt_length=int(encoded["input_ids"].shape[1]),
                    device=device,
                ),
                score_candidate(
                    tokenizer=tokenizer,
                    model=model,
                    candidate=candidate_y,
                    first_logits=first_logits,
                    past_key_values=model_output.past_key_values,
                    prompt_length=int(encoded["input_ids"].shape[1]),
                    device=device,
                ),
            ],
            dtype=np.float64,
        ),
        tracer.cache[("resid_pre", int(readout_layer))][0].numpy(),
    )


def readout_probability(bundle: Dict[str, Any], vector: np.ndarray) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    return predict_probability(
        bundle["model"], bundle["scaler"].transform(x)
    )[0]


def mode_row(
    *,
    model_name: str,
    base_id: str,
    gold_label: str,
    wrong_label: str,
    mode: str,
    output_logits: np.ndarray,
    downstream: np.ndarray,
    readout_bundle: Dict[str, Any],
    delta_norm: float,
    repair_layer: int,
    readout_layer: int,
) -> Dict[str, Any]:
    gold_index = MAPPING[gold_label]
    wrong_index = MAPPING[wrong_label]
    output = option_softmax(
        np.asarray(output_logits, dtype=np.float64).reshape(1, -1)
    )[0]
    readout = readout_probability(readout_bundle, downstream)
    return {
        "model_name": model_name,
        "base_id": base_id,
        "gold_option_label": gold_label,
        "wrong_option_label": wrong_label,
        "mode": mode,
        "repair_layer": int(repair_layer),
        "readout_layer": int(readout_layer),
        "restoration_delta_norm": float(delta_norm),
        "output_logit_X": float(output_logits[0]),
        "output_logit_Y": float(output_logits[1]),
        "output_probability_X": float(output[0]),
        "output_probability_Y": float(output[1]),
        "output_pred_label": LABELS[int(output.argmax())],
        "output_gold_preferred": bool(int(output.argmax()) == gold_index),
        "output_gold_probability": float(output[gold_index]),
        "output_wrong_probability": float(output[wrong_index]),
        "output_gold_vs_wrong_gap": float(
            output[gold_index] - output[wrong_index]
        ),
        "readout_probability_X": float(readout[0]),
        "readout_probability_Y": float(readout[1]),
        "readout_pred_label": LABELS[int(readout.argmax())],
        "readout_gold_preferred": bool(int(readout.argmax()) == gold_index),
        "readout_gold_probability": float(readout[gold_index]),
        "readout_wrong_probability": float(readout[wrong_index]),
        "readout_gold_vs_wrong_gap": float(
            readout[gold_index] - readout[wrong_index]
        ),
        "downstream_vector_norm": float(
            np.linalg.norm(np.asarray(downstream, dtype=np.float64))
        ),
    }


def bootstrap_ci(values: Iterable[float], *, n: int, seed: int) -> Dict[str, float]:
    return bootstrap_mean(values, n_bootstrap=n, seed=seed)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - radius, centre + radius


def analyze(
    rows: pd.DataFrame,
    *,
    readout_metrics: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    paired = rows.pivot(index="base_id", columns="mode")
    baseline = "target_baseline"
    restoration = "clean_state_restoration"
    effects = []
    for mode in MODES:
        frame = rows[rows["mode"].eq(mode)]
        output_delta = (
            paired["output_gold_vs_wrong_gap"][mode]
            - paired["output_gold_vs_wrong_gap"][baseline]
        ).to_numpy()
        readout_delta = (
            paired["readout_gold_vs_wrong_gap"][mode]
            - paired["readout_gold_vs_wrong_gap"][baseline]
        ).to_numpy()
        out_ci = bootstrap_ci(
            output_delta, n=n_bootstrap, seed=seed + MODES.index(mode) * 2
        )
        read_ci = bootstrap_ci(
            readout_delta,
            n=n_bootstrap,
            seed=seed + MODES.index(mode) * 2 + 1,
        )
        effects.append(
            {
                "mode": mode,
                "n_base": int(len(frame)),
                "output_gold_preferred_rate": float(
                    frame["output_gold_preferred"].mean()
                ),
                "readout_gold_preferred_rate": float(
                    frame["readout_gold_preferred"].mean()
                ),
                "output_gap_delta": out_ci["estimate"],
                "output_gap_delta_ci_low": out_ci["ci_low"],
                "output_gap_delta_ci_high": out_ci["ci_high"],
                "readout_gap_delta": read_ci["estimate"],
                "readout_gap_delta_ci_low": read_ci["ci_low"],
                "readout_gap_delta_ci_high": read_ci["ci_high"],
            }
        )

    restoration_output = (
        paired["output_gold_vs_wrong_gap"][restoration]
        - paired["output_gold_vs_wrong_gap"][baseline]
    ).to_numpy()
    restoration_readout = (
        paired["readout_gold_vs_wrong_gap"][restoration]
        - paired["readout_gold_vs_wrong_gap"][baseline]
    ).to_numpy()
    comparisons = []
    matched_minus: Dict[str, Any] = {}
    for control in ["random_orthogonal", "shuffled_cross_item"]:
        control_output = (
            paired["output_gold_vs_wrong_gap"][control]
            - paired["output_gold_vs_wrong_gap"][baseline]
        ).to_numpy()
        control_readout = (
            paired["readout_gold_vs_wrong_gap"][control]
            - paired["readout_gold_vs_wrong_gap"][baseline]
        ).to_numpy()
        out_ci = bootstrap_ci(
            restoration_output - control_output,
            n=n_bootstrap,
            seed=seed + 100 + len(comparisons) * 2,
        )
        read_ci = bootstrap_ci(
            restoration_readout - control_readout,
            n=n_bootstrap,
            seed=seed + 101 + len(comparisons) * 2,
        )
        row = {
            "control": control,
            "n_base": int(len(paired)),
            "output_difference": out_ci["estimate"],
            "output_ci_low": out_ci["ci_low"],
            "output_ci_high": out_ci["ci_high"],
            "readout_difference": read_ci["estimate"],
            "readout_ci_low": read_ci["ci_low"],
            "readout_ci_high": read_ci["ci_high"],
        }
        comparisons.append(row)
        matched_minus[control] = row

    baseline_output_wrong = ~paired["output_gold_preferred"][baseline].astype(bool)
    baseline_readout_wrong = ~paired["readout_gold_preferred"][baseline].astype(bool)
    restored_output_recovery = paired["output_gold_preferred"][restoration].astype(bool)
    restored_readout_recovery = paired["readout_gold_preferred"][restoration].astype(bool)
    biased_surface_repairs = (
        baseline_output_wrong
        & paired["output_gold_preferred"]["output_logit_bias"].astype(bool)
    )
    biased_readout_wrong = ~paired["readout_gold_preferred"][
        "output_logit_bias"
    ].astype(bool)
    n_bias = int(biased_surface_repairs.sum())
    n_bias_readout_wrong = int(
        (biased_surface_repairs & biased_readout_wrong).sum()
    )
    bias_rate = n_bias_readout_wrong / n_bias if n_bias else math.nan
    bias_ci = wilson_interval(n_bias_readout_wrong, n_bias)
    test = readout_metrics[readout_metrics["split"].astype(str).eq("test")].iloc[0]
    effect_frame = pd.DataFrame(effects).set_index("mode")
    decision: Dict[str, Any] = {
        "n_complete_paired_base": int(len(paired)),
        "clean_test_preference_agreement": float(test["preference_agreement"]),
        "clean_test_balanced_agreement": float(
            test["preference_balanced_agreement"]
        ),
        "clean_test_option_probability_r2": float(test["option_probability_r2"]),
        "clean_test_option_probability_kl": float(test["option_probability_kl"]),
        "restoration_output_gap_delta": float(
            effect_frame.loc[restoration, "output_gap_delta"]
        ),
        "restoration_output_gap_delta_ci_low": float(
            effect_frame.loc[restoration, "output_gap_delta_ci_low"]
        ),
        "restoration_readout_gap_delta": float(
            effect_frame.loc[restoration, "readout_gap_delta"]
        ),
        "restoration_readout_gap_delta_ci_low": float(
            effect_frame.loc[restoration, "readout_gap_delta_ci_low"]
        ),
        "restoration_delta_spearman": safe_correlation(
            restoration_output, restoration_readout, rank=True
        ),
        "restoration_delta_pearson": safe_correlation(
            restoration_output, restoration_readout
        ),
        "restoration_delta_sign_agreement": float(
            (
                np.signbit(restoration_output)
                == np.signbit(restoration_readout)
            ).mean()
        ),
        "baseline_output_wrong_n": int(baseline_output_wrong.sum()),
        "baseline_readout_wrong_n": int(baseline_readout_wrong.sum()),
        "restored_output_recovery_among_baseline_wrong": float(
            restored_output_recovery[baseline_output_wrong].mean()
        )
        if baseline_output_wrong.any()
        else math.nan,
        "restored_readout_recovery_among_baseline_wrong": float(
            restored_readout_recovery[baseline_readout_wrong].mean()
        )
        if baseline_readout_wrong.any()
        else math.nan,
        "n_output_bias_surface_repairs": n_bias,
        "output_bias_readout_wrong_rate": bias_rate,
        "output_bias_readout_wrong_ci_low": bias_ci[0],
        "output_bias_readout_wrong_ci_high": bias_ci[1],
        "output_bias_readout_probability_delta_max_abs": float(
            max(
                np.max(
                    np.abs(
                        paired[f"readout_probability_{label}"][
                            "output_logit_bias"
                        ]
                        - paired[f"readout_probability_{label}"][baseline]
                    )
                )
                for label in LABELS
            )
        ),
        "matched_minus_controls": matched_minus,
    }
    gates = {
        "target_500_complete_paired_base": len(paired) == 500,
        "clean_test_preference_agreement_gte_0p90": decision[
            "clean_test_preference_agreement"
        ]
        >= 0.90,
        "clean_test_option_probability_r2_gte_0p95": decision[
            "clean_test_option_probability_r2"
        ]
        >= 0.95,
        "clean_test_option_probability_kl_lte_0p02": decision[
            "clean_test_option_probability_kl"
        ]
        <= 0.02,
        "restoration_output_ci_above_zero": decision[
            "restoration_output_gap_delta_ci_low"
        ]
        > 0,
        "restoration_readout_ci_above_zero": decision[
            "restoration_readout_gap_delta_ci_low"
        ]
        > 0,
        "restoration_delta_spearman_gte_0p50": decision[
            "restoration_delta_spearman"
        ]
        >= 0.50,
        "matched_output_exceeds_controls": all(
            row["output_ci_low"] > 0 for row in comparisons
        ),
        "matched_readout_exceeds_controls": all(
            row["readout_ci_low"] > 0 for row in comparisons
        ),
        "output_bias_readout_unchanged": decision[
            "output_bias_readout_probability_delta_max_abs"
        ]
        <= 1e-12,
    }
    decision["gates"] = gates
    decision["all_gates_pass"] = bool(all(gates.values()))
    return pd.DataFrame(effects), pd.DataFrame(comparisons), decision


def save_partial(
    rows: List[Dict[str, Any]], errors: List[Dict[str, Any]], outdir: Path
) -> None:
    pd.DataFrame(rows).to_csv(
        outdir / "实验E_23_partial_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(errors, columns=["base_id", "error"]).to_csv(
        outdir / "实验E_23_errors.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--readout_model", required=True)
    parser.add_argument("--readout_metrics", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--repair_layer", type=int, default=22)
    parser.add_argument("--readout_layer", type=int, default=24)
    parser.add_argument("--pressure_turn", type=int, default=3)
    parser.add_argument("--clean_turn", type=int, default=0)
    parser.add_argument("--output_logit_bias", type=float, default=8.0)
    parser.add_argument("--bootstrap_samples", type=int, default=3000)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--reproduction_checks", type=int, default=20)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.readout_layer <= args.repair_layer:
        raise ValueError("Readout layer must be strictly downstream")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    split = json_load(Path(args.split_manifest))["split"]
    eligible = list(map(str, split["causal_eval"]))
    if args.max_items is not None:
        eligible = eligible[: int(args.max_items)]
    run_dir = Path(args.run_dir)
    run_manifest = json_load(run_dir / "manifest.json")
    metadata = pd.read_csv(run_dir / "metadata.csv")
    contexts = {
        int(row["row_idx"]): row
        for row in read_jsonl(run_dir / "实验E_21_diagnostic_contexts.jsonl")
    }
    l22 = np.load(
        run_dir / "activations" / f"resid_pre_L{args.repair_layer}.npy",
        mmap_mode="r",
    )
    l24 = np.load(
        run_dir / "activations" / f"resid_pre_L{args.readout_layer}.npy",
        mmap_mode="r",
    )
    if not (len(metadata) == len(contexts) == len(l22) == len(l24)):
        raise ValueError("E21 row alignment mismatch")
    causal = metadata[metadata["experiment_split"].astype(str).eq("causal_eval")]
    row_index = {
        (str(row["base_id"]), int(row["turn_index"])): row.to_dict()
        for _, row in causal.iterrows()
    }
    captures = []
    for base_id in eligible:
        clean = row_index[(base_id, int(args.clean_turn))]
        pressure = row_index[(base_id, int(args.pressure_turn))]
        clean_idx = int(clean["row_idx"])
        pressure_idx = int(pressure["row_idx"])
        if contexts[clean_idx]["diagnostic_prompt_hash"] != clean[
            "diagnostic_prompt_hash"
        ]:
            raise ValueError("Clean context hash mismatch")
        if contexts[pressure_idx]["diagnostic_prompt_hash"] != pressure[
            "diagnostic_prompt_hash"
        ]:
            raise ValueError("Pressure context hash mismatch")
        clean_upstream = torch.as_tensor(
            np.asarray(l22[clean_idx], dtype=np.float32)
        )
        pressure_upstream = torch.as_tensor(
            np.asarray(l22[pressure_idx], dtype=np.float32)
        )
        captures.append(
            {
                "base_id": base_id,
                "clean": clean,
                "pressure": pressure,
                "clean_upstream": clean_upstream,
                "pressure_upstream": pressure_upstream,
                "clean_downstream": np.asarray(l24[clean_idx], dtype=np.float32),
                "pressure_downstream": np.asarray(
                    l24[pressure_idx], dtype=np.float32
                ),
                "clean_prompt": str(contexts[clean_idx]["diagnostic_prompt"]),
                "pressure_prompt": str(
                    contexts[pressure_idx]["diagnostic_prompt"]
                ),
                "delta": clean_upstream - pressure_upstream,
            }
        )

    cfg = load_config(args.config)
    tokenizer, model = load_tokenizer_and_model(cfg)
    readout_bundle = joblib.load(args.readout_model)
    if set(eligible) & set(readout_bundle["training_base_ids"]):
        raise AssertionError("Causal items leaked into readout training")

    reproduction_rows = []
    for capture in captures[: int(args.reproduction_checks)]:
        output, downstream = forward_capture(
            tokenizer=tokenizer,
            model=model,
            prompt=capture["pressure_prompt"],
            candidate_x=str(capture["pressure"]["candidate_X"]),
            candidate_y=str(capture["pressure"]["candidate_Y"]),
            readout_layer=int(args.readout_layer),
        )
        saved_output = np.asarray(
            [
                float(capture["pressure"]["logit_X"]),
                float(capture["pressure"]["logit_Y"]),
            ]
        )
        saved_downstream = capture["pressure_downstream"]
        reproduction_rows.append(
            {
                "base_id": capture["base_id"],
                "output_max_abs_error": float(
                    np.max(np.abs(output - saved_output))
                ),
                "downstream_max_abs_error": float(
                    np.max(np.abs(downstream - saved_downstream))
                ),
                "prediction_match": bool(output.argmax() == saved_output.argmax()),
            }
        )
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(outdir / "实验E_23_reproduction_checks.csv", index=False)
    if not reproduction["prediction_match"].all():
        raise RuntimeError("Reproduction prediction mismatch")

    partial_path = outdir / "实验E_23_partial_rows.csv.gz"
    error_path = outdir / "实验E_23_errors.csv"
    if args.resume and partial_path.exists():
        result_rows = pd.read_csv(partial_path).to_dict("records")
    else:
        result_rows: List[Dict[str, Any]] = []
    if args.resume and error_path.exists():
        errors = pd.read_csv(error_path).to_dict("records")
    else:
        errors: List[Dict[str, Any]] = []
    existing = pd.DataFrame(result_rows)
    complete_ids = (
        set(
            existing.groupby("base_id")["mode"]
            .nunique()
            .loc[lambda value: value.eq(len(MODES))]
            .index.astype(str)
        )
        if not existing.empty
        else set()
    )
    for index, capture in enumerate(captures):
        base_id = str(capture["base_id"])
        if base_id in complete_ids:
            continue
        try:
            clean_output = np.asarray(
                [
                    float(capture["clean"]["logit_X"]),
                    float(capture["clean"]["logit_Y"]),
                ]
            )
            pressure_output = np.asarray(
                [
                    float(capture["pressure"]["logit_X"]),
                    float(capture["pressure"]["logit_Y"]),
                ]
            )
            restored_output, restored_downstream = forward_patch(
                tokenizer=tokenizer,
                model=model,
                prompt=capture["pressure_prompt"],
                candidate_x=str(capture["pressure"]["candidate_X"]),
                candidate_y=str(capture["pressure"]["candidate_Y"]),
                patch_vector=capture["clean_upstream"],
                repair_layer=int(args.repair_layer),
                readout_layer=int(args.readout_layer),
            )
            random_vector = capture[
                "pressure_upstream"
            ] + equal_norm_orthogonal(
                capture["delta"], base_id=base_id, seed=int(args.seed)
            )
            random_output, random_downstream = forward_patch(
                tokenizer=tokenizer,
                model=model,
                prompt=capture["pressure_prompt"],
                candidate_x=str(capture["pressure"]["candidate_X"]),
                candidate_y=str(capture["pressure"]["candidate_Y"]),
                patch_vector=random_vector,
                repair_layer=int(args.repair_layer),
                readout_layer=int(args.readout_layer),
            )
            other_delta = captures[(index + 1) % len(captures)]["delta"]
            shuffled_vector = capture["pressure_upstream"] + rescale_like(
                other_delta, capture["delta"]
            )
            shuffled_output, shuffled_downstream = forward_patch(
                tokenizer=tokenizer,
                model=model,
                prompt=capture["pressure_prompt"],
                candidate_x=str(capture["pressure"]["candidate_X"]),
                candidate_y=str(capture["pressure"]["candidate_Y"]),
                patch_vector=shuffled_vector,
                repair_layer=int(args.repair_layer),
                readout_layer=int(args.readout_layer),
            )
            gold_label = str(capture["pressure"]["gold_option_label"])
            wrong_label = str(capture["pressure"]["wrong_option_label"])
            biased_output = pressure_output.copy()
            biased_output[MAPPING[gold_label]] += float(args.output_logit_bias)
            common = {
                "model_name": args.model_name,
                "base_id": base_id,
                "gold_label": gold_label,
                "wrong_label": wrong_label,
                "readout_bundle": readout_bundle,
                "repair_layer": int(args.repair_layer),
                "readout_layer": int(args.readout_layer),
                "delta_norm": float(
                    torch.linalg.vector_norm(capture["delta"].float())
                ),
            }
            result_rows.extend(
                [
                    mode_row(
                        mode="source_clean",
                        output_logits=clean_output,
                        downstream=capture["clean_downstream"],
                        **common,
                    ),
                    mode_row(
                        mode="target_baseline",
                        output_logits=pressure_output,
                        downstream=capture["pressure_downstream"],
                        **common,
                    ),
                    mode_row(
                        mode="clean_state_restoration",
                        output_logits=restored_output,
                        downstream=restored_downstream,
                        **common,
                    ),
                    mode_row(
                        mode="random_orthogonal",
                        output_logits=random_output,
                        downstream=random_downstream,
                        **common,
                    ),
                    mode_row(
                        mode="shuffled_cross_item",
                        output_logits=shuffled_output,
                        downstream=shuffled_downstream,
                        **common,
                    ),
                    mode_row(
                        mode="output_logit_bias",
                        output_logits=biased_output,
                        downstream=capture["pressure_downstream"],
                        **common,
                    ),
                ]
            )
            complete_ids.add(base_id)
        except Exception as error:
            errors.append({"base_id": base_id, "error": repr(error)})
        if len(complete_ids) % int(args.checkpoint_every) == 0:
            save_partial(result_rows, errors, outdir)
            print(
                f"[E23 {args.model_name}] {len(complete_ids)}/{len(captures)} "
                f"errors={len(errors)}",
                flush=True,
            )
    save_partial(result_rows, errors, outdir)
    rows = pd.DataFrame(result_rows)
    complete_counts = rows.groupby("base_id")["mode"].nunique()
    final_ids = set(
        complete_counts[complete_counts.eq(len(MODES))].index.astype(str)
    )
    rows = rows[rows["base_id"].astype(str).isin(final_ids)].copy()
    rows_path = outdir / "实验E_23_causal_rows.csv.gz"
    rows.to_csv(rows_path, index=False, compression="gzip")
    readout_metrics = pd.read_csv(args.readout_metrics)
    effects, comparisons, decision = analyze(
        rows,
        readout_metrics=readout_metrics,
        n_bootstrap=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    effects.to_csv(outdir / "实验E_23_mode_effects.csv", index=False)
    comparisons.to_csv(
        outdir / "实验E_23_control_comparisons.csv", index=False
    )
    decision_path = outdir / "实验E_23_claim_decision.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    json_dump(
        {
            "experiment": "E_23_open_dialogue_causal_restoration",
            "command": command_record(),
            "model_name": args.model_name,
            "n_requested": len(eligible),
            "n_complete": len(final_ids),
            "n_errors": len(errors),
            "run_manifest": file_fingerprint(run_dir / "manifest.json"),
            "split_manifest": file_fingerprint(Path(args.split_manifest)),
            "readout_model": file_fingerprint(Path(args.readout_model)),
            "readout_metrics": file_fingerprint(Path(args.readout_metrics)),
            "repair_layer": int(args.repair_layer),
            "readout_layer": int(args.readout_layer),
            "diagnostic_scope": (
                "candidate-conditioned measurement checkpoint appended after "
                "unrestricted free-form GSM8K dialogue"
            ),
            "decision": decision,
            "compute": {
                "gpu_count_used": 1,
                "single_model_process": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
        outdir / "实验E_23_manifest.json",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
