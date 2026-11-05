#!/usr/bin/env python3
"""Independent structural and numerical audit for Experiments E19--E23."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score


LABELS = ["X", "Y"]
MAPPING = {"X": 0, "Y": 1}
MODES = {
    "source_clean",
    "target_baseline",
    "clean_state_restoration",
    "random_orthogonal",
    "shuffled_cross_item",
    "output_logit_bias",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def close(actual: float, expected: float, name: str, atol: float = 1e-10) -> None:
    require(
        bool(np.isclose(float(actual), float(expected), atol=atol, rtol=1e-8)),
        f"{name}: observed={actual}, expected={expected}",
    )


def balanced(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(balanced_accuracy_score(actual, predicted))


def audit_model(
    *,
    model_name: str,
    run_dir: Path,
    readout_dir: Path,
    causal_dir: Path,
    split_sets: Dict[str, set[str]],
) -> Dict[str, Any]:
    checks: List[str] = []
    metadata = pd.read_csv(run_dir / "metadata.csv")
    require(len(metadata) == 3819, f"{model_name}: E21 must have 3819 rows")
    require(
        metadata["row_idx"].astype(int).tolist() == list(range(3819)),
        f"{model_name}: non-contiguous E21 row_idx",
    )
    observed_counts = (
        metadata["experiment_split"].astype(str).value_counts().to_dict()
    )
    require(
        observed_counts
        == {
            "causal_eval": 3000,
            "readout_train": 500,
            "readout_dev": 159,
            "readout_test": 160,
        },
        f"{model_name}: unexpected E21 split counts {observed_counts}",
    )
    causal = metadata[
        metadata["experiment_split"].astype(str).eq("causal_eval")
    ]
    causal_turns = causal.groupby("base_id")["turn_index"].agg(
        lambda values: sorted(map(int, values))
    )
    require(
        len(causal_turns) == 500
        and all(turns == list(range(6)) for turns in causal_turns),
        f"{model_name}: causal items must each have six ordered turns",
    )
    clean = metadata[
        ~metadata["experiment_split"].astype(str).eq("causal_eval")
    ]
    require(
        clean["base_id"].astype(str).nunique() == 819
        and clean["turn_index"].astype(int).eq(0).all(),
        f"{model_name}: clean pool must contain one T0 row per item",
    )
    all_expected = set().union(*split_sets.values())
    require(
        set(metadata["base_id"].astype(str)) == all_expected,
        f"{model_name}: metadata IDs differ from frozen E19 split",
    )
    for row in metadata.itertuples(index=False):
        candidate = {"X": str(row.candidate_X), "Y": str(row.candidate_Y)}
        require(
            candidate[str(row.gold_option_label)] == str(row.gold_answer),
            f"{model_name}: gold candidate mapping mismatch",
        )
        require(
            candidate[str(row.wrong_option_label)] == str(row.wrong_answer),
            f"{model_name}: wrong candidate mapping mismatch",
        )
        require(
            {str(row.gold_option_label), str(row.wrong_option_label)}
            == set(LABELS),
            f"{model_name}: invalid randomized labels",
        )
    probability = metadata[["probability_X", "probability_Y"]].to_numpy(
        dtype=np.float64
    )
    require(np.isfinite(probability).all(), f"{model_name}: non-finite targets")
    require(
        np.max(np.abs(probability.sum(axis=1) - 1.0)) <= 1e-10,
        f"{model_name}: targets do not sum to one",
    )
    predicted_label = np.asarray(LABELS, dtype=object)[probability.argmax(axis=1)]
    require(
        np.array_equal(
            predicted_label, metadata["diagnostic_pred_label"].astype(str)
        ),
        f"{model_name}: stored diagnostic labels mismatch probabilities",
    )
    activation_shapes = {}
    for layer in [22, 24]:
        activation = np.load(
            run_dir / "activations" / f"resid_pre_L{layer}.npy",
            mmap_mode="r",
        )
        require(
            activation.shape[0] == 3819 and activation.ndim == 2,
            f"{model_name}: invalid L{layer} activation shape {activation.shape}",
        )
        require(
            np.isfinite(np.asarray(activation[:8], dtype=np.float32)).all(),
            f"{model_name}: non-finite activation sample at L{layer}",
        )
        activation_shapes[f"L{layer}"] = list(activation.shape)
    checks.extend(
        [
            "E21 split/row alignment",
            "six-turn causal histories",
            "candidate randomization mapping",
            "two-answer probability normalization",
            "L22/L24 activation alignment",
        ]
    )

    predictions_path = readout_dir / "实验E_22_all_diagnostic_predictions.csv.gz"
    predictions = pd.read_csv(predictions_path)
    require(
        len(predictions) == len(metadata),
        f"{model_name}: E22 prediction row mismatch",
    )
    require(
        predictions["row_idx"].astype(int).tolist()
        == metadata["row_idx"].astype(int).tolist(),
        f"{model_name}: E22 predictions lost row alignment",
    )
    model_bundle = joblib.load(readout_dir / "实验E_22_readout.joblib")
    require(
        set(map(str, model_bundle["training_base_ids"]))
        == split_sets["readout_train"],
        f"{model_name}: readout training IDs differ from frozen split",
    )
    require(
        not (
            set(map(str, model_bundle["training_base_ids"]))
            & split_sets["causal_eval"]
        ),
        f"{model_name}: causal-evaluation leakage into readout training",
    )
    metrics = pd.read_csv(readout_dir / "实验E_22_clean_heldout_metrics.csv")
    require(
        set(metrics["split"].astype(str)) == {"train", "dev", "test"},
        f"{model_name}: missing E22 metric split",
    )
    test = predictions[
        predictions["base_id"].astype(str).isin(split_sets["readout_test"])
    ].copy()
    require(len(test) == 160, f"{model_name}: held-out test must have 160 rows")
    target = test[["probability_X", "probability_Y"]].to_numpy(dtype=np.float64)
    readout = test[
        ["readout_probability_X", "readout_probability_Y"]
    ].to_numpy(dtype=np.float64)
    target_preference = target.argmax(axis=1)
    readout_preference = readout.argmax(axis=1)
    gold_index = test["gold_option_label"].map(MAPPING).to_numpy(dtype=int)
    metric_test = metrics[metrics["split"].astype(str).eq("test")].iloc[0]
    close(
        metric_test["preference_agreement"],
        np.mean(readout_preference == target_preference),
        f"{model_name}: held-out preference agreement",
    )
    close(
        metric_test["preference_balanced_agreement"],
        balanced(target_preference, readout_preference),
        f"{model_name}: held-out balanced agreement",
    )
    close(
        metric_test["gold_accuracy"],
        np.mean(readout_preference == gold_index),
        f"{model_name}: held-out gold accuracy",
    )
    close(
        metric_test["option_probability_r2"],
        r2_score(target.reshape(-1), readout.reshape(-1)),
        f"{model_name}: held-out probability R2",
        atol=1e-9,
    )
    checks.extend(
        [
            "clean-only readout training IDs",
            "causal-evaluation leakage exclusion",
            "held-out metric recomputation",
        ]
    )

    causal_manifest = load_json(causal_dir / "实验E_23_manifest.json")
    require(
        int(causal_manifest["n_requested"]) == 500
        and int(causal_manifest["n_complete"]) == 500,
        f"{model_name}: causal restoration must complete 500 items",
    )
    require(
        int(causal_manifest["n_errors"]) == 0,
        f"{model_name}: causal restoration contains errors",
    )
    causal_rows = pd.read_csv(causal_dir / "实验E_23_causal_rows.csv.gz")
    require(
        len(causal_rows) == 3000,
        f"{model_name}: causal table must have 3000 rows",
    )
    mode_sets = causal_rows.groupby("base_id")["mode"].agg(
        lambda values: set(map(str, values))
    )
    require(
        len(mode_sets) == 500 and all(value == MODES for value in mode_sets),
        f"{model_name}: incomplete causal control modes",
    )
    paired = causal_rows.pivot(index="base_id", columns="mode")
    bias_delta = max(
        float(
            np.max(
                np.abs(
                    paired[f"readout_probability_{label}"]["output_logit_bias"]
                    - paired[f"readout_probability_{label}"]["target_baseline"]
                )
            )
        )
        for label in LABELS
    )
    require(
        bias_delta <= 1e-12,
        f"{model_name}: output-only bias altered the readout",
    )
    reproduction = pd.read_csv(causal_dir / "实验E_23_reproduction_checks.csv")
    require(
        len(reproduction) == 20
        and reproduction["prediction_match"].astype(bool).all(),
        f"{model_name}: failed causal reproduction checks",
    )
    checks.extend(
        [
            "500 paired causal restorations",
            "all preregistered controls",
            "output-only bias invariance",
            "saved-context reproduction",
        ]
    )
    decision = causal_manifest["decision"]
    return {
        "model_name": model_name,
        "checks_passed": checks,
        "activation_shapes": activation_shapes,
        "candidate_gold_X_base_count": int(
            metadata.drop_duplicates("base_id")["gold_option_label"]
            .astype(str)
            .eq("X")
            .sum()
        ),
        "heldout": metric_test.to_dict(),
        "natural_summary": load_json(
            readout_dir / "实验E_22_natural_summary.json"
        ),
        "causal_decision": decision,
        "artifacts": {
            "E21_manifest_sha256": sha256(run_dir / "manifest.json"),
            "E22_model_sha256": sha256(
                readout_dir / "实验E_22_readout.joblib"
            ),
            "E23_manifest_sha256": sha256(
                causal_dir / "实验E_23_manifest.json"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("NAME", "E21_DIR", "E22_DIR", "E23_DIR"),
        required=True,
    )
    args = parser.parse_args()
    split_document = load_json(Path(args.split_manifest))
    split_sets = {
        name: set(map(str, values))
        for name, values in split_document["split"].items()
    }
    expected_counts = {
        "causal_eval": 500,
        "readout_train": 500,
        "readout_dev": 159,
        "readout_test": 160,
    }
    require(
        {name: len(values) for name, values in split_sets.items()}
        == expected_counts,
        "E19 split counts changed",
    )
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            require(
                not (split_sets[left] & split_sets[right]),
                f"E19 split overlap: {left}/{right}",
            )
    require(
        len(set().union(*split_sets.values())) == 1319,
        "E19 split does not cover 1319 unique items",
    )
    results = []
    for model_name, run_dir, readout_dir, causal_dir in args.model:
        results.append(
            audit_model(
                model_name=model_name,
                run_dir=Path(run_dir),
                readout_dir=Path(readout_dir),
                causal_dir=Path(causal_dir),
                split_sets=split_sets,
            )
        )
    require(
        len({result["model_name"] for result in results}) == len(results),
        "Duplicate model names",
    )
    if len(results) > 1:
        require(
            len(
                {
                    result["artifacts"]["E22_model_sha256"]
                    for result in results
                }
            )
            == len(results),
            "The two models unexpectedly share an identical readout artifact",
        )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit = {
        "experiment": "E_24_independent_open_dialogue_validation",
        "status": "PASS",
        "split_counts": expected_counts,
        "models": results,
    }
    (outdir / "实验E_24_validation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    heldout = pd.DataFrame(
        [
            {"model_name": result["model_name"], **result["heldout"]}
            for result in results
        ]
    )
    heldout.to_csv(outdir / "实验E_24_heldout_metrics.csv", index=False)
    report = [
        "# 实验 E_24：开放式对话证据独立校验",
        "",
        "**结论：PASS。** 数据隔离、概率归一化、held-out 指标复算、"
        "500-item 配对因果修复和控制条件均通过结构审计。",
        "",
    ]
    for result in results:
        held = result["heldout"]
        decision = result["causal_decision"]
        report.extend(
            [
                f"## {result['model_name']}",
                "",
                f"- clean held-out preference agreement: "
                f"{float(held['preference_agreement']):.4f}",
                f"- clean held-out balanced agreement: "
                f"{float(held['preference_balanced_agreement']):.4f}",
                f"- option-probability R²: "
                f"{float(held['option_probability_r2']):.4f}",
                f"- option-probability KL: "
                f"{float(held['option_probability_kl']):.4f}",
                f"- causal paired items: "
                f"{int(decision['n_complete_paired_base'])}",
                f"- all preregistered claim gates pass: "
                f"{bool(decision['all_gates_pass'])}",
                "",
            ]
        )
    (outdir / "实验E_24_VALIDATION_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
