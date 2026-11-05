#!/usr/bin/env python3
"""Prepare the common N=500 full-trajectory and all-history T5-fork panels."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from exp_B_prepare import canonical_wrong_records, prepare_panel1, validate_rendered
from exp_B_templates import FAMILY_ORDER, TEMPLATES, make_fork_turn, registry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_results(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files matched {pattern}")
    rows: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(rows)
    frame.attrs["source_files"] = files
    return frame


def write_template_manifest(outdir: Path) -> None:
    payload = {
        "experiment": "B",
        "design": "common_n500_full_trajectory_plus_all_history_t5_fork",
        "family_order": FAMILY_ORDER,
        "n_families": len(FAMILY_ORDER),
        "variants_per_family": 3,
        "templates": list(registry(include_neutral=True)),
        "fixed_design_note": (
            "The same 500 base items, wrong targets, turn structure, and frozen "
            "template registry are used for every model. Panel 2 forks every "
            "canonical direct-assertion T0-T4 history; strict PRR relapse is "
            "reported only for the behaviorally defined surface-recovered stratum."
        ),
    }
    (outdir / "exp_B_template_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_panel1_only(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    outdir = Path(args.outdir)
    records = canonical_wrong_records(dataset, args.n_items)
    write_template_manifest(outdir)
    audit = prepare_panel1(records, outdir, args.model_tag, args.n_shards)
    payload = {
        "experiment": "B",
        "mode": "panel1",
        "model_tag": args.model_tag,
        "source_dataset": str(dataset),
        "source_dataset_sha256": sha256_file(dataset),
        "n_items": args.n_items,
        "n_shards": args.n_shards,
        "panel1": audit,
    }
    (outdir / f"exp_B_prepare_panel1_audit_{args.model_tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def canonical_histories(
    panel1: pd.DataFrame,
    ordered_base_ids: List[str],
) -> pd.DataFrame:
    canonical = panel1[
        panel1["pressure_family"].eq("direct_assertion")
        & panel1["pressure_variant"].eq("legacy_anchor")
        & panel1["turn_index"].between(0, 4)
    ].copy()
    duplicate_count = int(canonical.duplicated(["base_id", "turn_index"]).sum())
    if duplicate_count:
        raise ValueError(f"Canonical panel has {duplicate_count} duplicate base/turn rows")

    rows: List[Dict[str, Any]] = []
    by_id = {str(k): v for k, v in canonical.groupby("base_id", sort=False)}
    for base_id in ordered_base_ids:
        if base_id not in by_id:
            raise ValueError(f"Missing canonical direct-assertion history for {base_id}")
        group = by_id[base_id].sort_values("turn_index")
        turns = group["turn_index"].astype(int).tolist()
        if turns != [0, 1, 2, 3, 4]:
            raise ValueError(f"Incomplete canonical history for {base_id}: {turns}")
        preds = group["pred_label"].astype(str).tolist()
        t0 = group.iloc[0]
        t3 = group.iloc[3]
        t4 = group.iloc[4]
        t0_correct = bool(t0["output_correct"])
        t3_wrong = bool(t3["output_wrong"])
        t4_correct = bool(t4["output_correct"])
        rows.append(
            {
                "base_id": base_id,
                "canonical_history_T0_T4": preds,
                "cohort_t0_correct": t0_correct,
                "cohort_t4_correct": t4_correct,
                "cohort_t0_t4_correct": t0_correct and t4_correct,
                "cohort_surface_recovered": t0_correct and t3_wrong and t4_correct,
                "cohort_persistent_hard_wrong": t0_correct
                and t3_wrong
                and not t4_correct,
            }
        )
    return pd.DataFrame(rows)


def prepare_panel2_all_histories(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    outdir = Path(args.outdir)
    records = canonical_wrong_records(dataset, args.n_items)
    ordered_base_ids = [str(row["base_id"]) for row in records]
    by_id = {str(row["base_id"]): row for row in records}
    panel1 = load_results(args.panel1_glob)
    panel1["base_id"] = panel1["base_id"].astype(str)
    panel1["turn_index"] = pd.to_numeric(panel1["turn_index"], errors="raise").astype(int)
    histories = canonical_histories(panel1, ordered_base_ids)
    history_map = histories.set_index("base_id").to_dict(orient="index")

    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(args.n_shards)]
    templates = list(registry(include_neutral=True))
    for item_index, base_id in enumerate(ordered_base_ids):
        source = by_id[base_id]
        history = history_map[base_id]
        for template in templates:
            row = deepcopy(source)
            family = str(template["family"])
            variant = str(template["variant"])
            row["trajectory_type"] = "exp_B_panel2_all_history_t5_fork"
            row["pressure_family"] = family
            row["pressure_variant"] = variant
            row["template_cell"] = f"{family}__{variant}"
            row["record_id"] = (
                f"exp_B_allfork::{args.model_tag}::{base_id}::{family}::{variant}"
            )
            row["turns"] = deepcopy(source["turns"][:5]) + [make_fork_turn(row, template)]
            row["canonical_history_T0_T4"] = list(history["canonical_history_T0_T4"])
            for key, value in history.items():
                if key != "canonical_history_T0_T4":
                    row[key] = bool(value)
            validate_rendered(row, [row["turns"][5]], family)
            shard_rows[item_index % args.n_shards].append(row)

    for shard, rows in enumerate(shard_rows):
        write_jsonl(
            outdir
            / "panel2"
            / f"exp_B_panel2_{args.model_tag}_shard{shard}.jsonl",
            rows,
        )
    population = histories.drop(columns=["canonical_history_T0_T4"])
    population.to_csv(
        outdir / "population" / f"exp_B_panel2_population_{args.model_tag}.csv",
        index=False,
    )
    cohort_counts = {
        col: int(population[col].astype(bool).sum())
        for col in population.columns
        if col.startswith("cohort_")
    }
    payload = {
        "experiment": "B",
        "mode": "panel2_all_history_t5_fork",
        "model_tag": args.model_tag,
        "source_dataset": str(dataset),
        "source_dataset_sha256": sha256_file(dataset),
        "panel1_result_files": panel1.attrs["source_files"],
        "n_items": args.n_items,
        "n_fork_cells": len(templates),
        "n_records": sum(len(rows) for rows in shard_rows),
        "n_shards": args.n_shards,
        "records_per_shard": [len(rows) for rows in shard_rows],
        "cohort_counts": cohort_counts,
        "interpretation_rule": (
            "Use the surface-recovered stratum for strict PRR relapse. "
            "Other strata measure paired pressure-template sensitivity but are "
            "not labeled post-recovery relapse."
        ),
    }
    (outdir / f"exp_B_prepare_panel2_audit_{args.model_tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ["panel1", "panel2"]:
        child = sub.add_parser(mode)
        child.add_argument("--dataset", required=True)
        child.add_argument("--outdir", required=True)
        child.add_argument("--model-tag", required=True)
        child.add_argument("--n-items", type=int, default=500)
        child.add_argument("--n-shards", type=int, default=1)
        if mode == "panel2":
            child.add_argument("--panel1-glob", required=True)
    args = parser.parse_args()
    Path(args.outdir, "population").mkdir(parents=True, exist_ok=True)
    if args.mode == "panel1":
        prepare_panel1_only(args)
    else:
        prepare_panel2_all_histories(args)


if __name__ == "__main__":
    main()
