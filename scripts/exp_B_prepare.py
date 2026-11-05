#!/usr/bin/env python3
"""Prepare frozen datasets and populations for Experiment B."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from exp_B_templates import FAMILY_ORDER, TEMPLATES, make_fork_turn, make_turns, registry


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def stable_score(base_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{base_id}".encode("utf-8")).hexdigest()


def canonical_wrong_records(dataset: Path, n_items: int | None = None) -> List[Dict[str, Any]]:
    rows = read_jsonl(dataset)
    selected: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        if row.get("trajectory_type") != "wrong_pressure":
            continue
        base_id = str(row["base_id"])
        if base_id in seen:
            continue
        seen.add(base_id)
        selected.append(row)
        if n_items is not None and len(selected) >= n_items:
            break
    if n_items is not None and len(selected) != n_items:
        raise ValueError(f"Requested {n_items} canonical items, found {len(selected)} in {dataset}")
    return selected


def validate_rendered(record: Dict[str, Any], turns: List[Dict[str, Any]], family: str) -> None:
    true_label = str(record["true_label"])
    wrong_label = str(record["wrong_label"])
    if true_label == wrong_label:
        raise ValueError(f"true_label == wrong_label for {record['base_id']}")
    for turn in turns:
        msg = str(turn["user_message"])
        if "{" in msg or "}" in msg:
            raise ValueError(f"Unrendered placeholder: {msg}")
        if turn["turn_index"] in {1, 2, 3, 5} and family != "neutral_control":
            if wrong_label not in msg:
                raise ValueError(f"Wrong target missing from pressure message: {msg}")
        if family == "evidence_containing" and turn["turn_index"] in {1, 2, 3, 5}:
            if (
                str(record["wrong_entity"]) not in msg
                and "other three" not in msg
                and "Eliminating the entries" not in msg
            ):
                raise ValueError(f"Evidence message lacks its deterministic evidence anchor: {msg}")
            if str(record["true_entity"]) in msg:
                raise ValueError(f"Evidence message leaks true entity: {msg}")


def prepare_panel1(
    records: List[Dict[str, Any]],
    outdir: Path,
    model_tag: str,
    n_shards: int,
) -> Dict[str, Any]:
    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(n_shards)]
    for item_index, source in enumerate(records):
        for template in TEMPLATES:
            row = deepcopy(source)
            family = str(template["family"])
            variant = str(template["variant"])
            row["trajectory_type"] = "exp_B_panel1"
            row["pressure_family"] = family
            row["pressure_variant"] = variant
            row["template_cell"] = f"{family}__{variant}"
            row["record_id"] = f"exp_B::{model_tag}::{row['base_id']}::{family}::{variant}"
            row["turns"] = make_turns(row, template, include_final=True)
            validate_rendered(row, row["turns"], family)
            shard_rows[item_index % n_shards].append(row)
    for shard, rows in enumerate(shard_rows):
        write_jsonl(outdir / "panel1" / f"exp_B_panel1_{model_tag}_shard{shard}.jsonl", rows)
    return {
        "n_base_items": len(records),
        "n_template_cells": len(TEMPLATES),
        "n_records": sum(len(x) for x in shard_rows),
        "n_turn_rows_expected": sum(len(x) for x in shard_rows) * 7,
        "n_shards": n_shards,
        "records_per_shard": [len(x) for x in shard_rows],
    }


def select_surface_population(scored_rows: Path, max_items: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(scored_rows)
    frame["base_id"] = frame["base_id"].astype(str)
    frame["turn_index"] = pd.to_numeric(frame["turn_index"], errors="raise").astype(int)
    # Reconstruct strict target labels from predictions for compatibility with
    # older state_rows files, whose output_wrong column was pairwise-margin based.
    if {"pred_label", "true_label", "wrong_label"}.issubset(frame.columns):
        frame["output_correct"] = (
            frame["pred_label"].astype(str) == frame["true_label"].astype(str)
        )
        frame["output_wrong"] = (
            frame["pred_label"].astype(str) == frame["wrong_label"].astype(str)
        )
    else:
        for col in ["output_correct", "output_wrong"]:
            frame[col] = as_bool(frame[col])
    wp = frame[frame["trajectory_type"].eq("wrong_pressure")].copy()
    t0 = set(wp[(wp["turn_index"] == 0) & wp["output_correct"]]["base_id"])
    t3 = set(wp[(wp["turn_index"] == 3) & wp["output_wrong"]]["base_id"])
    t4 = set(wp[(wp["turn_index"] == 4) & wp["output_correct"]]["base_id"])
    surface = sorted(t0 & t3 & t4)
    population = pd.DataFrame(
        {
            "base_id": surface,
            "selection_score": [stable_score(x, seed) for x in surface],
        }
    ).sort_values(["selection_score", "base_id"])
    selected = population.head(max_items).copy()
    selected.insert(0, "selection_rank", range(len(selected)))
    selected.attrs["n_unique_base"] = int(wp["base_id"].nunique())
    selected.attrs["n_init"] = len(t0)
    selected.attrs["n_hard_wrong_among_init"] = len(t0 & t3)
    selected.attrs["n_surface_available"] = len(surface)
    return selected


def canonical_history_map(metadata_path: Path) -> Dict[str, List[str]]:
    frame = pd.read_csv(metadata_path)
    frame["base_id"] = frame["base_id"].astype(str)
    frame["turn_index"] = pd.to_numeric(frame["turn_index"], errors="raise").astype(int)
    wp = frame[frame["trajectory_type"].eq("wrong_pressure") & frame["turn_index"].between(0, 4)]
    histories: Dict[str, List[str]] = {}
    for base_id, group in wp.groupby("base_id", sort=False):
        ordered = group.sort_values("turn_index")
        turns = ordered["turn_index"].tolist()
        if turns != [0, 1, 2, 3, 4]:
            raise ValueError(f"Incomplete canonical T0-T4 history for {base_id}: {turns}")
        histories[base_id] = ordered["pred_label"].astype(str).tolist()
    return histories


def prepare_panel2(
    records: List[Dict[str, Any]],
    selected: pd.DataFrame,
    canonical_metadata: Path,
    outdir: Path,
    model_tag: str,
    n_shards: int,
) -> Dict[str, Any]:
    by_id = {str(r["base_id"]): r for r in records}
    histories = canonical_history_map(canonical_metadata)
    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(n_shards)]
    all_templates = list(registry(include_neutral=True))
    missing = []
    for item_index, (_, selected_row) in enumerate(selected.iterrows()):
        base_id = str(selected_row["base_id"])
        if base_id not in by_id or base_id not in histories:
            missing.append(base_id)
            continue
        source = by_id[base_id]
        for template in all_templates:
            row = deepcopy(source)
            family = str(template["family"])
            variant = str(template["variant"])
            row["trajectory_type"] = "exp_B_panel2_t5_fork"
            row["pressure_family"] = family
            row["pressure_variant"] = variant
            row["template_cell"] = f"{family}__{variant}"
            row["record_id"] = f"exp_B_fork::{model_tag}::{base_id}::{family}::{variant}"
            row["turns"] = deepcopy(source["turns"][:5]) + [make_fork_turn(row, template)]
            row["canonical_history_T0_T4"] = histories[base_id]
            validate_rendered(row, [row["turns"][5]], family)
            shard_rows[item_index % n_shards].append(row)
    if missing:
        raise ValueError(f"Missing canonical records/histories for {len(missing)} IDs: {missing[:5]}")
    for shard, rows in enumerate(shard_rows):
        write_jsonl(outdir / "panel2" / f"exp_B_panel2_{model_tag}_shard{shard}.jsonl", rows)
    selected.to_csv(outdir / "population" / f"exp_B_panel2_population_{model_tag}.csv", index=False)
    return {
        "n_selected_base_items": len(selected),
        "n_fork_cells": len(all_templates),
        "n_records": sum(len(x) for x in shard_rows),
        "n_forward_rows_expected": sum(len(x) for x in shard_rows),
        "n_shards": n_shards,
        "records_per_shard": [len(x) for x in shard_rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--canonical-metadata", required=True)
    ap.add_argument("--canonical-scored-rows", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--panel1-items", type=int, required=True)
    ap.add_argument("--panel2-items", type=int, default=200)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2508)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "population").mkdir(parents=True, exist_ok=True)
    # Load the complete canonical item bank because the hashed Panel 2 population
    # may include IDs beyond the Panel 1 prefix.
    records = canonical_wrong_records(Path(args.dataset), None)
    if len(records) < args.panel1_items:
        raise ValueError(
            f"Panel 1 requests {args.panel1_items} items, but dataset has {len(records)}"
        )
    panel1_records = records[: args.panel1_items]
    selected = select_surface_population(
        Path(args.canonical_scored_rows), args.panel2_items, args.seed
    )
    if len(selected) < args.panel2_items:
        raise ValueError(
            f"Requested {args.panel2_items} surface-recovered items, but only "
            f"{len(selected)} are available."
        )

    template_manifest = {
        "experiment": "B",
        "family_order": FAMILY_ORDER,
        "n_families": len(FAMILY_ORDER),
        "variants_per_family": 3,
        "templates": list(registry(include_neutral=True)),
        "fixed_design_note": (
            "Frozen before outcome inspection; item-specific evidence is rendered "
            "deterministically from wrong_entity and target field/value."
        ),
    }
    (outdir / "exp_B_template_manifest.json").write_text(
        json.dumps(template_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    panel1_audit = prepare_panel1(panel1_records, outdir, args.model_tag, args.n_shards)
    panel2_audit = prepare_panel2(
        records,
        selected,
        Path(args.canonical_metadata),
        outdir,
        args.model_tag,
        args.n_shards,
    )
    audit = {
        "experiment": "B",
        "model_tag": args.model_tag,
        "source_dataset": args.dataset,
        "canonical_metadata": args.canonical_metadata,
        "canonical_scored_rows": args.canonical_scored_rows,
        "seed": args.seed,
        "surface_population": {
            "n_unique_base": selected.attrs["n_unique_base"],
            "n_init": selected.attrs["n_init"],
            "n_hard_wrong_among_init": selected.attrs["n_hard_wrong_among_init"],
            "n_surface_available": selected.attrs["n_surface_available"],
            "n_selected": len(selected),
            "selection_rule": "SHA256(seed, base_id), ascending; no T5 conditioning",
        },
        "panel1": panel1_audit,
        "panel2": panel2_audit,
    }
    (outdir / f"exp_B_prepare_audit_{args.model_tag}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
