#!/usr/bin/env python3
"""Prepare locked MMLU/ARC main and option-permutation datasets.

Splits are assigned from a stable hash of the source item before option
permutation. The alternate dataset contains the same held-out questions with a
second option order, enabling a paired option-permutation consistency test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_pressure.config import load_config  # noqa: E402
from latent_pressure.data import pressure_turns  # noqa: E402
from 实验E_00_common import command_record, json_dump, sha256_file  # noqa: E402


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def assigned_split(
    source_id: str, seed: int, train_fraction: float, dev_fraction: float
) -> str:
    value = stable_int(f"split|{seed}|{source_id}") / float(16**16)
    if value < train_fraction:
        return "train"
    if value < train_fraction + dev_fraction:
        return "dev"
    return "test"


def shuffled_order(n: int, seed_text: str) -> List[int]:
    order = list(range(n))
    random.Random(stable_int(seed_text)).shuffle(order)
    return order


def remap_item(
    choices: List[str],
    answer_index: int,
    wrong_index: int,
    labels: List[str],
    order: List[int],
) -> Dict[str, Any]:
    inverse = {old_index: new_index for new_index, old_index in enumerate(order)}
    return {
        "options": [
            {"label": labels[new_index], "text": str(choices[old_index])}
            for new_index, old_index in enumerate(order)
        ],
        "true_label": labels[inverse[answer_index]],
        "wrong_label": labels[inverse[wrong_index]],
    }


def load_source(
    source: str, split: str, local_file: str | None = None
) -> Iterable[Dict[str, Any]]:
    if local_file:
        path = Path(local_file)
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                choice_block = row["choices"]
                texts = [str(value) for value in choice_block["text"]]
                source_labels = [str(value) for value in choice_block["label"]]
                answer_key = str(row["answerKey"])
                if answer_key not in source_labels:
                    continue
                yield {
                    "source_id": f"{source}:{row.get('id', index)}",
                    "question": str(row["question"]),
                    "choices": texts,
                    "answer_index": source_labels.index(answer_key),
                    "subject": (
                        "ARC-Challenge" if source == "arc" else source
                    ),
                }
        return

    from datasets import load_dataset

    if source == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split=split)
        for index, row in enumerate(dataset):
            choices = [str(value) for value in row["choices"]]
            yield {
                "source_id": f"mmlu:{row.get('subject', 'unknown')}:{index}",
                "question": str(row["question"]),
                "choices": choices,
                "answer_index": int(row["answer"]),
                "subject": str(row.get("subject", "unknown")),
            }
        return

    if source == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        for index, row in enumerate(dataset):
            choice_block = row["choices"]
            texts = [str(value) for value in choice_block["text"]]
            source_labels = [str(value) for value in choice_block["label"]]
            answer_key = str(row["answerKey"])
            if answer_key not in source_labels:
                continue
            yield {
                "source_id": f"arc:{row.get('id', index)}",
                "question": str(row["question"]),
                "choices": texts,
                "answer_index": source_labels.index(answer_key),
                "subject": "ARC-Challenge",
            }
        return
    raise ValueError(f"Unsupported source={source!r}")


def choose_items(
    rows: Iterable[Dict[str, Any]],
    labels: List[str],
    max_items: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if len(row["choices"]) == len(labels)
        and 0 <= int(row["answer_index"]) < len(labels)
    ]
    eligible.sort(key=lambda row: stable_int(f"sample|{seed}|{row['source_id']}"))
    if max_items is not None:
        eligible = eligible[: int(max_items)]
    return eligible


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=["mmlu", "arc"], required=True)
    parser.add_argument("--source_split", default="test")
    parser.add_argument(
        "--local_file",
        default=None,
        help="Optional source JSONL; useful when the server has no HF access.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train_fraction", type=float, default=0.60)
    parser.add_argument("--dev_fraction", type=float, default=0.20)
    parser.add_argument("--trajectory_types", default="wrong_pressure")
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.dev_fraction <= 0:
        raise ValueError("train_fraction and dev_fraction must be positive")
    if args.train_fraction + args.dev_fraction >= 1:
        raise ValueError("train_fraction + dev_fraction must be below 1")

    cfg = load_config(args.config)
    labels = [str(value) for value in cfg["experiment"].get("labels", "ABCD")]
    trajectories = [
        value.strip()
        for value in str(args.trajectory_types).split(",")
        if value.strip()
    ]
    selected = choose_items(
        load_source(args.source, args.source_split, args.local_file),
        labels=labels,
        max_items=args.max_items,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("No eligible four-option items found")

    main_records: List[Dict[str, Any]] = []
    alternate_records: List[Dict[str, Any]] = []
    lock_rows: List[Dict[str, Any]] = []
    split_counts = {"train": 0, "dev": 0, "test": 0}

    for item_index, item in enumerate(selected):
        source_id = str(item["source_id"])
        split = assigned_split(
            source_id,
            seed=args.seed,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
        )
        split_counts[split] += 1
        n_options = len(labels)
        answer_index = int(item["answer_index"])
        wrong_candidates = [index for index in range(n_options) if index != answer_index]
        wrong_index = wrong_candidates[
            stable_int(f"wrong|{args.seed}|{source_id}") % len(wrong_candidates)
        ]
        main_order = shuffled_order(n_options, f"main|{args.seed}|{source_id}")
        alternate_order = shuffled_order(
            n_options, f"alternate|{args.seed}|{source_id}"
        )
        if alternate_order == main_order:
            alternate_order = alternate_order[1:] + alternate_order[:1]

        main_item = remap_item(
            item["choices"], answer_index, wrong_index, labels, main_order
        )
        alternate_item = remap_item(
            item["choices"], answer_index, wrong_index, labels, alternate_order
        )
        base_hash = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
        base_id = f"E_{args.source}_{base_hash}"

        def build_records(permuted: Dict[str, Any], permutation_id: str):
            output = []
            for trajectory in trajectories:
                output.append(
                    {
                        "record_id": f"{base_id}__{permutation_id}__{trajectory}",
                        "base_id": base_id,
                        "source_id": source_id,
                        "dataset": f"{args.source}:{args.source_split}",
                        "subject": item["subject"],
                        "split": split,
                        "trajectory_type": trajectory,
                        "true_label": permuted["true_label"],
                        "wrong_label": permuted["wrong_label"],
                        "options": permuted["options"],
                        "turns": pressure_turns(
                            trajectory,
                            permuted["true_label"],
                            permuted["wrong_label"],
                        ),
                        "source_question": item["question"],
                        "fact_context": "",
                        "item_code": base_id,
                        "item_mode": "external_mcqa_experiment_e",
                        "permutation_id": permutation_id,
                    }
                )
            return output

        main_records.extend(build_records(main_item, "main"))
        if split == "test":
            alternate_records.extend(build_records(alternate_item, "alternate"))
        lock_rows.append(
            {
                "item_index": item_index,
                "base_id": base_id,
                "source_id": source_id,
                "subject": item["subject"],
                "split": split,
                "canonical_answer_index": answer_index,
                "canonical_wrong_index": wrong_index,
                "main_order": json.dumps(main_order),
                "alternate_order": json.dumps(alternate_order),
                "main_true_label": main_item["true_label"],
                "alternate_true_label": alternate_item["true_label"],
                "main_wrong_label": main_item["wrong_label"],
                "alternate_wrong_label": alternate_item["wrong_label"],
            }
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    main_path = outdir / f"实验E_00_{args.source}_main.jsonl"
    alternate_path = outdir / f"实验E_00_{args.source}_option_permutation_test.jsonl"
    lock_path = outdir / f"实验E_00_{args.source}_split_lock.csv"
    manifest_path = outdir / f"实验E_00_{args.source}_manifest.json"
    write_jsonl(main_records, main_path)
    write_jsonl(alternate_records, alternate_path)
    with lock_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lock_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lock_rows)
    json_dump(
        {
            "experiment": "E_00_external_data_lock",
            "command": command_record(),
            "source": args.source,
            "source_split": args.source_split,
            "local_file": (
                str(Path(args.local_file).resolve()) if args.local_file else None
            ),
            "seed": args.seed,
            "labels": labels,
            "trajectory_types": trajectories,
            "n_source_items": len(selected),
            "n_main_records": len(main_records),
            "n_alternate_test_records": len(alternate_records),
            "split_counts": split_counts,
            "train_fraction": args.train_fraction,
            "dev_fraction": args.dev_fraction,
            "main_path": str(main_path.resolve()),
            "main_sha256": sha256_file(main_path),
            "alternate_path": str(alternate_path.resolve()),
            "alternate_sha256": sha256_file(alternate_path),
            "lock_path": str(lock_path.resolve()),
            "lock_sha256": sha256_file(lock_path),
        },
        manifest_path,
    )
    print(json.dumps({"split_counts": split_counts}, ensure_ascii=False))
    print(f"[OK] {main_path}")
    print(f"[OK] {alternate_path}")
    print(f"[OK] {lock_path}")


if __name__ == "__main__":
    main()
