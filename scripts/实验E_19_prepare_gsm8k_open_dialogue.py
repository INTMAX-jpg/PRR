#!/usr/bin/env python3
"""Freeze the open-dialogue GSM8K causal/readout split.

The first 500 shuffled test items exactly reproduce Stage-17 causal items.
The remaining 819 official GSM8K test items are used only for clean readout
training, development and held-out validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_pressure.numeric_answers import canonical_decimal, parse_number


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_wrong_target(gold: Decimal, question: str, seed: int) -> Decimal:
    digest = hashlib.sha256(f"{seed}\0{question}".encode("utf-8")).digest()
    multipliers = [Decimal(1), Decimal(2), Decimal(3), Decimal(5), Decimal(7)]
    multiplier = multipliers[digest[0] % len(multipliers)]
    magnitude = abs(gold)
    if gold == gold.to_integral_value():
        if magnitude >= 1000:
            scale = Decimal(10) ** max(0, len(str(abs(int(gold)))) - 3)
        elif magnitude >= 100:
            scale = Decimal(10)
        else:
            scale = Decimal(1)
    else:
        exponent = magnitude.adjusted() if magnitude else 0
        scale = max(Decimal(10) ** (exponent - 1), Decimal("0.01"))
    delta = multiplier * scale
    candidate = gold - delta if digest[1] % 2 else gold + delta
    if gold >= 0 and candidate < 0:
        candidate = gold + delta
    if candidate == gold:
        candidate = gold + max(Decimal(1), delta)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_jsonl", default="data/source/gsm8k_test_official.jsonl"
    )
    parser.add_argument(
        "--stage17_causal", default="data/gsm8k_stage17_test_500.jsonl"
    )
    parser.add_argument(
        "--out", default="data/实验E_19_gsm8k_open_dialogue_1319.jsonl"
    )
    parser.add_argument(
        "--split_manifest",
        default="data/实验E_19_gsm8k_open_dialogue_split.json",
    )
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    source_path = Path(args.source_jsonl)
    stage17_path = Path(args.stage17_causal)
    source = read_jsonl(source_path)
    if len(source) != 1319:
        raise ValueError(f"Expected 1319 official test items, got {len(source)}")
    order = list(range(len(source)))
    random.Random(args.seed).shuffle(order)
    split_indices = {
        "causal_eval": order[:500],
        "readout_train": order[500:1000],
        "readout_dev": order[1000:1159],
        "readout_test": order[1159:1319],
    }
    if {key: len(value) for key, value in split_indices.items()} != {
        "causal_eval": 500,
        "readout_train": 500,
        "readout_dev": 159,
        "readout_test": 160,
    }:
        raise AssertionError("Unexpected split counts")
    all_indices = [index for values in split_indices.values() for index in values]
    if len(all_indices) != len(set(all_indices)) or set(all_indices) != set(
        range(1319)
    ):
        raise AssertionError("Split is not a disjoint cover of GSM8K test")

    index_to_split = {
        index: split_name
        for split_name, values in split_indices.items()
        for index in values
    }
    index_to_rank = {
        index: rank for rank, index in enumerate(order)
    }
    rows: List[Dict[str, Any]] = []
    for dataset_index in order:
        item = source[dataset_index]
        answer_text = str(item["answer"])
        if "####" not in answer_text:
            raise ValueError(f"Missing final marker at index {dataset_index}")
        gold = parse_number(answer_text.rsplit("####", 1)[1].strip())
        wrong = make_wrong_target(gold, str(item["question"]), int(args.seed))
        rows.append(
            {
                "base_id": f"gsm8k_test_{dataset_index:04d}",
                "dataset": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "dataset_index": int(dataset_index),
                "sample_rank": int(index_to_rank[dataset_index]),
                "experiment_split": index_to_split[dataset_index],
                "question": str(item["question"]),
                "gold_answer": canonical_decimal(gold),
                "wrong_answer": canonical_decimal(wrong),
                "gold_rationale": answer_text,
                "selection_seed": int(args.seed),
            }
        )

    stage17 = read_jsonl(stage17_path)
    causal_rows = [row for row in rows if row["experiment_split"] == "causal_eval"]
    if len(stage17) != 500 or len(causal_rows) != 500:
        raise AssertionError("Stage-17 or E19 causal count is not 500")
    comparison_fields = [
        "base_id",
        "dataset_index",
        "question",
        "gold_answer",
        "wrong_answer",
    ]
    for old, new in zip(stage17, causal_rows):
        if any(str(old[field]) != str(new[field]) for field in comparison_fields):
            raise ValueError(
                f"E19 causal order/content does not reproduce Stage-17: "
                f"{old['base_id']} vs {new['base_id']}"
            )

    out_path = Path(args.out)
    split_path = Path(args.split_manifest)
    write_jsonl(out_path, rows)
    split = {
        name: [
            f"gsm8k_test_{dataset_index:04d}" for dataset_index in values
        ]
        for name, values in split_indices.items()
    }
    split_path.write_text(
        json.dumps(
            {
                "experiment": "E_19_gsm8k_open_dialogue_split",
                "dataset": "openai/gsm8k main test",
                "source_path": str(source_path),
                "source_sha256": sha256(source_path),
                "seed": int(args.seed),
                "selection_variables": ["dataset_index", "seed"],
                "explicitly_unused_for_selection": [
                    "model output",
                    "free-form answer",
                    "pressure response",
                    "recovery",
                    "readout",
                    "intervention outcome",
                ],
                "split": split,
                "counts": {name: len(values) for name, values in split.items()},
                "causal_exactly_matches_stage17_500": True,
                "output_path": str(out_path),
                "output_sha256": sha256(out_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(split_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
