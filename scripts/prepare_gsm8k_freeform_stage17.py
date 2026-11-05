#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path

from datasets import load_dataset

from latent_pressure.numeric_answers import canonical_decimal, parse_number


def make_wrong_target(gold: Decimal, question: str, seed: int) -> Decimal:
    """Create a deterministic, plausible, non-gold numeric target."""
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
        scale = Decimal(10) ** (exponent - 1)
        scale = max(scale, Decimal("0.01"))
    delta = multiplier * scale
    candidate = gold - delta if digest[1] % 2 else gold + delta
    if gold >= 0 and candidate < 0:
        candidate = gold + delta
    if candidate == gold:
        candidate = gold + max(Decimal(1), delta)
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a deterministic GSM8K Stage-17 subset.")
    ap.add_argument("--out", default="data/gsm8k_stage17_test_500.jsonl")
    ap.add_argument("--cache-dir", default="data/hf_cache/gsm8k")
    ap.add_argument(
        "--source-jsonl",
        default=None,
        help="Optional official GSM8K JSONL; avoids Hugging Face network access.",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1701)
    args = ap.parse_args()

    if args.source_jsonl:
        source_path = Path(args.source_jsonl)
        with source_path.open("r", encoding="utf-8") as handle:
            dataset = [json.loads(line) for line in handle if line.strip()]
        dataset_source = str(source_path)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    else:
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split=args.split,
            cache_dir=args.cache_dir,
        )
        dataset_source = "Hugging Face: openai/gsm8k"
        source_sha256 = None
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.max_items is not None:
        indices = indices[: args.max_items]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w", encoding="utf-8") as handle:
        for sample_rank, dataset_index in enumerate(indices):
            source = dataset[dataset_index]
            answer_text = str(source["answer"])
            if "####" not in answer_text:
                raise ValueError(f"Missing GSM8K final marker at dataset index {dataset_index}")
            gold_raw = answer_text.rsplit("####", 1)[1].strip()
            gold = parse_number(gold_raw)
            wrong = make_wrong_target(gold, str(source["question"]), args.seed)
            if wrong == gold:
                raise AssertionError("Wrong target unexpectedly equals gold")
            row = {
                "base_id": f"gsm8k_{args.split}_{dataset_index:04d}",
                "dataset": "openai/gsm8k",
                "config": "main",
                "split": args.split,
                "dataset_index": dataset_index,
                "sample_rank": sample_rank,
                "question": source["question"],
                "gold_answer": canonical_decimal(gold),
                "wrong_answer": canonical_decimal(wrong),
                "gold_rationale": answer_text,
                "selection_seed": args.seed,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    manifest = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": args.split,
        "seed": args.seed,
        "n_source": len(dataset),
        "n_written": n_written,
        "output": str(out),
        "source": dataset_source,
        "source_sha256": source_sha256,
        "wrong_target_rule": "deterministic SHA256-seeded signed numeric perturbation",
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
