#!/usr/bin/env python3
"""Generate clean T0 free-form GSM8K responses for the readout pool only."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latent_pressure.config import load_config, save_json, set_seed
from latent_pressure.hf_model import first_param_device, load_tokenizer_and_model
from latent_pressure.numeric_answers import extract_final_number


READOUT_SPLITS = {"readout_train", "readout_dev", "readout_test"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["base_id"])
        for row in load_jsonl(path)
        if str(row.get("response", "")).strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset", default="data/实验E_19_gsm8k_open_dialogue_1319.jsonl"
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["experiment"].get("seed", 1701))
    set_seed(seed)
    rows = [
        row
        for row in load_jsonl(Path(args.dataset))
        if str(row["experiment_split"]) in READOUT_SPLITS
    ]
    if len(rows) != 819:
        raise ValueError(f"Expected 819 readout-pool items, got {len(rows)}")
    if args.max_items is not None:
        rows = rows[: int(args.max_items)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "实验E_20_clean_generations.jsonl"
    done = set() if args.no_resume else completed_ids(output_path)
    pending = [row for row in rows if str(row["base_id"]) not in done]

    tokenizer, model = load_tokenizer_and_model(cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_cfg = cfg.get("generation", {})
    chat_template_kwargs = {}
    if "enable_thinking" in generation_cfg:
        chat_template_kwargs["enable_thinking"] = bool(
            generation_cfg["enable_thinking"]
        )
    if not bool(generation_cfg.get("do_sample", False)):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    system_prompt = str(cfg["prompt"]["system_template"]).strip()
    initial_template = str(cfg["prompt"]["initial_user_template"])
    device = first_param_device(model)
    mode = "w" if args.no_resume else "a"
    started = time.time()
    print(
        f"[E20] assigned={len(rows)} completed={len(done)} pending={len(pending)}",
        flush=True,
    )
    with output_path.open(mode, encoding="utf-8") as handle:
        for item_index, row in enumerate(pending, 1):
            user_prompt = initial_template.format(
                question=row["question"], wrong_answer=row["wrong_answer"]
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **chat_template_kwargs,
            ).to(device)
            max_context = int(generation_cfg.get("max_context_tokens", 8192))
            if int(input_ids.shape[1]) > max_context:
                raise RuntimeError(f"Input exceeds max_context_tokens: {row['base_id']}")
            kwargs = {
                "max_new_tokens": int(generation_cfg.get("max_new_tokens", 512)),
                "do_sample": bool(generation_cfg.get("do_sample", False)),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if kwargs["do_sample"]:
                kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
                kwargs["top_p"] = float(generation_cfg.get("top_p", 0.95))
            generation_started = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    **kwargs,
                )
            generated_ids = output_ids[0, input_ids.shape[1] :]
            response = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            parsed = extract_final_number(response)
            hit_max = int(generated_ids.shape[0]) >= int(kwargs["max_new_tokens"])
            if hit_max and not parsed.marker_found:
                parsed_answer = None
                parse_source = "truncated_no_marker"
            else:
                parsed_answer = parsed.normalized
                parse_source = parsed.source
            result = {
                "base_id": row["base_id"],
                "dataset_index": row["dataset_index"],
                "experiment_split": row["experiment_split"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "wrong_answer": row["wrong_answer"],
                "user_prompt": user_prompt,
                "response": response,
                "parsed_answer_online": parsed_answer,
                "parse_source_online": parse_source,
                "marker_found_online": bool(parsed.marker_found),
                "input_tokens": int(input_ids.shape[1]),
                "output_tokens": int(generated_ids.shape[0]),
                "hit_max_new_tokens": bool(hit_max),
                "generation_seconds": time.time() - generation_started,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"[E20] {item_index}/{len(pending)} {row['base_id']} "
                f"split={row['experiment_split']} parsed={parsed_answer or 'NA'}",
                flush=True,
            )

    save_json(
        {
            "experiment": "E_20_clean_gsm8k_freeform",
            "config": args.config,
            "dataset": args.dataset,
            "model_path": cfg["model"]["path"],
            "n_assigned": len(rows),
            "n_previously_completed": len(done),
            "n_newly_completed": len(pending),
            "output": str(output_path),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "generation": generation_cfg,
            "elapsed_seconds": time.time() - started,
        },
        outdir / "实验E_20_manifest.json",
    )


if __name__ == "__main__":
    main()
