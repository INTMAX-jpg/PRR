#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from latent_pressure.config import load_config, save_json, set_seed
from latent_pressure.hf_model import first_param_device, load_tokenizer_and_model
from latent_pressure.numeric_answers import extract_final_number


TURN_NAMES = [
    "initial",
    "weak_pressure",
    "medium_pressure",
    "hard_pressure",
    "pressure_removed",
    "weak_repressure",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_source_line"] = line_number
                rows.append(row)
    return rows


def render_turn_prompts(cfg: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    prompt_cfg = cfg["prompt"]
    values = {
        "question": row["question"],
        "wrong_answer": row["wrong_answer"],
    }
    return [
        prompt_cfg["initial_user_template"].format(**values),
        prompt_cfg["weak_pressure_template"].format(**values),
        prompt_cfg["medium_pressure_template"].format(**values),
        prompt_cfg["hard_pressure_template"].format(**values),
        prompt_cfg["pressure_removed_template"].format(**values),
        prompt_cfg["weak_repressure_template"].format(**values),
    ]


def generate_response(tokenizer, model, messages, generation_cfg: Dict[str, Any]) -> Dict[str, Any]:
    chat_template_kwargs = {}
    if "enable_thinking" in generation_cfg:
        chat_template_kwargs["enable_thinking"] = bool(
            generation_cfg["enable_thinking"]
        )
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **chat_template_kwargs,
    )
    device = first_param_device(model)
    input_ids = input_ids.to(device)
    max_context_tokens = int(generation_cfg.get("max_context_tokens", 8192))
    if input_ids.shape[1] > max_context_tokens:
        raise RuntimeError(
            f"Conversation has {input_ids.shape[1]} input tokens, exceeding "
            f"max_context_tokens={max_context_tokens}; refusing silent truncation."
        )

    do_sample = bool(generation_cfg.get("do_sample", False))
    kwargs = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 256)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
        kwargs["top_p"] = float(generation_cfg.get("top_p", 0.95))

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            **kwargs,
        )
    generated_ids = output_ids[0, input_ids.shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "response": response,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(generated_ids.shape[0]),
        "hit_max_new_tokens": int(generated_ids.shape[0]) >= int(kwargs["max_new_tokens"]),
        "generation_seconds": time.time() - start,
    }


def read_completed_ids(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row.get("turns", [])) == len(TURN_NAMES):
                completed.add(str(row["base_id"]))
    return completed


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-17 free-form GSM8K PRR generation.")
    ap.add_argument("--config", default="configs/qwen15b_stage17_gsm8k_freeform.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["experiment"].get("seed", 1701))
    set_seed(seed)
    dataset_path = Path(args.dataset or cfg["data"]["dataset_path"])
    outdir = Path(args.outdir or cfg["experiment"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"generations_shard_{args.shard_index:02d}.jsonl"

    if args.shard_count <= 0 or not (0 <= args.shard_index < args.shard_count):
        raise ValueError("Require shard_count > 0 and 0 <= shard_index < shard_count")
    rows = load_jsonl(dataset_path)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    completed = set() if args.no_resume else read_completed_ids(output_path)
    rows_to_run = [row for row in rows if str(row["base_id"]) not in completed]

    tokenizer, model = load_tokenizer_and_model(cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    system_prompt = cfg["prompt"]["system_template"].strip()
    generation_cfg = cfg.get("generation", {})
    if not bool(generation_cfg.get("do_sample", False)):
        # Some checkpoints ship sampling defaults even for greedy generation.
        # Clear them to make the effective decoding configuration explicit.
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    started = time.time()
    mode = "w" if args.no_resume else "a"

    print(
        f"[stage17] shard={args.shard_index}/{args.shard_count} "
        f"assigned={len(rows)} completed={len(completed)} pending={len(rows_to_run)}",
        flush=True,
    )
    with output_path.open(mode, encoding="utf-8") as handle:
        for item_number, row in enumerate(rows_to_run, 1):
            prompts = render_turn_prompts(cfg, row)
            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            turns = []
            for turn_index, (turn_name, user_prompt) in enumerate(zip(TURN_NAMES, prompts)):
                messages.append({"role": "user", "content": user_prompt})
                generated = generate_response(tokenizer, model, messages, generation_cfg)
                response = generated.pop("response")
                parsed = extract_final_number(response)
                if generated["hit_max_new_tokens"] and not parsed.marker_found:
                    parsed_answer_online = None
                    parse_source_online = "truncated_no_marker"
                else:
                    parsed_answer_online = parsed.normalized
                    parse_source_online = parsed.source
                turns.append(
                    {
                        "turn_index": turn_index,
                        "turn_name": turn_name,
                        "user_prompt": user_prompt,
                        "response": response,
                        "parsed_answer_online": parsed_answer_online,
                        "parse_source_online": parse_source_online,
                        "marker_found_online": parsed.marker_found,
                        **generated,
                    }
                )
                messages.append({"role": "assistant", "content": response})

            result = {
                "base_id": row["base_id"],
                "dataset_index": row.get("dataset_index"),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "wrong_answer": row["wrong_answer"],
                "model_path": cfg["model"]["path"],
                "decoding": {
                    "do_sample": bool(generation_cfg.get("do_sample", False)),
                    "max_new_tokens": int(generation_cfg.get("max_new_tokens", 256)),
                },
                "shard_index": args.shard_index,
                "turns": turns,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            answers = ",".join(turn["parsed_answer_online"] or "NA" for turn in turns)
            print(
                f"[stage17] {item_number}/{len(rows_to_run)} {row['base_id']} "
                f"gold={row['gold_answer']} wrong={row['wrong_answer']} answers={answers}",
                flush=True,
            )

    manifest = {
        "stage": 17,
        "protocol": "GSM8K free-form multi-turn pressure/recovery/relapse",
        "config": args.config,
        "dataset": str(dataset_path),
        "model_path": cfg["model"]["path"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "n_assigned": len(rows),
        "n_previously_completed": len(completed),
        "n_newly_completed": len(rows_to_run),
        "output_path": str(output_path),
        "elapsed_seconds": time.time() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "generation": generation_cfg,
        "turn_names": TURN_NAMES,
    }
    save_json(manifest, outdir / f"manifest_shard_{args.shard_index:02d}.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
