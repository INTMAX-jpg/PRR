#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from latent_pressure.config import ensure_dir, load_config, save_json, set_seed
from latent_pressure.hf_model import (
    ActivationTracer,
    choose_label,
    first_param_device,
    get_label_token_info,
    load_tokenizer_and_model,
    logits_for_labels,
)
from latent_pressure.prompts import apply_chat_template, build_messages
from latent_pressure.utils import read_jsonl, stable_hash


def main():
    ap = argparse.ArgumentParser(description='Run sequential multi-turn extraction and save logits + activations.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--max_records', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('experiment', {}).get('seed', 42)))
    outdir = ensure_dir(args.outdir or cfg['experiment']['output_dir'])
    act_dir = ensure_dir(outdir / 'activations')

    records = list(read_jsonl(args.dataset))
    max_records = args.max_records if args.max_records is not None else cfg['experiment'].get('max_records')
    if max_records:
        records = records[: int(max_records)]
    total_rows = sum(len(r['turns']) for r in records)
    if total_rows == 0:
        raise ValueError('Dataset has no turns')

    tok, model = load_tokenizer_and_model(cfg)
    device = first_param_device(model)
    label_info = get_label_token_info(tok, list(cfg['experiment']['labels']))
    hidden_size = int(model.config.hidden_size)
    sites = list(cfg['activations']['sites'])
    layers = [int(x) for x in cfg['activations']['layers']]
    dtype = np.float16 if cfg['activations'].get('dtype', 'float16') == 'float16' else np.float32
    read_pos = int(cfg['activations'].get('read_pos', -1))

    # Preallocate activation arrays as .npy memmaps.
    feature_paths = {}
    feature_arrays = {}
    for site in sites:
        for layer in layers:
            p = act_dir / f'{site}_L{layer}.npy'
            arr = np.lib.format.open_memmap(p, mode='w+', dtype=dtype, shape=(total_rows, hidden_size))
            feature_paths[f'{site}_L{layer}'] = str(p)
            feature_arrays[(site, layer)] = arr

    meta_rows = []
    row_idx = 0
    history_mode = cfg['experiment'].get('history_mode', 'model')

    with ActivationTracer(model, layers=layers, sites=sites, read_pos=read_pos) as tracer:
        for rec in tqdm(records, desc='records'):
            assistant_history = []
            for turn_idx, turn in enumerate(rec['turns']):
                tracer.clear()
                messages = build_messages(rec, turn_idx, assistant_history, cfg)
                prompt_text = apply_chat_template(tok, messages)
                enc = tok(prompt_text, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.inference_mode():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                next_logits = logits[0, -1, :]
                label_logits = logits_for_labels(next_logits, label_info)
                pred_label = choose_label(next_logits, label_info)
                true_label = rec['true_label']
                wrong_label = rec['wrong_label']
                logit_margin = label_logits[true_label] - label_logits[wrong_label]

                for key, arr in feature_arrays.items():
                    if key not in tracer.cache:
                        raise RuntimeError(f'Missing activation {key}; cache keys={list(tracer.cache.keys())[:5]}...')
                    arr[row_idx, :] = tracer.cache[key][0].numpy().astype(dtype, copy=False)

                row = {
                    'row_idx': row_idx,
                    'record_id': rec['record_id'],
                    'base_id': rec['base_id'],
                    'dataset': rec.get('dataset', ''),
                    'split': rec.get('split', ''),
                    'trajectory_type': rec['trajectory_type'],
                    'turn_index': int(turn_idx),
                    'condition': turn.get('condition', ''),
                    'pressure_level': turn.get('pressure_level', 0),
                    'control_family': turn.get('control_family', ''),
                    'true_label': true_label,
                    'wrong_label': wrong_label,
                    'pred_label': pred_label,
                    'logit_margin': float(logit_margin),
                    'prompt_hash': stable_hash(prompt_text),
                    'n_tokens': int(input_ids.shape[1]),
                    'item_code': rec.get('item_code', ''),
                    'item_mode': rec.get('item_mode', ''),
                    'target_field': rec.get('target_field', ''),
                    'target_value': rec.get('target_value', ''),
                    'true_entity': rec.get('true_entity', ''),
                    'wrong_entity': rec.get('wrong_entity', ''),
                }
                for lab, val in label_logits.items():
                    row[f'logit_{lab}'] = val
                meta_rows.append(row)

                if history_mode == 'model':
                    assistant_history.append(pred_label)
                elif history_mode == 'oracle_true':
                    assistant_history.append(true_label)
                else:
                    raise ValueError(f'Unsupported history_mode: {history_mode}')
                row_idx += 1

    # Flush memmaps.
    for arr in feature_arrays.values():
        arr.flush()

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(outdir / 'metadata.csv', index=False)
    manifest = {
        'config_path': str(args.config),
        'dataset_path': str(args.dataset),
        'n_records': len(records),
        'n_rows': int(total_rows),
        'hidden_size': hidden_size,
        'sites': sites,
        'layers': layers,
        'feature_paths': feature_paths,
        'label_token_ids': label_info.token_ids,
        'label_token_texts': label_info.token_texts,
        'history_mode': history_mode,
    }
    save_json(manifest, outdir / 'manifest.json')
    print(f'Saved metadata to {outdir / "metadata.csv"}')
    print(f'Saved activations to {act_dir}')
    print(json.dumps({k: manifest[k] for k in ['n_records','n_rows','hidden_size','sites','layers']}, indent=2))


if __name__ == '__main__':
    main()
