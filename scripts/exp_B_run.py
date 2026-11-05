"""Pressure-template behavior: sequential trajectories and matched-history T5 forks."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prr_behavior.config import load_config, set_seed
from prr_behavior.hf_model import choose_label, first_param_device, get_label_token_info, load_tokenizer_and_model, logits_for_labels
from prr_behavior.prompts import apply_chat_template, build_messages
from prr_behavior.utils import read_jsonl, stable_hash

def forward_one(tok, model, label_info, record: Dict[str, Any], turn_index: int, assistant_history: List[str], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    messages = build_messages(record, turn_index, assistant_history, cfg)
    prompt_text = apply_chat_template(tok, messages)
    enc = tok(prompt_text, return_tensors='pt')
    device = first_param_device(model)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    next_logits = logits[0, -1, :]
    pred_label = choose_label(next_logits, label_info)
    label_logits = logits_for_labels(next_logits, label_info)
    true_label = str(record['true_label'])
    wrong_label = str(record['wrong_label'])
    output_correct = pred_label == true_label
    output_wrong = pred_label == wrong_label
    row: Dict[str, Any] = {'pred_label': pred_label, 'output_correct': output_correct, 'output_wrong': output_wrong, 'output_other': not output_correct and (not output_wrong), 'pairwise_output_wrong': label_logits[true_label] < label_logits[wrong_label], 'logit_margin': float(label_logits[true_label] - label_logits[wrong_label]), 'prompt_hash': stable_hash(prompt_text), 'n_tokens': int(input_ids.shape[1])}
    for label, value in label_logits.items():
        row[f'logit_{label}'] = float(value)
    return (row, pred_label)

def common_row(record: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    turn = record['turns'][turn_index]
    message = str(turn['user_message'])
    return {'record_id': str(record['record_id']), 'base_id': str(record['base_id']), 'dataset': str(record.get('dataset', '')), 'split': str(record.get('split', '')), 'trajectory_type': str(record['trajectory_type']), 'pressure_family': str(record['pressure_family']), 'pressure_variant': str(record['pressure_variant']), 'template_cell': str(record['template_cell']), 'turn_index': int(turn_index), 'condition': str(turn.get('condition', '')), 'pressure_level': int(turn.get('pressure_level', 0)), 'true_label': str(record['true_label']), 'wrong_label': str(record['wrong_label']), 'item_code': str(record.get('item_code', '')), 'target_field': str(record.get('target_field', '')), 'target_value': str(record.get('target_value', '')), 'true_entity': str(record.get('true_entity', '')), 'wrong_entity': str(record.get('wrong_entity', '')), 'pressure_message': message, 'pressure_message_chars': len(message), 'pressure_message_words': len(message.split())}

def run_panel1(records: List[Dict[str, Any]], tok, model, label_info, cfg, handle, progress_every: int, only_turn0: bool=False) -> Tuple[int, int]:
    rows_written = 0
    forwards = 0
    t0_cache: Dict[str, Dict[str, Any]] = {}
    start = time.time()
    for record_index, record in enumerate(records, start=1):
        history: List[str] = []
        base_id = str(record['base_id'])
        turn_indices = [0] if only_turn0 else range(7)
        for turn_index in turn_indices:
            prefix = common_row(record, turn_index)
            if turn_index == 0 and base_id in t0_cache:
                evaluation = dict(t0_cache[base_id])
                pred_label = str(evaluation['pred_label'])
                evaluation['forward_reused'] = True
            else:
                evaluation, pred_label = forward_one(tok, model, label_info, record, turn_index, history, cfg)
                evaluation['forward_reused'] = False
                forwards += 1
                if turn_index == 0:
                    t0_cache[base_id] = dict(evaluation)
            prefix.update(evaluation)
            handle.write(json.dumps(prefix, ensure_ascii=False) + '\n')
            rows_written += 1
            history.append(pred_label)
        if record_index % progress_every == 0 or record_index == len(records):
            handle.flush()
            elapsed = time.time() - start
            print(f'[exp_B panel1] records={record_index}/{len(records)} rows={rows_written} forwards={forwards} elapsed={elapsed:.1f}s', flush=True)
    return (rows_written, forwards)

def run_panel2(records: List[Dict[str, Any]], tok, model, label_info, cfg, handle, progress_every: int) -> Tuple[int, int]:
    start = time.time()
    for record_index, record in enumerate(records, start=1):
        history = [str(x) for x in record['canonical_history_T0_T4']]
        if len(history) != 5:
            raise ValueError(f"Expected five canonical replies for {record['base_id']}")
        evaluation, _ = forward_one(tok, model, label_info, record, 5, history, cfg)
        evaluation['forward_reused'] = False
        row = common_row(record, 5)
        row['canonical_history_T0_T4'] = '|'.join(history)
        row.update(evaluation)
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        if record_index % progress_every == 0 or record_index == len(records):
            handle.flush()
            elapsed = time.time() - start
            print(f'[exp_B panel2] rows={record_index}/{len(records)} elapsed={elapsed:.1f}s', flush=True)
    return (len(records), len(records))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['panel1', 'panel2'], required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest-out', default='')
    ap.add_argument('--progress-every', type=int, default=25)
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--filter-family', default='')
    ap.add_argument('--filter-variant', default='')
    ap.add_argument('--only-turn0', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get('experiment', {}).get('seed', 42)))
    records = list(read_jsonl(args.dataset))
    if args.filter_family:
        records = [r for r in records if r.get('pressure_family') == args.filter_family]
    if args.filter_variant:
        records = [r for r in records if r.get('pressure_variant') == args.filter_variant]
    if args.max_records is not None:
        records = records[:args.max_records]
    if not records:
        raise ValueError('Empty dataset')
    tok, model = load_tokenizer_and_model(cfg)
    label_info = get_label_token_info(tok, list(cfg['experiment']['labels']))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with out.open('w', encoding='utf-8', buffering=1) as handle:
        if args.mode == 'panel1':
            n_rows, n_forwards = run_panel1(records, tok, model, label_info, cfg, handle, args.progress_every, args.only_turn0)
        else:
            n_rows, n_forwards = run_panel2(records, tok, model, label_info, cfg, handle, args.progress_every)
    manifest = {'experiment': 'B', 'mode': args.mode, 'config': args.config, 'dataset': args.dataset, 'n_input_records': len(records), 'n_rows': n_rows, 'n_model_forwards': n_forwards, 'output': str(out), 'elapsed_seconds': time.time() - start}
    manifest_out = Path(args.manifest_out) if args.manifest_out else out.with_suffix('.manifest.json')
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
