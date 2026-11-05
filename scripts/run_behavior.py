#!/usr/bin/env python3
"""Constrained-label PRR behavior, using the archived model and prompt helpers."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--outdir',required=True)
    a=ap.parse_args()
    from prr_behavior.config import load_config,set_seed
    from prr_behavior.hf_model import load_tokenizer_and_model,first_param_device,get_label_token_info,choose_label,logits_for_labels
    from prr_behavior.prompts import apply_chat_template,build_messages
    from prr_behavior.utils import stable_hash
    import torch
    cfg=load_config(a.config);set_seed(int(cfg['experiment'].get('seed',42)))
    src=Path(a.dataset);records=[json.loads(x) for x in src.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    n=len({r['base_id'] for r in records})
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=False)
    tok,model=load_tokenizer_and_model(cfg);device=first_param_device(model)
    info=get_label_token_info(tok,list(cfg['experiment']['labels']));history_mode=cfg['experiment'].get('history_mode','model')
    if history_mode not in ['model','oracle_true']:raise ValueError('Unsupported history mode')
    fields=['row_idx','record_id','base_id','dataset','split','trajectory_type','turn_index','condition','pressure_level','true_label','wrong_label','pred_label','output_correct','output_wrong','output_other','logit_margin','prompt_hash','n_tokens']+[f'logit_{x}' for x in info.labels]
    i=0
    save_prompts=bool(cfg['experiment'].get('save_rendered_prompts',False))
    with ExitStack() as stack:
        f=stack.enter_context((out/'metadata.csv').open('x',encoding='utf-8',newline=''))
        prompt_log=stack.enter_context((out/'rendered_prompts.jsonl').open('x',encoding='utf-8')) if save_prompts else None
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            history=[]
            for ti,turn in enumerate(r['turns']):
                messages=build_messages(r,ti,history,cfg)
                text=apply_chat_template(tok,messages)
                enc=tok(text,return_tensors='pt');ids=enc['input_ids'].to(device);mask=enc.get('attention_mask')
                if mask is not None:mask=mask.to(device)
                with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask).logits
                nxt=logits[0,-1,:];pred=choose_label(nxt,info);ll=logits_for_labels(nxt,info)
                row={k:r.get(k,'') for k in ['record_id','base_id','dataset','split','trajectory_type','true_label','wrong_label']}
                row.update(row_idx=i,turn_index=ti,condition=turn.get('condition',''),pressure_level=turn.get('pressure_level',0),pred_label=pred,output_correct=pred==r['true_label'],output_wrong=pred==r['wrong_label'],output_other=pred not in [r['true_label'],r['wrong_label']],logit_margin=ll[r['true_label']]-ll[r['wrong_label']],prompt_hash=stable_hash(text),n_tokens=int(ids.shape[1]))
                row.update({f'logit_{k}':v for k,v in ll.items()});writer.writerow(row)
                if prompt_log is not None:
                    prompt_record={k:row[k] for k in ['row_idx','record_id','base_id','trajectory_type','turn_index','prompt_hash']}
                    prompt_record.update(messages=messages,rendered_prompt=text)
                    prompt_log.write(json.dumps(prompt_record,ensure_ascii=False)+'\n')
                i+=1
                history.append(pred if history_mode=='model' else r['true_label'])
    manifest=dict(base_items=n,trajectory_records=len(records),rows=i,dataset_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),history_mode=history_mode)
    manifest.update(rendered_prompts_file='rendered_prompts.jsonl' if save_prompts else None,rendered_prompt_rows=i if save_prompts else 0)
    with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
