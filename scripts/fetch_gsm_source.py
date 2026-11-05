#!/usr/bin/env python3
"""Fetch the official GSM8K test set at an explicit repository revision."""
import argparse,hashlib,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--out',type=Path,default=Path('data/source/gsm8k_test_official.jsonl'));a=p.parse_args()
    from datasets import load_dataset
    d=load_dataset('openai/gsm8k','main',split='test',revision=a.revision);assert len(d)==1319
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:
        for r in d:f.write(json.dumps({'question':r['question'],'answer':r['answer']},ensure_ascii=False)+'\n')
    with a.out.with_suffix('.source.json').open('x',encoding='utf-8') as f:json.dump({'repository':'openai/gsm8k','revision':a.revision,'config':'main','split':'test','rows':len(d),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()},f,indent=2)
if __name__=='__main__':main()
