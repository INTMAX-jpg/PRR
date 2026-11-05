#!/usr/bin/env python3
"""Generate the frozen CW design using the archived pure-data implementation."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prr_behavior.data import generate_synthetic_records
TYPES=['wrong_pressure','correct_pressure','recency_matched','confidence_no_answer','neutral_control']
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n-items',type=int,default=2000);p.add_argument('--seed',type=int,default=42);p.add_argument('--verify-against')
    a=p.parse_args();rows=generate_synthetic_records(n_items=a.n_items,labels=list('ABCD'),trajectory_types=TYPES,seed=a.seed,item_mode='diverse_lookup')
    if a.verify_against:
        original=[json.loads(x) for x in Path(a.verify_against).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        if rows!=original:raise ValueError('Generated records differ in content or order')
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(dict(base_items=a.n_items,trajectory_records=len(rows),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),verified_content_and_order=bool(a.verify_against)),indent=2))
if __name__=='__main__':main()
