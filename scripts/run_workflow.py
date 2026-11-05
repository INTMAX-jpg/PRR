#!/usr/bin/env python3
"""Print or execute a documented experiment workflow with a new output directory."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
def main():
    root=Path(__file__).resolve().parents[1];flows=json.loads((root/'configs/workflows.json').read_text(encoding='utf-8'))
    p=argparse.ArgumentParser();p.add_argument('workflow',choices=sorted(flows));p.add_argument('--workdir',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    work=Path(a.workdir).resolve();commands=[[x.replace('{work}',str(work)) for x in cmd] for cmd in flows[a.workflow]]
    for cmd in commands:print(shlex.join([sys.executable]+cmd),flush=True)
    if a.execute:
        work.mkdir(parents=True,exist_ok=False)
        for cmd in commands:subprocess.run([sys.executable]+cmd,cwd=root,check=True)
if __name__=='__main__':main()
