from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Dict, Any, List


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:n]


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_int_list(s: str | None) -> List[int] | None:
    if s is None or s == '':
        return None
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_str_list(s: str | None) -> List[str] | None:
    if s is None or s == '':
        return None
    return [x.strip() for x in s.split(',') if x.strip()]
