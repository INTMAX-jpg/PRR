from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dtype_from_name(name: str | None):
    if name is None or name == 'auto':
        return 'auto'
    name = str(name).lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16', 'half'}:
        return torch.float16
    if name in {'fp32', 'float32'}:
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')
