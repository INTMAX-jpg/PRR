from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import dtype_from_name


@dataclass
class LabelTokenInfo:
    labels: List[str]
    token_ids: Dict[str, int]
    token_texts: Dict[str, str]


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _parse_max_memory(x: Any):
    """Parse optional max_memory config for very large models.

    Accepts either a YAML dict, or a comma-separated string such as:
        "0:22GiB,1:22GiB,2:22GiB,3:22GiB,cpu:180GiB"
    The returned object is passed directly to transformers.from_pretrained.
    """
    if x in (None, '', 'none', 'None', 'null'):
        return None
    if isinstance(x, dict):
        # JSON object keys are strings; Accelerate expects integer GPU IDs.
        return {int(k) if isinstance(k, str) and k.isdecimal() else k: v
                for k, v in x.items()}
    out = {}
    for part in str(x).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f'Invalid max_memory entry {part!r}; expected device:value')
        k, v = part.split(':', 1)
        k = k.strip()
        v = v.strip()
        try:
            key = int(k)
        except ValueError:
            key = k
        out[key] = v
    return out or None


def load_tokenizer_and_model(cfg: Dict[str, Any]):
    model_cfg = cfg['model']
    # Qwen3 itself does not need torchao.  This opt-out avoids importing an
    # incompatible optional torchao build from a shared environment.
    if _as_bool(model_cfg.get('disable_torchao', False)):
        import transformers.utils.import_utils as transformers_import_utils

        transformers_import_utils._torchao_available = False
    tok = AutoTokenizer.from_pretrained(
        model_cfg['path'],
        trust_remote_code=bool(model_cfg.get('trust_remote_code', True)),
    )
    dtype = dtype_from_name(model_cfg.get('dtype', 'auto'))
    kwargs = {
        'trust_remote_code': bool(model_cfg.get('trust_remote_code', True)),
    }
    if dtype != 'auto':
        kwargs['torch_dtype'] = dtype

    # Large-model options. These are no-ops for small models, but are useful for
    # Qwen2.5-72B / similar models that need accelerate device maps or CPU offload.
    device_map = model_cfg.get('device_map', None)
    if device_map:
        kwargs['device_map'] = device_map
    max_memory = _parse_max_memory(model_cfg.get('max_memory', None))
    if max_memory:
        kwargs['max_memory'] = max_memory
    offload_folder = model_cfg.get('offload_folder', None)
    if offload_folder not in (None, '', 'none', 'None', 'null'):
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        kwargs['offload_folder'] = str(offload_folder)
    if 'offload_state_dict' in model_cfg:
        kwargs['offload_state_dict'] = _as_bool(model_cfg.get('offload_state_dict'), default=True)
    if 'low_cpu_mem_usage' in model_cfg:
        kwargs['low_cpu_mem_usage'] = _as_bool(model_cfg.get('low_cpu_mem_usage'), default=True)
    attn_impl = model_cfg.get('attn_implementation', None)
    if attn_impl not in (None, '', 'none', 'None', 'null'):
        kwargs['attn_implementation'] = str(attn_impl)

    # Optional bitsandbytes quantization flags. These are useful when the local
    # model directory is not already quantized but the run should load in int8.
    # If the model is already GPTQ/AWQ/int8-serialized, leave these unset and
    # AutoModelForCausalLM will use the model's own config where supported.
    if _as_bool(model_cfg.get('load_in_8bit', False)):
        kwargs['load_in_8bit'] = True
    if _as_bool(model_cfg.get('load_in_4bit', False)):
        kwargs['load_in_4bit'] = True

    print('[load_model] from_pretrained kwargs:', {k: v for k, v in kwargs.items() if k != 'torch_dtype'}, flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_cfg['path'], **kwargs)
    if not device_map:
        device = model_cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    model.eval()
    return tok, model


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def get_decoder_layers(model):
    # Qwen/Llama/Mistral style.
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError('Could not find decoder layers. Expected model.model.layers or model.transformer.h')


def get_label_token_info(tokenizer, labels: List[str]) -> LabelTokenInfo:
    """Choose a single-token representation for each label.

    We try bare label first because prompts end with a generation prompt. If that fails,
    we try a leading-space form.
    """
    token_ids: Dict[str, int] = {}
    token_texts: Dict[str, str] = {}
    for lab in labels:
        candidates = [lab, ' ' + lab, '\n' + lab]
        found = None
        for c in candidates:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                found = (c, ids[0])
                break
        if found is None:
            raise ValueError(
                f'Could not find a single-token representation for label {lab!r}. '
                f'Tried {candidates}; got {[tokenizer.encode(c, add_special_tokens=False) for c in candidates]}'
            )
        token_texts[lab] = found[0]
        token_ids[lab] = int(found[1])
    return LabelTokenInfo(labels=labels, token_ids=token_ids, token_texts=token_texts)


def logits_for_labels(next_token_logits: torch.Tensor, label_info: LabelTokenInfo) -> Dict[str, float]:
    return {lab: float(next_token_logits[label_info.token_ids[lab]].detach().float().cpu()) for lab in label_info.labels}


def choose_label(next_token_logits: torch.Tensor, label_info: LabelTokenInfo) -> str:
    scores = logits_for_labels(next_token_logits, label_info)
    return max(scores.items(), key=lambda kv: kv[1])[0]


class ActivationTracer:
    """Capture selected last-token activations from a HF decoder-only model."""

    def __init__(self, model, layers: Iterable[int], sites: Iterable[str], read_pos: int = -1):
        self.model = model
        self.layers_idx = list(layers)
        self.sites = list(sites)
        self.read_pos = read_pos
        self.layers = get_decoder_layers(model)
        self.handles = []
        self.cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def clear(self):
        self.cache = {}

    def _slice(self, x: torch.Tensor) -> torch.Tensor:
        # Keep batch dimension; extraction code usually uses batch=1.
        return x[:, self.read_pos, :].detach().float().cpu()

    def _hook_resid_pre(self, layer_idx: int):
        def hook(module, args):
            if args and torch.is_tensor(args[0]):
                self.cache[('resid_pre', layer_idx)] = self._slice(args[0])
            return None
        return hook

    def _hook_forward(self, site: str, layer_idx: int):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out):
                self.cache[(site, layer_idx)] = self._slice(out)
            return None
        return hook

    def __enter__(self):
        for li in self.layers_idx:
            layer = self.layers[li]
            if 'resid_pre' in self.sites:
                self.handles.append(layer.register_forward_pre_hook(self._hook_resid_pre(li)))
            if 'attn_out' in self.sites:
                if not hasattr(layer, 'self_attn'):
                    raise ValueError('Layer has no self_attn module')
                self.handles.append(layer.self_attn.register_forward_hook(self._hook_forward('attn_out', li)))
            if 'mlp_out' in self.sites:
                if not hasattr(layer, 'mlp'):
                    raise ValueError('Layer has no mlp module')
                self.handles.append(layer.mlp.register_forward_hook(self._hook_forward('mlp_out', li)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []


@contextmanager
def patch_activation(model, site: str, layer_idx: int, patch_vec: torch.Tensor, read_pos: int = -1):
    """Patch one site/layer at the read position with patch_vec.

    patch_vec should be [hidden_size] on CPU or any device.
    """
    layers = get_decoder_layers(model)
    layer = layers[layer_idx]
    handles = []

    def patch_tensor(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        vec = patch_vec.to(device=y.device, dtype=y.dtype)
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        y[:, read_pos, :] = vec.expand(y.shape[0], -1)
        return y

    if site == 'resid_pre':
        def pre_hook(module, args):
            if not args or not torch.is_tensor(args[0]):
                return None
            new0 = patch_tensor(args[0])
            return (new0,) + tuple(args[1:])
        handles.append(layer.register_forward_pre_hook(pre_hook))
    elif site == 'attn_out':
        def fwd_hook(module, args, output):
            if isinstance(output, tuple):
                out0 = patch_tensor(output[0])
                return (out0,) + tuple(output[1:])
            return patch_tensor(output)
        handles.append(layer.self_attn.register_forward_hook(fwd_hook))
    elif site == 'mlp_out':
        def fwd_hook(module, args, output):
            return patch_tensor(output)
        handles.append(layer.mlp.register_forward_hook(fwd_hook))
    else:
        raise ValueError(f'Unsupported patch site: {site}')

    try:
        yield
    finally:
        for h in handles:
            h.remove()
