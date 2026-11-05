from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .config import dtype_from_name

@dataclass
class LabelTokenInfo:
    labels: List[str]
    token_ids: Dict[str, int]
    token_texts: Dict[str, str]

def _as_bool(x: Any, default: bool=False) -> bool:
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
    model_path = model_cfg['path']
    local_files_only = _as_bool(model_cfg.get('local_files_only', True), default=True)
    trust_remote_code = _as_bool(model_cfg.get('trust_remote_code', False), default=False)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only, use_fast=_as_bool(model_cfg.get('use_fast_tokenizer', True), default=True))
    tok._prr_chat_template_kwargs = dict(model_cfg.get('chat_template_kwargs') or {})
    tok._prr_strict_chat_template = _as_bool(model_cfg.get('strict_chat_template', False), default=False)
    dtype = dtype_from_name(model_cfg.get('dtype', 'auto'))
    kwargs = {'trust_remote_code': trust_remote_code, 'local_files_only': local_files_only}
    if dtype != 'auto':
        kwargs['torch_dtype'] = dtype
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
    if _as_bool(model_cfg.get('load_in_8bit', False)):
        kwargs['load_in_8bit'] = True
    if _as_bool(model_cfg.get('load_in_4bit', False)):
        kwargs['load_in_4bit'] = True
    loader_name = str(model_cfg.get('loader', 'auto')).strip().lower()
    model_type = str(getattr(hf_config, 'model_type', '')).lower()
    if loader_name == 'auto':
        loader_name = 'image_text_to_text' if model_type == 'gemma3' else 'causal_lm'
    if loader_name == 'causal_lm':
        loader = AutoModelForCausalLM
    elif loader_name in {'image_text_to_text', 'conditional_generation'}:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError('This model requires AutoModelForImageTextToText. Use the pinned Experiment-C transformers environment.') from exc
        loader = AutoModelForImageTextToText
    else:
        raise ValueError(f'Unsupported model.loader={loader_name!r}')
    printable_kwargs = {k: str(v) if k == 'torch_dtype' else v for k, v in kwargs.items()}
    print('[load_model]', {'model_path': model_path, 'model_type': model_type, 'loader': loader.__name__, 'chat_template_kwargs': tok._prr_chat_template_kwargs, 'from_pretrained_kwargs': printable_kwargs}, flush=True)
    model = loader.from_pretrained(model_path, **kwargs)
    if not device_map:
        device = model_cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    model.eval()
    return (tok, model)

def first_param_device(model) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return next(embeddings.parameters()).device
    except (AttributeError, StopIteration):
        pass
    return next(model.parameters()).device

def get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, 'hidden_size'):
        return int(cfg.hidden_size)
    text_cfg = getattr(cfg, 'text_config', None)
    if text_cfg is not None and hasattr(text_cfg, 'hidden_size'):
        return int(text_cfg.hidden_size)
    language_cfg = getattr(cfg, 'language_config', None)
    if language_cfg is not None and hasattr(language_cfg, 'hidden_size'):
        return int(language_cfg.hidden_size)
    raise ValueError('Could not determine language-model hidden size from config')

def get_decoder_layers(model):
    candidates = [model, getattr(model, 'model', None), getattr(model, 'language_model', None), getattr(getattr(model, 'model', None), 'language_model', None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, 'layers'):
            return candidate.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError('Could not find decoder layers. Expected a .layers collection on the model/language_model wrapper or model.transformer.h')

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
            raise ValueError(f'Could not find a single-token representation for label {lab!r}. Tried {candidates}; got {[tokenizer.encode(c, add_special_tokens=False) for c in candidates]}')
        token_texts[lab] = found[0]
        token_ids[lab] = int(found[1])
    return LabelTokenInfo(labels=labels, token_ids=token_ids, token_texts=token_texts)

def logits_for_labels(next_token_logits: torch.Tensor, label_info: LabelTokenInfo) -> Dict[str, float]:
    return {lab: float(next_token_logits[label_info.token_ids[lab]].detach().float().cpu()) for lab in label_info.labels}

def choose_label(next_token_logits: torch.Tensor, label_info: LabelTokenInfo) -> str:
    scores = logits_for_labels(next_token_logits, label_info)
    return max(scores.items(), key=lambda kv: kv[1])[0]

@contextmanager
def patch_activation(model, site: str, layer_idx: int, patch_vec: torch.Tensor, read_pos: int=-1):
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
