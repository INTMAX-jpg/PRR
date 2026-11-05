from __future__ import annotations

from typing import Dict, List, Any, Sequence


def options_block(options: Sequence[Dict[str, str]]) -> str:
    return '\n'.join([f"{o['label']}. {o['text']}" for o in options])


def format_turn_message(turn: Dict[str, Any], true_label: str, wrong_label: str) -> str:
    msg = turn['user_message']
    return msg.format(true_label=true_label, wrong_label=wrong_label)


def build_messages(
    record: Dict[str, Any],
    upto_turn: int,
    assistant_history: Sequence[str],
    cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build chat messages up to current user turn.

    assistant_history contains one assistant letter for each previous turn.
    Stage-2 supports item-diverse fields: fact_context, source_question, item_code.
    """
    labels = cfg['experiment']['labels']
    p_cfg = cfg['prompt']
    true_label = record['true_label']
    wrong_label = record['wrong_label']
    fmt = {
        'true_label': true_label,
        'wrong_label': wrong_label,
        'label_list': ', '.join(labels),
        'options_block': options_block(record['options']),
        'source_question': record.get('source_question', 'Which option is correct?'),
        'fact_context': record.get('fact_context', ''),
        'item_code': record.get('item_code', record.get('base_id', '')),
        'target_field': record.get('target_field', ''),
        'target_value': record.get('target_value', ''),
    }
    sys = p_cfg['system_template'].format(**fmt)
    messages: List[Dict[str, str]] = [{'role': 'system', 'content': sys.strip()}]

    for i in range(upto_turn + 1):
        turn = record['turns'][i]
        if i == 0:
            base_question = p_cfg['base_question_template'].format(**fmt)
            content = base_question.strip()
            extra = format_turn_message(turn, true_label, wrong_label).strip()
            if extra:
                content = content + '\n\n' + extra
        else:
            content = format_turn_message(turn, true_label, wrong_label).strip()
        messages.append({'role': 'user', 'content': content})
        if i < upto_turn:
            label = assistant_history[i]
            reply = p_cfg.get('assistant_reply_template', '{label}').format(label=label)
            messages.append({'role': 'assistant', 'content': reply})
    return messages


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    parts = []
    for m in messages:
        parts.append(f"{m['role'].capitalize()}: {m['content']}")
    parts.append('Assistant:')
    return '\n'.join(parts)
