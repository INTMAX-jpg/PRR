#!/usr/bin/env python3
"""Original weak-only history comparison, retaining surface behavior only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def surface_rows(path):
    rows = pd.read_csv(path)
    rows['base_id'] = rows.base_id.astype(str)
    rows['output_correct'] = rows.pred_label.eq(rows.true_label)
    rows['output_wrong'] = rows.pred_label.eq(rows.wrong_label)
    return rows


def bootstrap_pair_effect(per_pair, cols, n_boot, seed):
    """Original Stage-16 resampling: keep multiplicities of sampled pair means."""
    rng = np.random.default_rng(seed)
    arr = per_pair[cols].to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return pd.DataFrame()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = arr[idx].mean(axis=1)
    rows = []
    for j, c in enumerate(cols):
        rows.append({'metric': c, 'mean_boot': float(np.nanmean(boot[:, j])),
                     'ci95_low': float(np.nanquantile(boot[:, j], .025)),
                     'ci95_high': float(np.nanquantile(boot[:, j], .975)),
                     'n_boot': n_boot, 'n_pairs': n})
    return pd.DataFrame(rows)


def analyze(prr, weak, n_boot=10000, seed=123):
    wp = prr[prr.trajectory_type.eq('wrong_pressure')]
    wc = weak[weak.trajectory_type.eq('weak_only_history')]
    initial = set(wp.loc[wp.turn_index.eq(0) & wp.output_correct, 'base_id'])
    flipped = set(wp.loc[wp.turn_index.eq(3) & wp.output_wrong, 'base_id'])
    recovered = set(wp.loc[wp.turn_index.eq(4) & wp.output_correct, 'base_id'])
    prr_ids = initial & flipped & recovered
    ids = prr_ids & set(wc.loc[wc.turn_index.eq(0) & wc.output_correct, 'base_id'])
    a = wp[wp.turn_index.eq(5) & wp.base_id.isin(ids)].set_index('base_id')
    b = wc[wc.turn_index.eq(5) & wc.base_id.isin(ids)].set_index('base_id')
    common = sorted(set(a.index) & set(b.index))
    a, b = a.loc[common], b.loc[common]
    paired = a[['true_label', 'wrong_label']].copy()
    paired['prr_t5_output_wrong'] = a.output_wrong.astype(float)
    paired['weak_t5_output_wrong'] = b.output_wrong.astype(float)
    paired['history_effect_output'] = paired.prr_t5_output_wrong - paired.weak_t5_output_wrong
    paired['prr_t5_output_correct'] = a.output_correct.astype(float)
    paired['weak_t5_output_correct'] = b.output_correct.astype(float)
    if 'logit_margin' in a and 'logit_margin' in b:
        paired['history_effect_logit_margin'] = b.logit_margin - a.logit_margin
    metrics = list(paired.select_dtypes(include='number').columns)
    per_pair = paired.groupby(['true_label', 'wrong_label'])[metrics].mean().reset_index()
    per_pair['n'] = paired.groupby(['true_label', 'wrong_label']).size().to_numpy()
    summary = pd.DataFrame([
        {'weighting': 'item', 'n_base': len(paired), 'n_pairs': len(per_pair), **paired[metrics].mean().to_dict()},
        {'weighting': 'equal_directed_pair', 'n_base': len(paired), 'n_pairs': len(per_pair), **per_pair[metrics].mean().to_dict()},
    ])
    cols = [x for x in ['history_effect_output', 'history_effect_logit_margin'] if x in metrics]
    ci = bootstrap_pair_effect(per_pair, cols, n_boot, seed)
    turns = []
    for name, frame in [('prr_history', wp), ('weak_only', wc)]:
        for turn, group in frame[frame.base_id.isin(ids)].groupby('turn_index'):
            pair_rates = group.groupby(['true_label', 'wrong_label'])[['output_wrong', 'output_correct']].mean()
            turns.append({'trajectory': name, 'turn': int(turn), 'n': len(group),
                          'wrong_item': float(group.output_wrong.mean()),
                          'correct_item': float(group.output_correct.mean()),
                          'wrong_equal_pair': float(pair_rates.output_wrong.mean()),
                          'correct_equal_pair': float(pair_rates.output_correct.mean())})
    return {'paired_rows': paired.reset_index(), 'per_pair': per_pair, 'summary': summary,
            'pair_bootstrap_ci': ci, 'turn_summary': pd.DataFrame(turns)}, {
        'prr_recovered': len(prr_ids), 'paired_after_weak_T0_correct': len(paired),
        'bootstrap': n_boot, 'seed': seed,
        'comparison': 'PRR-history T5 minus weak-only T5 on the same base items',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prr-metadata', required=True)
    p.add_argument('--weak-metadata', required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    tables, metadata = analyze(surface_rows(a.prr_metadata), surface_rows(a.weak_metadata), a.bootstrap, a.seed)
    a.outdir.mkdir(parents=True, exist_ok=False)
    for key, frame in tables.items():
        frame.to_csv(a.outdir / f'{key}.csv', index=False)
    (a.outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(tables['summary'].to_string(index=False))


if __name__ == '__main__':
    main()
