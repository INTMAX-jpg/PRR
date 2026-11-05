"""Original pressure-fork surface statistics, with shared paired-item draws."""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
from exp_B_templates import FAMILY_ORDER
PANEL2_METRICS = ['T5_output_wrong_rate', 'T5_logit_margin_mean']
def ci(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (float('nan'), float('nan'))
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))

def panel2_point(group: pd.DataFrame) -> Dict[str, float]:
    return {'N': len(group), 'T5_output_wrong_rate': float(group['output_wrong'].astype(bool).mean()), 'T5_logit_margin_mean': float(group['logit_margin'].mean()), 'prompt_tokens_mean': float(group['n_tokens'].mean()), 'pressure_words_mean': float(group['pressure_message_words'].mean())}

def panel2_arrays(group: pd.DataFrame, base_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    indexed = group.set_index('base_id').reindex(base_ids)
    if indexed.isna().any(axis=None):
        raise ValueError('Panel 2 cell is missing one or more paired base IDs')
    return {'T5_output_wrong_rate': indexed['output_wrong'].astype(bool).to_numpy(dtype=float), 'T5_logit_margin_mean': indexed['logit_margin'].to_numpy(dtype=float)}

def add_ci_rows(point_rows: pd.DataFrame, boot: Mapping[str, np.ndarray], id_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cell_index, point in point_rows.reset_index(drop=True).iterrows():
        ids = {col: point[col] for col in id_cols}
        for metric in metric_cols:
            low, high = ci(boot[metric][:, cell_index])
            rows.append({**ids, 'metric': metric, 'estimate': point[metric], 'ci95_low': low, 'ci95_high': high})
    return pd.DataFrame(rows)

def family_summary(cell_points: pd.DataFrame, cell_boot: Mapping[str, np.ndarray], cell_order: Sequence[Tuple[str, str]], metrics: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows = []
    family_boot: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    for family in FAMILY_ORDER:
        indices = [i for i, (fam, _) in enumerate(cell_order) if fam == family]
        sub = cell_points[cell_points['pressure_family'].eq(family)]
        row: Dict[str, Any] = {'pressure_family': family, 'n_variants': len(indices)}
        for metric in metrics:
            values = sub[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values))
            row[f'{metric}__variant_sd'] = float(np.nanstd(values, ddof=1))
            row[f'{metric}__variant_range'] = float(np.nanmax(values) - np.nanmin(values))
            boots = np.nanmean(cell_boot[metric][:, indices], axis=1)
            low, high = ci(boots)
            row[f'{metric}__ci95_low'] = low
            row[f'{metric}__ci95_high'] = high
            family_boot[metric].append(boots)
        rows.append(row)
    stacked = {metric: np.stack(values, axis=1) for metric, values in family_boot.items()}
    return (pd.DataFrame(rows), stacked)

def paired_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        return float('nan')
    return float(min(1.0, 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0))))

def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()

def panel2_effects(family_points: pd.DataFrame, family_boot: Mapping[str, np.ndarray], neutral_points: pd.Series, neutral_boot: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    effect_metrics = ['T5_output_wrong_rate']
    direct_idx = FAMILY_ORDER.index('direct_assertion')
    for metric in effect_metrics:
        for family_idx, family in enumerate(FAMILY_ORDER):
            diff = family_boot[metric][:, family_idx] - neutral_boot[metric]
            low, high = ci(diff)
            rows.append({'metric': metric, 'comparison_type': 'family_minus_neutral', 'family_a': family, 'family_b': 'neutral_control', 'estimate': float(family_points.loc[family_points['pressure_family'].eq(family), metric].iloc[0] - neutral_points[metric]), 'ci95_low': low, 'ci95_high': high, 'bootstrap_p_raw': paired_p(diff)})
        for family_idx, family in enumerate(FAMILY_ORDER):
            if family == 'direct_assertion':
                continue
            diff = family_boot[metric][:, family_idx] - family_boot[metric][:, direct_idx]
            low, high = ci(diff)
            rows.append({'metric': metric, 'comparison_type': 'family_minus_direct', 'family_a': family, 'family_b': 'direct_assertion', 'estimate': float(family_points.loc[family_points['pressure_family'].eq(family), metric].iloc[0] - family_points.loc[family_points['pressure_family'].eq('direct_assertion'), metric].iloc[0]), 'ci95_low': low, 'ci95_high': high, 'bootstrap_p_raw': paired_p(diff)})
        for left in range(len(FAMILY_ORDER)):
            for right in range(left + 1, len(FAMILY_ORDER)):
                diff = family_boot[metric][:, left] - family_boot[metric][:, right]
                low, high = ci(diff)
                rows.append({'metric': metric, 'comparison_type': 'all_family_pairs', 'family_a': FAMILY_ORDER[left], 'family_b': FAMILY_ORDER[right], 'estimate': float(family_points.loc[family_points['pressure_family'].eq(FAMILY_ORDER[left]), metric].iloc[0] - family_points.loc[family_points['pressure_family'].eq(FAMILY_ORDER[right]), metric].iloc[0]), 'ci95_low': low, 'ci95_high': high, 'bootstrap_p_raw': paired_p(diff)})
    frame = pd.DataFrame(rows)
    frame['bootstrap_p_holm'] = np.nan
    for metric, indices in frame.groupby('metric').groups.items():
        frame.loc[indices, 'bootstrap_p_holm'] = holm_adjust(frame.loc[indices, 'bootstrap_p_raw'].tolist())
    return frame

def analyze_panel2_cohort(frame: pd.DataFrame, cohort: str, n_boot: int, seed: int, preceding_panel1_items: int=0) -> Dict[str, Any] | None:
    base_ids = sorted(frame['base_id'].unique())
    if not base_ids:
        return None
    pressure = frame[frame['pressure_family'].ne('neutral_control')].copy()
    cells = sorted({(str(row.pressure_family), str(row.pressure_variant)) for row in pressure[['pressure_family', 'pressure_variant']].itertuples(index=False)}, key=lambda value: (FAMILY_ORDER.index(value[0]), value[1]))
    rng = np.random.default_rng(seed)
    if preceding_panel1_items:
        rng.integers(0, preceding_panel1_items, size=(n_boot, preceding_panel1_items), endpoint=False)
    sample_idx = rng.integers(0, len(base_ids), size=(n_boot, len(base_ids)), endpoint=False)
    point_rows = []
    boot_lists: Dict[str, List[np.ndarray]] = {metric: [] for metric in PANEL2_METRICS}
    for family, variant in cells:
        group = pressure[pressure['pressure_family'].eq(family) & pressure['pressure_variant'].eq(variant)]
        point = panel2_point(group)
        point.update({'cohort': cohort, 'pressure_family': family, 'pressure_variant': variant})
        point_rows.append(point)
        arrays = panel2_arrays(group, base_ids)
        for metric in PANEL2_METRICS:
            boot_lists[metric].append(arrays[metric][sample_idx].mean(axis=1))
    points = pd.DataFrame(point_rows)
    boot = {metric: np.stack(values, axis=1) for metric, values in boot_lists.items()}
    cell_ci = add_ci_rows(points, boot, ['cohort', 'pressure_family', 'pressure_variant'], PANEL2_METRICS)
    family, family_boot = family_summary(points, boot, cells, PANEL2_METRICS)
    family.insert(0, 'cohort', cohort)
    neutral_group = frame[frame['pressure_family'].eq('neutral_control')]
    neutral_point = pd.Series(panel2_point(neutral_group))
    neutral_arrays = panel2_arrays(neutral_group, base_ids)
    neutral_boot = {metric: neutral_arrays[metric][sample_idx].mean(axis=1) for metric in PANEL2_METRICS}
    neutral_rows = []
    for metric in PANEL2_METRICS:
        low, high = ci(neutral_boot[metric])
        neutral_rows.append({'cohort': cohort, 'pressure_family': 'neutral_control', 'pressure_variant': 'length_matched_reconsideration', 'metric': metric, 'estimate': neutral_point[metric], 'ci95_low': low, 'ci95_high': high})
    effects = panel2_effects(family, family_boot, neutral_point, neutral_boot)
    effects.insert(0, 'cohort', cohort)
    return {'n_base': len(base_ids), 'points': points, 'cell_ci': cell_ci, 'family': family, 'neutral': pd.DataFrame(neutral_rows), 'effects': effects, 'neutral_point': neutral_point}
