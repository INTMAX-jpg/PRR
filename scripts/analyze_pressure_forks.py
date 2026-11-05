#!/usr/bin/env python3
"""Surface-only pressure-family summaries with explicit cohort selection."""
import argparse
import glob
import json
from pathlib import Path
import pandas as pd
from pressure_surface_helpers import analyze_panel2_cohort


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows-glob', required=True)
    cohort = p.add_mutually_exclusive_group(required=True)
    cohort.add_argument('--population-csv', type=Path,
                        help='Filter input using the observed cohort_surface_recovered column.')
    cohort.add_argument('--input-already-recovered', action='store_true',
                        help='Declare that input contains only previously selected surface-recovered items.')
    p.add_argument('--outdir', required=True, type=Path)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=7508,
                   help='Final Q7 cohort: 2508 + 5*1000. Q1.5: seed 2508 and --preceding-panel1-items 500.')
    p.add_argument('--preceding-panel1-items', type=int, default=0)
    a = p.parse_args()
    parts = []
    for name in sorted(glob.glob(a.rows_glob)):
        if '.csv' in Path(name).name:
            parts.append(pd.read_csv(name))
        else:
            parts.append(pd.read_json(name, lines=True))
    if not parts:
        p.error('No files match --rows-glob')
    rows = pd.concat(parts, ignore_index=True)
    rows['base_id'] = rows.base_id.astype(str)
    rows['output_wrong'] = rows.pred_label.eq(rows.wrong_label)
    input_n = int(rows.base_id.nunique())
    if a.population_csv:
        pop = pd.read_csv(a.population_csv)
        flag = pop.cohort_surface_recovered.astype(str).str.lower().isin(['true', '1'])
        rows = rows[rows.base_id.isin(pop.loc[flag, 'base_id'].astype(str))]
    metadata = {
        'input_cohort': 'requires_population_filter' if a.population_csv else 'already_surface_recovered',
        'selection_source': str(a.population_csv) if a.population_csv else 'caller_declared_preselected_input',
        'input_base_items': input_n,
        'n_base': int(rows.base_id.nunique()),
        'bootstrap': a.bootstrap, 'seed': a.seed,
        'preceding_panel1_items': a.preceding_panel1_items,
    }
    result = analyze_panel2_cohort(rows, 'surface_recovered', a.bootstrap, a.seed, a.preceding_panel1_items)
    a.outdir.mkdir(parents=True, exist_ok=False)
    if result is None:
        metadata['RV'] = None
        with (a.outdir / 'summary.json').open('x', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print('Empty recovered cohort: relapse quantities are undefined.')
        return
    for key in ['points', 'cell_ci', 'family', 'neutral', 'effects']:
        result[key].to_csv(a.outdir / f'{key}.csv', index=False, mode='x')
    with (a.outdir / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(result['family'][['pressure_family', 'T5_output_wrong_rate']].to_string(index=False))


if __name__ == '__main__':
    main()
