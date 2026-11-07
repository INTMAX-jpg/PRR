#!/usr/bin/env python3
"""Verify the supplied numerical artifacts. Does not run a language model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def verify_gsm_behavior(root):
    """Recompute the paper-reported GSM8K behavior counts and sorted-bootstrap intervals."""
    import random,sys
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    from analyze_gsm8k_freeform_prr_stage17 import COMPUTATION_VERSION,bootstrap_mean
    d=root/'results/behavior/gsm_qwen15b';rows=pd.read_csv(d/'scored_turns.csv')
    assert len(rows)==6*rows.base_id.nunique()==3000 and not rows.duplicated(['base_id','turn_index']).any()
    wt=rows.pivot(index='base_id',columns='turn_index',values='output_wrong_target').astype(bool)
    oc=rows.pivot(index='base_id',columns='turn_index',values='output_correct').astype(bool)
    init=sorted(oc.index[oc[0]]);hard=[i for i in init if wt.at[i,3]];surf=[i for i in hard if oc.at[i,4]]
    assert [len(init),len(hard),len(surf)]==[325,207,144]
    rng=random.Random(1701)
    boot={'PFR_out':bootstrap_mean([float(wt.at[i,3]) for i in init],n_boot=10000,rng=rng),
          'RSR_out':bootstrap_mean([float(oc.at[i,4]) for i in hard],n_boot=10000,rng=rng),
          'RV_out':bootstrap_mean([float(wt.at[i,5])-float(wt.at[i,1]) for i in surf],n_boot=10000,rng=rng)}
    ci=pd.read_csv(d/f'stage17_bootstrap_ci_{COMPUTATION_VERSION}.csv').set_index('metric')
    for metric,v in boot.items():
        assert int(ci.loc[metric,'n'])==v['n'] and np.allclose(ci.loc[metric,['point','ci_low','ci_high']].astype(float),[v['point'],v['ci_low'],v['ci_high']],atol=1e-12),metric
    return dict(check='gsm_behavior_intervals',counts=[len(init),len(hard),len(surf)],intervals={k:[v['ci_low'],v['ci_high']] for k,v in boot.items()},passed=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path);a=ap.parse_args();root=a.root;results=[]
    manifest=json.loads((root/'artifact_manifest.json').read_text(encoding='utf-8'))
    for rel,info in manifest['files'].items():
        b=(root/rel).read_bytes();assert hashlib.sha256(b).hexdigest()==info['sha256'] and len(b)==info['bytes'],rel
    results.append({'check':'file_integrity','files':len(manifest['files']),'passed':True})
    expected={'mmlu':(627,.9728867623604466,.9945079406838646),'arc':(241,.9875518672199171,.99417936950899),'gsm_qwen15b':(160,.85,.6128568716792026),'gsm_qwen7b':(160,.89375,.7100284170029678),'gsm_qwen3_17b':(160,.8,.5909400508596118)}
    for key,(n,agree,r2) in expected.items():
        d=root/'results/readout'/key
        if key in ['mmlu','arc']:
            rows=pd.read_csv(d/'clean_predictions.csv.gz').query("split=='test'")
            target=rows[[f'target_option_probability_{k}' for k in 'ABCD']].to_numpy();pred=rows[[f'readout_option_probability_{k}' for k in 'ABCD']].to_numpy()
        else:
            rows=pd.read_csv(d/'diagnostic_predictions.csv.gz').query("experiment_split=='readout_test'")
            target=rows[['probability_X','probability_Y']].to_numpy();pred=rows[['readout_probability_X','readout_probability_Y']].to_numpy()
        observed_agree=float((target.argmax(1)==pred.argmax(1)).mean());observed_r2=float(1-np.square(target-pred).sum()/np.square(target-target.mean()).sum())
        assert len(rows)==rows.base_id.nunique()==n
        assert np.isclose(observed_agree,agree,atol=1e-12) and np.isclose(observed_r2,r2,atol=1e-12)
        summary=pd.read_csv(d/'heldout_metrics.csv').query("split=='test'").iloc[0]
        assert np.isclose(observed_agree,summary.preference_agreement) and np.isclose(observed_r2,summary.option_probability_r2)
        results.append(dict(check='heldout_fidelity',task_model=key,n=n,agreement=observed_agree,r2=observed_r2,passed=True))
    for task,n,strict,g in [('mmlu',225,24,.033941064010793644),('arc',135,14,.03924580802441652)]:
        d=root/'results/readout'/task;it=pd.read_csv(d/'natural_items.csv.gz');allrows=pd.read_csv(d/'all_turn_scores.csv.gz')
        gap=(it.T0_readout_gold_probability-it.T4_readout_gold_probability)-(it.T0_output_gold_probability-it.T4_output_gold_probability)
        assert len(it)==it.base_id.nunique()==n and int(it.T4_readout_pairwise_wrong_aligned.sum())==strict and np.isclose(gap.mean(),g)
        tr=allrows[allrows.base_id.isin(it.base_id)&allrows.trajectory_type.eq('wrong_pressure')&allrows.turn_index.between(0,5)]
        assert len(tr)==6*n and not tr.duplicated(['base_id','turn_index']).any()
        plot=pd.read_csv(root/'results/figures/trajectory_plot_data.csv');plot=plot[plot.dataset.eq(task)]
        for col in ['output_gold_probability','readout_gold_probability']:
            assert np.allclose(tr.groupby('turn_index')[col].mean().to_numpy(),plot[plot.series.eq(col)].sort_values('turn')['mean'].to_numpy(),atol=1e-12)
        results.append(dict(check='natural_T4_and_trajectory',task=task,n=n,strict_cases=strict,mean_gap=float(gap.mean()),passed=True))
    for key,nfixed,wrongrate in [('qwen15b',340,180/340),('qwen7b',211,140/211),('qwen3_17b',153,107/153)]:
        folder=root/'results/causal'/('gsm_'+key);rows=pd.read_csv(folder/'causal_rows.csv.gz');assert not rows.duplicated(['base_id','mode']).any()
        read=rows.pivot(index='base_id',columns='mode',values='readout_gold_vs_wrong_gap');oc=rows.pivot(index='base_id',columns='mode',values='output_gold_preferred').astype(bool);rc=rows.pivot(index='base_id',columns='mode',values='readout_gold_preferred').astype(bool)
        assert len(read)==500 and not read.isna().any().any()
        effects=pd.read_csv(folder/'mode_effects.csv').set_index('mode');vals={}
        for mode in ['clean_state_restoration','random_orthogonal','shuffled_cross_item']:
            val=float((read[mode]-read.target_baseline).mean());assert np.isclose(val,effects.loc[mode,'readout_gap_delta'],atol=1e-12);vals[mode]=val
        repaired=(~oc.target_baseline)&oc.output_logit_bias;rate=float((~rc.loc[repaired,'output_logit_bias']).mean())
        assert int(repaired.sum())==nfixed and np.isclose(rate,wrongrate,atol=1e-12)
        assert np.array_equal(read.target_baseline.to_numpy(),read.output_logit_bias.to_numpy())
        results.append(dict(check='gsm_paired_causal',model=key,n=500,effects=vals,corrected_outputs=nfixed,remaining_wrong_preference=rate,passed=True))
    exp={'qwen15b':[.855,.785,.7816666666666666,.10166666666666667,.565],'qwen7b':[.7493438320209974,1.,.7992125984251969,.2073490813648294,.6404199475065617]}
    families=['direct_assertion','authority_backed','evidence_containing','soft_uncertainty','social_consensus']
    for key,values in exp.items():
        d=root/'results/pressure'/key;s=pd.read_csv(d/'family_metrics.csv').set_index('pressure_family');assert np.allclose(s.loc[families,'T5_output_wrong_rate'],values)
        if key=='qwen15b':
            rows=pd.concat([pd.read_csv(f) for f in sorted(d.glob('rows_*.csv.gz'))],ignore_index=True)
            assert rows.base_id.nunique()==200 and not rows.duplicated(['base_id','pressure_family','pressure_variant']).any()
            obs=rows.groupby(['pressure_family','pressure_variant']).output_wrong.mean().groupby('pressure_family').mean()
            assert np.allclose(obs.loc[families],values)
        results.append(dict(check='pressure_family_estimates',model=key,passed=True))
    for key,counts in [('qwen3_0_6b',[644,473,0,0]),('qwen3_1_7b',[978,406,39,5]),('qwen3_8b',[999,0,0,0]),('gemma3_12b_it',[1000,0,0,0])]:
        d=pd.read_csv(root/f'results/newer_models/{key}/population_membership.csv');assert len(d)==d.base_id.nunique()==1000
        frozen=[json.loads(x) for x in (root/'data/newer_models/cw_1000_wrong_pressure.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        labels={r['base_id']:(r['true_label'],r['wrong_label']) for r in frozen}
        true=d.base_id.map(lambda x:labels[x][0]);wrong=d.base_id.map(lambda x:labels[x][1])
        ini=d.T0_pred_label.eq(true);flip=ini&d.T3_pred_label.eq(wrong);rec=flip&d.T4_pred_label.eq(true);relapse=rec&d.T5_pred_label.eq(wrong)
        got=[int(x.sum()) for x in [ini,flip,rec,relapse]];assert got==counts,(key,got)
        results.append(dict(check='newer_model_filtering',model=key,counts=got,passed=True))
    results.append(verify_gsm_behavior(root))
    report={'status':'passed','scope':'integrity and CPU recomputation of the supplied results','checks':results}
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
