#!/usr/bin/env python3
"""Render the supplied trajectory and causal statistics in Times New Roman."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',required=True,type=Path);ap.add_argument('--font-file',type=Path);a=ap.parse_args();root=Path(__file__).resolve().parents[1]
    if a.font_file:font_manager.fontManager.addfont(str(a.font_file))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    a.outdir.mkdir(parents=True,exist_ok=False)
    data=pd.read_csv(root/'results/figures/trajectory_plot_data.csv');fig,axes=plt.subplots(1,2,figsize=(6.8,2.25),sharey=True)
    for ax,task,title in zip(axes,['mmlu','arc'],['MMLU','ARC-Challenge']):
        ax.axvspan(3.8,4.2,color='#eeeeee',zorder=0)
        for series,label,color in [('output_gold_probability','Output','#2166ac'),('readout_gold_probability','Readout','#d66000')]:
            d=data[data.dataset.eq(task)&data.series.eq(series)].sort_values('turn');x=d.turn.to_numpy()
            ax.plot(x,d['mean'],marker='o',ms=3,color=color,label=label);ax.fill_between(x,d.ci_low,d.ci_high,color=color,alpha=.16,lw=0)
        ax.set(title=title,xlabel='Turn',ylim=(0,1.04),xticks=range(6),xticklabels=[f'T{i}' for i in range(6)]);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Correct-answer probability');axes[1].legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(a.outdir/'probability_trajectory.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.5));labels=['Qwen2.5-1.5B','Qwen2.5-7B','Qwen3-1.7B'];keys=['qwen15b','qwen7b','qwen3_17b'];effects=[];claims=[]
    modes=['clean_state_restoration','random_orthogonal','shuffled_cross_item']
    for key in keys:
        folder=root/'results/causal'/('gsm_'+key);effects.append(pd.read_csv(folder/'mode_effects.csv').set_index('mode').loc[modes,'readout_gap_delta'].to_numpy());claims.append(json.loads((folder/'claim_decision.json').read_text()))
    effects=np.array(effects);x=np.arange(3)
    for j,(label,color) in enumerate(zip(['Matched','Orthogonal','Shuffled'],['#2166ac','#aaaaaa','#d66000'])):axes[0].bar(x+(j-1)*.24,effects[:,j],.23,label=label,color=color)
    axes[0].set_ylabel('Change in readout gap');axes[0].set_xticks(x,labels,rotation=12);axes[0].legend(frameon=False,fontsize=8)
    rates=[c['output_bias_readout_wrong_rate'] for c in claims];axes[1].bar(x,rates,color='#2166ac',width=.6)
    axes[1].set(ylim=(0,1),ylabel='Readout still wrong-preferring');axes[1].set_xticks(x,[f'{l}\nN={c["n_output_bias_surface_repairs"]}' for l,c in zip(labels,claims)],rotation=12)
    for i,v in enumerate(rates):axes[1].text(i,v+.02,f'{100*v:.1f}%',ha='center',fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(a.outdir/'gsm_causal.pdf');plt.close(fig)
if __name__=='__main__':main()
