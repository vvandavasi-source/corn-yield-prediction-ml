#!/usr/bin/env python
"""Corn yield V5: self-contained 50-seed model and comparison runner.

Run: python corn_yield_50_seed_all_in_one.py
Defaults: D:\\corn_yield\\model_inputs -> D:\\corn_yield\\yield_model_results\\v5_50_seed_all_in_one
CLEAN_REBUILT = OUR ICDLs. STUDY = PUBLISHED ICDLs.
Embeds CY2 V5.2 INPUT_VALIDATION_AND_ANOMALY_METRICS_PLOT_FIX unchanged.
Requires the existing corn-yield environment and original input CSVs, not other scripts.
Runs seeds 1..50 sequentially, then generates mean/SD metrics, county counts,
paired seed tests, and state-clustered bootstrap comparisons of the 50-seed
ENSEMBLE predictions. Ensemble statistics differ from mean single-run metrics.
Seed tests measure training randomness conditional on these same data.
County bootstrap resamples whole states; with only 12 states CIs are approximate.
Legacy county-level Wilcoxon and HLN-DM diagnostics are also reported, but are
not used as spatially robust significance evidence.
Retrospective PRISM, hindsight FINAL_CDL and the original temporal assumptions
remain unchanged. Changing random seeds cannot validate upstream leakage.

Resume: only successful runs with matching input SHA256/config fingerprints
and unchanged output hashes are skipped. Partial/unmarked runs stop with a
message; --retry-incomplete explicitly reruns those seed folders.
--analysis-only regenerates summaries from all verified completed runs.
Outputs for missing year/mask cells are reported, not invented.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

VERSION = 'CY2_ALL_IN_ONE_50_SEEDS_V1'  # model-run fingerprint unchanged
ANALYSIS_VERSION = 'V2_WITH_LEGACY_WILCOXON_AND_HLN_DM'
LABELS = {
 ('PREVIOUS_CDL','PREVIOUS_CDL'): 'Previous CDL training / previous CDL testing',
 ('SAME_YEAR_CDL','PREVIOUS_CDL'): 'Same-year CDL training / previous CDL testing',
 ('SAME_YEAR_CDL','FINAL_CDL'): 'Final CDL hindsight benchmark',
 ('SAME_YEAR_CDL','CLEAN_REBUILT'): 'Our ICDLs',
 ('SAME_YEAR_CDL','STUDY'): 'Published ICDLs',
}
PAIRS = [
 ('previous_vs_ours', ('PREVIOUS_CDL','PREVIOUS_CDL'), ('SAME_YEAR_CDL','CLEAN_REBUILT')),
 ('previous_vs_published', ('PREVIOUS_CDL','PREVIOUS_CDL'), ('SAME_YEAR_CDL','STUDY')),
 ('ours_vs_published', ('SAME_YEAR_CDL','CLEAN_REBUILT'), ('SAME_YEAR_CDL','STUDY')),
 ('training_mask_effect', ('PREVIOUS_CDL','PREVIOUS_CDL'), ('SAME_YEAR_CDL','PREVIOUS_CDL')),
]
METRICS = ['r2','rmse','mae','anomaly_r2','mse_skill_vs_trend']
KEYS = ['training_mask','testing_mask','year','doy','FIPS']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def save_json(path,value):
    pending=path.with_suffix('.tmp')
    pending.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(pending,path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path(r'D:\corn_yield'))
    p.add_argument('--input-dir',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--n-seeds',type=int,default=50)
    p.add_argument('--first-seed',type=int,default=1)
    p.add_argument('--test-years',type=int,nargs='+',default=[2021,2022,2023,2024,2025])
    p.add_argument('--doys',type=int,nargs='+',default=[193,209,225,241,257,273])
    p.add_argument('--trees',type=int,default=500)
    p.add_argument('--jobs',type=int,default=-1)
    p.add_argument('--n-boot',type=int,default=2000)
    p.add_argument('--analysis-only',action='store_true')
    p.add_argument('--retry-incomplete',action='store_true')
    return p.parse_args()


def required_outputs(folder):
    return [folder/'model_comparison_metrics.csv',folder/'county_predictions.csv',
            folder/'feature_importance'/'feature_importance.csv',folder/'run_config.json']


def verify_complete(folder,run_key):
    marker=folder/'completed_seed.json'
    if not marker.exists(): return False
    info=json.loads(marker.read_text())
    if info['run_key']!=run_key:
        raise RuntimeError(f'{folder}: inputs or configuration changed. Use a new --output-dir.')
    for path in required_outputs(folder):
        if not path.exists() or sha(path)!=info['outputs'].get(str(path.relative_to(folder))):
            raise RuntimeError(f'{folder}: saved output changed or is missing; restore it or use a new output folder.')
    return True


def run_models(args,source,base_key,out):
    seeds=list(range(args.first_seed,args.first_seed+args.n_seeds))
    engine=out/'engine'/'v5_model_embedded.py'
    engine.parent.mkdir(parents=True,exist_ok=True)
    engine.write_bytes(source)
    for pos,seed in enumerate(seeds,1):
        folder=out/'seeds'/f'seed_{seed}'
        key=digest({'base_key':base_key,'seed':seed})
        if verify_complete(folder,key):
            print(f'[{pos}/{len(seeds)}] seed {seed}: verified complete; skipping.',flush=True)
            continue
        if args.analysis_only:
            raise RuntimeError(f'Seed {seed} is not verified complete. Run without --analysis-only first.')
        if folder.exists() and any(folder.iterdir()) and not args.retry_incomplete:
            raise RuntimeError(f'{folder} has an incomplete/unmarked run. Rerun with --retry-incomplete to repeat it.')
        folder.mkdir(parents=True,exist_ok=True)
        print(f'[{pos}/{len(seeds)}] running seed {seed}. Log: {folder / "model_run.log"}',flush=True)
        cmd=[sys.executable,'-u',str(engine),'--input-dir',str(args.input_dir),
             '--work-dir',str(out/'work'),'--output-dir',str(folder),'--seed',str(seed),
             '--trees',str(args.trees),'--jobs',str(args.jobs),'--test-years',*map(str,args.test_years),
             '--doys',*map(str,args.doys)]
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        with (folder/'model_run.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                  encoding='utf-8',errors='replace',env=env)
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        if code:
            raise RuntimeError(f'Seed {seed} failed (exit {code}). See {folder / "model_run.log"}. Completed seeds are preserved.')
        for path in required_outputs(folder):
            if not path.exists(): raise RuntimeError(f'Model did not produce required output: {path}')
        save_json(folder/'completed_seed.json',{'run_key':key,'seed':seed,
            'outputs':{str(p.relative_to(folder)):sha(p) for p in required_outputs(folder)}})
    return seeds


def scenario(df,pair):
    return df[(df.training_mask==pair[0]) & (df.testing_mask==pair[1])].copy()


def holm(p):
    import numpy as np
    p=np.asarray(p,dtype=float);out=np.full(len(p),np.nan)
    idx=np.flatnonzero(np.isfinite(p));order=idx[np.argsort(p[idx])]
    if len(order): out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
    return out


def hln_dm_test(e_a,e_b,loss='squared',h=1):
    """Legacy HLN-corrected Diebold-Mariano diagnostic.

    Positive mean loss difference (A-B) means B has lower average loss.
    With h=1 this does not account for spatial dependence among counties, so
    treat the p-value as a secondary diagnostic rather than a spatially robust
    significance test.
    """
    import numpy as np
    from scipy import stats
    if loss=='squared':
        d=e_a**2-e_b**2
    elif loss=='absolute':
        d=np.abs(e_a)-np.abs(e_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
    d=d[np.isfinite(d)];n=len(d)
    if n<8: return np.nan,np.nan,np.nan
    dbar=d.mean();gamma0=np.mean((d-dbar)**2);lrv=gamma0
    for k in range(1,h):
        gk=np.mean((d[k:]-dbar)*(d[:-k]-dbar))
        lrv += 2*(1-k/h)*gk
    if lrv<=0: return np.nan,np.nan,dbar
    dm=dbar/np.sqrt(lrv/n)
    corr=np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    dm_hln=dm*corr
    p=2*(1-stats.t.cdf(abs(dm_hln),df=n-1))
    return dm_hln,p,dbar


def analyze(out,seeds,n_boot):
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'summary';figdir=dest/'figures';figdir.mkdir(parents=True,exist_ok=True)
    met=[];ensemble_sum=None;reference=None;importance=[]
    for seed in seeds:
        folder=out/'seeds'/f'seed_{seed}'
        m=pd.read_csv(folder/'model_comparison_metrics.csv')
        mk=['training_mask','testing_mask','year','doy']
        if m.duplicated(mk).any(): raise ValueError(f'Duplicate metrics in seed {seed}')
        m['seed']=seed;met.append(m)
        pred=pd.read_csv(folder/'county_predictions.csv',dtype={'FIPS':'string'})
        pred['FIPS']=pred.FIPS.str.strip().str.replace(r'\.0$','',regex=True).str.zfill(5)
        if not pred.FIPS.str.fullmatch(r'\d{5}').fillna(False).all(): raise ValueError('Invalid FIPS')
        if pred.duplicated(KEYS).any(): raise ValueError(f'Duplicate county keys in seed {seed}')
        pred=pred.set_index(KEYS).sort_index()
        cols=['yield_bu_acre','hist_yield_trend','predicted_yield']
        if not np.isfinite(pred[cols].to_numpy()).all(): raise ValueError(f'Nonfinite prediction data: seed {seed}')
        if reference is None:
            reference=pred[['yield_bu_acre','hist_yield_trend']].copy()
            ensemble_sum=pred.predicted_yield.copy()
        else:
            if not reference.index.equals(pred.index): raise ValueError(f'County/scenario coverage changes across seeds: {seed}')
            if not np.allclose(reference.values,pred[reference.columns].values,rtol=0,atol=1e-9):
                raise ValueError(f'Observed yields/trends changed in seed {seed}')
            ensemble_sum += pred.predicted_yield
        imp=pd.read_csv(folder/'feature_importance'/'feature_importance.csv');imp['seed']=seed;importance.append(imp)
    metrics=pd.concat(met,ignore_index=True)
    if (metrics.groupby(['training_mask','testing_mask','year','doy']).seed.nunique()!=len(seeds)).any():
        raise ValueError('Some metric cells lack seeds.')
    metrics['scenario']=[LABELS.get((a,b),f'{a} -> {b}') for a,b in zip(metrics.training_mask,metrics.testing_mask)]
    metrics.to_csv(dest/'all_seed_metrics.csv',index=False)
    available=[x for x in METRICS if x in metrics]
    summary=metrics.groupby(['training_mask','testing_mask','scenario','year','doy'])[available].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(dest/'seed_summary_mean_sd.csv',index=False)
    counts=metrics.groupby(['training_mask','testing_mask','scenario','year','doy']).agg(n_seeds=('seed','nunique'),
                        counties_min=('n_test','min'),counties_max=('n_test','max')).reset_index()
    counts.to_csv(dest/'county_counts.csv',index=False)
    if (counts.counties_min!=counts.counties_max).any(): raise ValueError('County counts differ across seeds')
    skipped=[];pair_rows=[]
    def test_pair(g,year,doy,tag,metric):
        a=g.A.to_numpy();b=g.B.to_numpy();keep=np.isfinite(a)&np.isfinite(b);a=a[keep];b=b[keep]
        d=a-b;n=len(d)
        if n<3: return
        if np.all(d==0): t,p,w,wp=0.,1.,0.,1.
        elif np.std(d,ddof=1)==0: t,p,w,wp=np.nan,np.nan,np.nan,np.nan
        else:
            t,p=stats.ttest_rel(a,b)
            try: w,wp=stats.wilcoxon(a,b)
            except ValueError: w,wp=np.nan,np.nan
        pair_rows.append(dict(comparison=tag,year=year,doy=doy,metric=metric,n_seeds=n,
            A_mean=a.mean(),B_mean=b.mean(),difference_A_minus_B=d.mean(),difference_sd=d.std(ddof=1),
            paired_t=t,paired_t_p=p,wilcoxon_p=wp))
    for tag,A,B in PAIRS:
        aa=scenario(metrics,A);bb=scenario(metrics,B)
        merged=aa.merge(bb,on=['seed','year','doy'],suffixes=('_a','_b'),validate='one_to_one')
        if merged.empty:
            skipped.append(dict(comparison=tag,reason='No shared year/DOY cells for these masks'));continue
        if not (merged.n_test_a==merged.n_test_b).all(): raise ValueError('Scenario county counts differ')
        for metric in available:
            g=merged[['seed','year','doy',metric+'_a',metric+'_b']].rename(columns={metric+'_a':'A',metric+'_b':'B'})
            for (year,doy),cell in g.groupby(['year','doy']): test_pair(cell,year,doy,tag,metric)
            # Retain the same finite cells in each seed before an overall paired test.
            finite=g[np.isfinite(g.A)&np.isfinite(g.B)]
            complete=finite.groupby(['year','doy']).seed.nunique()
            cells=complete[complete==len(seeds)].index
            finite=finite.set_index(['year','doy']).loc[lambda x:x.index.isin(cells)].reset_index()
            pooled=finite.groupby('seed')[['A','B']].mean().reset_index()
            test_pair(pooled,'ALL','ALL',tag,metric)
    tests=pd.DataFrame(pair_rows)
    if not tests.empty:
        tests['paired_t_p_holm']=holm(tests.paired_t_p)
        tests.to_csv(dest/'paired_seed_tests.csv',index=False)
    # Ensemble gives ONE prediction per county, never 50 duplicated observations.
    ens=reference.copy();ens['predicted_yield']=ensemble_sum/len(seeds);ens=ens.reset_index()
    ens['n_seeds']=len(seeds);ens.to_csv(dest/'ensemble_county_predictions.csv',index=False)
    rng=np.random.default_rng(42);boot_rows=[];legacy_rows=[]
    for tag,A,B in PAIRS:
        aa=scenario(ens,A);bb=scenario(ens,B)
        for (year,doy),a in aa.groupby(['year','doy']):
            b=bb[(bb.year==year)&(bb.doy==doy)]
            if b.empty:
                skipped.append(dict(comparison=tag,year=year,doy=doy,reason='Mask unavailable at this checkpoint'));continue
            if set(a.FIPS)!=set(b.FIPS): raise ValueError(f'Nonmatched counties: {tag} {year} {doy}')
            g=a.merge(b,on='FIPS',suffixes=('_a','_b'),validate='one_to_one')
            for c in ['yield_bu_acre','hist_yield_trend']:
                if not np.allclose(g[c+'_a'],g[c+'_b'],rtol=0,atol=1e-9): raise ValueError('Scenario targets/trends differ')
            y=g.yield_bu_acre_a.to_numpy();base=g.hist_yield_trend_a.to_numpy()
            ea=y-g.predicted_yield_a.to_numpy();eb=y-g.predicted_yield_b.to_numpy()

            # Legacy county-level diagnostics from the old fixed stats script.
            # These use the ONE ensemble prediction per county, not 50 duplicated
            # seed rows. They are retained for continuity but are not spatially robust.
            try:
                wil_stat,wil_p=stats.wilcoxon(np.abs(ea),np.abs(eb))
            except ValueError:
                wil_stat,wil_p=np.nan,np.nan
            dm_sq,dm_sq_p,dm_sq_diff=hln_dm_test(ea,eb,'squared',h=1)
            dm_abs,dm_abs_p,dm_abs_diff=hln_dm_test(ea,eb,'absolute',h=1)
            legacy_rows.append(dict(
                comparison=tag,year=year,doy=doy,N=len(g),n_seeds=len(seeds),
                A=LABELS[A],B=LABELS[B],
                Wilcoxon_abs_error_stat=wil_stat,Wilcoxon_abs_error_p=wil_p,
                DM_HLN_sq=dm_sq,DM_HLN_sq_p=dm_sq_p,DM_sq_mean_loss_A_minus_B=dm_sq_diff,
                DM_HLN_abs=dm_abs,DM_HLN_abs_p=dm_abs_p,DM_abs_mean_loss_A_minus_B=dm_abs_diff,
                note='Secondary county-level diagnostic; p-values do not account for within-state spatial dependence.'
            ))

            groups=g.FIPS.str[:2].to_numpy();states=np.unique(groups)
            def evaluate(idx):
                mse_a=np.mean(ea[idx]**2);mse_b=np.mean(eb[idx]**2);den=np.mean((y[idx]-base[idx])**2)
                return np.array([np.sqrt(mse_a)-np.sqrt(mse_b),np.mean(abs(ea[idx]))-np.mean(abs(eb[idx])),
                                 1-mse_a/den if den>0 else np.nan,1-mse_b/den if den>0 else np.nan])
            point=evaluate(np.arange(len(g)));lo=hi=np.full(4,np.nan)
            if len(states)>=2:
                idx={s:np.flatnonzero(groups==s) for s in states}
                draws=np.array([evaluate(np.concatenate([idx[s] for s in rng.choice(states,len(states),replace=True)])) for _ in range(n_boot)])
                lo,hi=np.nanpercentile(draws,[2.5,97.5],axis=0)
            rec=dict(comparison=tag,year=year,doy=doy,N=len(g),N_states=len(states),n_seeds=len(seeds),
                     A=LABELS[A],B=LABELS[B],RMSE_A=np.sqrt(np.mean(ea**2)),RMSE_B=np.sqrt(np.mean(eb**2)),
                     bias_A=-ea.mean(),bias_B=-eb.mean())
            for j,name in enumerate(['RMSE_gap_A_minus_B','MAE_gap_A_minus_B','Skill_A','Skill_B']):
                rec[name]=point[j];rec[name+'_lo95']=lo[j];rec[name+'_hi95']=hi[j]
            boot_rows.append(rec)
    boots=pd.DataFrame(boot_rows);boots.to_csv(dest/'ensemble_state_bootstrap.csv',index=False)

    legacy=pd.DataFrame(legacy_rows)
    if not legacy.empty:
        legacy['Wilcoxon_abs_error_p_holm']=np.nan
        legacy['DM_HLN_sq_p_holm']=np.nan
        legacy['DM_HLN_abs_p_holm']=np.nan
        for tag,idx in legacy.groupby('comparison').groups.items():
            idx=list(idx)
            legacy.loc[idx,'Wilcoxon_abs_error_p_holm']=holm(legacy.loc[idx,'Wilcoxon_abs_error_p'].to_numpy())
            legacy.loc[idx,'DM_HLN_sq_p_holm']=holm(legacy.loc[idx,'DM_HLN_sq_p'].to_numpy())
            legacy.loc[idx,'DM_HLN_abs_p_holm']=holm(legacy.loc[idx,'DM_HLN_abs_p'].to_numpy())
        legacy.to_csv(dest/'ensemble_legacy_wilcoxon_dm_tests.csv',index=False)

    pd.DataFrame(skipped,columns=['comparison','year','doy','reason']).to_csv(dest/'skipped_comparisons.csv',index=False)
    # Yearly seasonal plots: ALL available scenarios, including ours AND published.
    for year,g in summary.groupby('year'):
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for (train,test),h in g.groupby(['training_mask','testing_mask']):
            h=h.sort_values('doy');label=LABELS.get((train,test),f'{train} -> {test}')
            for ax,metric in zip(axes.flat,['r2','rmse','anomaly_r2','counties']):
                if metric=='counties':
                    c=counts[(counts.year==year)&(counts.training_mask==train)&(counts.testing_mask==test)].sort_values('doy')
                    ax.plot(c.doy,c.counties_min,'o-',label=label)
                elif metric in available:
                    ax.errorbar(h.doy,h[metric+'_mean'],yerr=h[metric+'_std'].fillna(0),fmt='o-',capsize=3,label=label)
                ax.set_title('Counties evaluated' if metric=='counties' else metric+' (mean +/- SD across seeds)')
                ax.set_xlabel('Forecast DOY');ax.grid(alpha=.25)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
        fig.suptitle(f'{int(year)}: {len(seeds)} seeds | CLEAN_REBUILT = ours; STUDY = published')
        fig.tight_layout(rect=[0,.1,1,.95]);fig.savefig(figdir/f'seasonal_comparison_{int(year)}.png',dpi=160);plt.close(fig)
    if not boots.empty:
        for tag,g in boots.groupby('comparison'):
            g=g.sort_values(['year','doy']);x=np.arange(len(g))
            fig,ax=plt.subplots(figsize=(max(9,len(g)*.35),5))
            ax.vlines(x,g.RMSE_gap_A_minus_B_lo95,g.RMSE_gap_A_minus_B_hi95,color='steelblue')
            ax.scatter(x,g.RMSE_gap_A_minus_B,color='navy');ax.axhline(0,color='black',ls='--')
            ax.set_xticks(x);ax.set_xticklabels([f'{int(r.year)}\nD{int(r.doy)}' for r in g.itertuples()],fontsize=7)
            ax.set_ylabel('Ensemble RMSE(A) - RMSE(B), bu/acre; positive favors B')
            ax.set_title(f'{g.A.iloc[0]} vs {g.B.iloc[0]}\nState-bootstrap 95% intervals (unadjusted)')
            fig.tight_layout();fig.savefig(figdir/f'ensemble_bootstrap_{tag}.png',dpi=160);plt.close(fig)

    # Legacy diagnostic p-value heatmaps, retained for comparison with the old
    # run_stat_tests_fixed.py outputs. These are secondary, non-spatial tests.
    if not legacy.empty:
        for tag,g in legacy.groupby('comparison'):
            for col,title_name in [('DM_HLN_sq_p','HLN-DM squared-loss p-value'),
                                   ('Wilcoxon_abs_error_p','Wilcoxon absolute-error p-value')]:
                piv=g.pivot(index='year',columns='doy',values=col).sort_index().sort_index(axis=1)
                fig,ax=plt.subplots(figsize=(10,4))
                im=ax.imshow(piv.values,vmin=0,vmax=.20,aspect='auto',cmap='RdYlGn_r')
                ax.set_xticks(range(len(piv.columns)));ax.set_xticklabels([int(x) for x in piv.columns])
                ax.set_yticks(range(len(piv.index)));ax.set_yticklabels([int(y) for y in piv.index])
                ax.set_xlabel('Forecast DOY');ax.set_ylabel('Test year')
                ax.set_title(f'{title_name}: {tag}\nSecondary county-level diagnostic; not spatially robust')
                for i in range(piv.shape[0]):
                    for j in range(piv.shape[1]):
                        v=piv.values[i,j]
                        if np.isfinite(v): ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8)
                fig.colorbar(im,ax=ax,label='p-value')
                fig.tight_layout();fig.savefig(figdir/f'legacy_{col}_{tag}.png',dpi=160,bbox_inches='tight');plt.close(fig)

    # Average importance by fit: seed/year/DOY, with absent features assigned zero.
    imp=pd.concat(importance,ignore_index=True)
    for train,g in imp.groupby('training_mask'):
        pivot=g.pivot_table(index=['seed','year','doy'],columns='feature',values='gain_share',fill_value=0)
        means=pivot.mean().sort_values(ascending=False)
        means.rename('mean_gain_share').to_csv(dest/f'{train}_mean_feature_importance.csv')
        top=100*means.head(25).sort_values();fig,ax=plt.subplots(figsize=(10,8))
        top.plot.barh(ax=ax);ax.set_xlabel('Mean training split-gain share (%)')
        ax.set_title(f'{train}: importance across seeds and evaluated fits')
        fig.tight_layout();fig.savefig(figdir/f'{train}_mean_feature_importance.png',dpi=160);plt.close(fig)
    print(f'Analysis complete: {dest}',flush=True)
    print('CLEAN_REBUILT = Our ICDLs; STUDY = Published ICDLs.',flush=True)
    print('Primary inference: paired-seed tests + state-clustered ensemble bootstrap.',flush=True)
    print('Legacy Wilcoxon/HLN-DM diagnostics: ensemble_legacy_wilcoxon_dm_tests.csv (secondary; non-spatial).',flush=True)


def main():
    args=parse_args()
    if args.n_seeds<3 or args.n_boot<100: raise SystemExit('Use at least 3 seeds and 100 bootstrap samples.')
    args.input_dir=(args.input_dir or args.root/'model_inputs').resolve()
    out=(args.output_dir or args.root/'yield_model_results'/'v5_50_seed_all_in_one').resolve()
    if not args.input_dir.is_dir(): raise SystemExit(f'Input folder not found: {args.input_dir}')
    for package in ['numpy','pandas','scipy','matplotlib','sklearn','xgboost']:
        if importlib.util.find_spec(package) is None:
            raise SystemExit(f'Missing package {package}; activate your corn-yield environment.')
    source=zlib.decompress(base64.b64decode(MODEL_PAYLOAD))
    print('CLEAN_REBUILT = OUR ICDLs | STUDY = PUBLISHED ICDLs',flush=True)
    print(f'{args.n_seeds} sequential model runs. Checking input fingerprints...',flush=True)
    files=sorted(p for p in args.input_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.csv','.zip'})
    if not files: raise SystemExit('No input CSVs or ZIP archives found.')
    if out == args.input_dir or args.input_dir in out.parents:
        raise SystemExit('Output directory must be outside the input directory.')
    config=dict(version=VERSION,model_sha256=hashlib.sha256(source).hexdigest(),
                test_years=args.test_years,doys=args.doys,trees=args.trees,jobs=args.jobs,
                input_dir=str(args.input_dir),inputs={str(p.relative_to(args.input_dir)):sha(p) for p in files})
    out.mkdir(parents=True,exist_ok=True)
    base_key=digest(config)
    old=out/'experiment_config.json'
    if old.exists() and digest(json.loads(old.read_text()))!=base_key:
        raise SystemExit('Inputs/configuration changed. Use a new --output-dir to keep experiments separate.')
    save_json(old,config)
    seeds=run_models(args,source,base_key,out)
    analyze(out,seeds,args.n_boot)
    save_json(out/'analysis_config.json',dict(seeds=seeds,n_boot=args.n_boot,ensemble='mean predictions across seeds',analysis_version=ANALYSIS_VERSION,legacy_tests='Wilcoxon absolute-error + HLN-DM squared/absolute loss on ensemble county predictions'))

# Embedded, compressed original V5.2 model, including its plot-save fix.
MODEL_PAYLOAD = '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'

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))
