"""Exploratory raw descriptors and per-checkpoint oracle targets, no training."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, pearsonr

P = Path(os.environ['LLIE_PROJECT_ROOT'])
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ['LLIE_DATA_ROOT'])
ROOT = DATA/'MILLs-DSLR_train_validation'
OUT = DATA/'content_features_oracle_20261102'
GRID = P/'06_experiments/postprocess_gate_ablation_20261026/results/analysis/grid_per_pair.csv'
BASE = P/'06_experiments/SCI_RUAS_comparison_20261026/results/analysis/real_per_pair.csv'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    assert ROOT.is_dir() and GRID.is_file() and BASE.is_file()
    assert not OUT.exists(), 'Output exists; preserve prior results.'
    base=pd.read_csv(BASE); base=base[base.model=='raw'].copy()
    assert len(base)==120 and not base.duplicated(['scene_id','level']).any()
    rows=[]
    for r in base.sort_values(['scene_id','level']).itertuples():
        path=ROOT/'validation'/'input'/f'{r.scene_id}-{r.level}.png'
        assert sha(path)==r.input_sha256, str(path)
        with Image.open(path) as im:
            assert im.mode=='RGB' and im.size==(600,400) and im.getexif().get(274,1)==1
            x=np.ascontiguousarray(np.asarray(im,dtype=np.float32)[:396]/np.float32(255))
        assert hashlib.sha256(x.tobytes()).hexdigest()==r.input_hash
        y=x @ np.array([.2126,.7152,.0722],dtype=np.float32)
        hist=np.histogram(y,bins=256,range=(0,1))[0].astype(np.float64); p=hist[hist>0]/hist.sum()
        rgbhist=np.histogramdd(x.reshape(-1,3),bins=(8,8,8),range=((0,1),(0,1),(0,1)))[0].ravel(); q=rgbhist[rgbhist>0]/rgbhist.sum()
        rows.append(dict(scene_id=r.scene_id,level=r.level,input_sha256=r.input_sha256,input_hash=r.input_hash,
            mean=float(y.mean()),std=float(y.std()),dark_ratio=float((y<.1).mean()),bright_ratio=float((y>.7).mean()),
            gray_entropy=float(-(p*np.log2(p)).sum()),rgb_entropy_8x8x8=float(-(q*np.log2(q)).sum())))
    xdf=pd.DataFrame(rows)
    grid=pd.read_csv(GRID); assert len(grid)==75600
    keys=['source_model','scene_id','level']; assert not grid.duplicated(keys+['threshold','temperature']).any()
    assert (grid.groupby(keys).size()==210).all() and grid.groupby(keys).ngroups==360
    grid=grid.sort_values(keys+['mse','threshold','temperature'],kind='stable')
    best=grid.groupby(keys,sort=False).first().reset_index()
    best=best.rename(columns={'threshold':'tau_star','temperature':'temperature_star','mse':'oracle_mse','psnr_db':'oracle_psnr','alpha':'oracle_alpha'})
    near=grid.merge(best[keys+['oracle_psnr']],on=keys,validate='many_to_one')
    near=near[near.psnr_db >= near.oracle_psnr-0.01-1e-10]
    widths=near.groupby(keys).agg(near_count=('threshold','size'),near_tau_min=('threshold','min'),near_tau_max=('threshold','max'),near_T_min=('temperature','min'),near_T_max=('temperature','max')).reset_index()
    best=best.merge(widths,on=keys,validate='one_to_one')
    data=best.merge(xdf,on=['scene_id','level'],validate='many_to_one'); assert len(data)==360
    fs=['mean','std','dark_ratio','bright_ratio','gray_entropy','rgb_entropy_8x8x8']
    corr=[]
    for model,g in data.groupby('source_model'):
        for feature in fs:
            for target in ['tau_star','temperature_star','oracle_alpha']:
                a=g[feature].to_numpy(); b=g[target].to_numpy()
                if np.std(a)==0 or np.std(b)==0: pear=spear=None
                else: pear=float(pearsonr(a,b).statistic); spear=float(spearmanr(a,b).statistic)
                # Remove level means to distinguish a within-level association from exposure trend.
                aa=g[feature]-g.groupby('level')[feature].transform('mean'); bb=g[target]-g.groupby('level')[target].transform('mean')
                within=float(pearsonr(aa,bb).statistic) if aa.std()>1e-12 and bb.std()>1e-12 else None
                corr.append(dict(model=model,feature=feature,target=target,pearson=pear,spearman=spear,pearson_within_level=within,n=len(g)))
    OUT.mkdir(parents=True)
    xdf.to_csv(OUT/'X_features.csv',index=False)
    data.to_csv(OUT/'Y_per_model_oracle.csv',index=False)
    pd.DataFrame(corr).to_csv(HERE/'correlations.csv',index=False)
    summary={}
    for model,g in data.groupby('source_model'):
        summary[model]={'n':len(g),'tau_boundary_fraction':float(g.tau_star.isin([.1,.3]).mean()),'T_boundary_fraction':float(g.temperature_star.isin([.01,.1]).mean()),'median_near_count_0_01db':float(g.near_count.median()),'median_near_tau_span':float((g.near_tau_max-g.near_tau_min).median()),'median_near_T_span':float((g.near_T_max-g.near_T_min).median())}
    meta={'inputs':120,'targets_per_model':120,'grid_rows':len(grid),'grid_sha256':sha(GRID),'baseline_sha256':sha(BASE),'dataset_root':str(ROOT),'outputs':str(OUT),'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'summary':summary}
    (HERE/'summary.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
