"""Independently check saved group targets, fold isolation, predictions and arithmetic."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
import json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
H=Path(__file__).resolve().parent
P=Path(os.environ['LLIE_PROJECT_ROOT'])
R=H/'results'
E=Path(os.environ['LLIE_DATA_ROOT'])
grid=pd.read_csv(P/'06_experiments/postprocess_gate_ablation_20261026/results/analysis/grid_per_pair.csv')
labels=pd.read_csv(E/'content_gate_loso_20261102/smooth_targets_005db.csv')
feat=pd.read_csv(E/'content_features_oracle_20261102/X_features.csv')
pr=pd.read_csv(R/'predicted_parameters.csv')
man=json.loads((R/'fold_manifest.json').read_text(encoding='utf8'))
keys=['source_model','scene_id','level']
g=grid.copy(); g['loss']=g.groupby(keys).psnr_db.transform('max')-g.psnr_db
check=g[g.loss<=.05+1e-10].groupby(keys).agg(tau=('threshold','mean'),temperature=('temperature','mean'),near_count=('threshold','size')).reset_index()
a=check.merge(labels,on=keys,validate='one_to_one',suffixes=('_a','_b')); assert len(a)==360
for col in ['tau','temperature','near_count']: assert np.allclose(a[col+'_a'],a[col+'_b'],rtol=0,atol=1e-12)
allscenes=set(feat.scene_id)
assert len(man)==36 and len(set((f['model'],f['heldout']) for f in man))==36
for f in man:
    assert set(f['train_scenes'])==allscenes-{f['heldout']} and f['train_count']==110 and f['test_count']==10
merged=check.merge(feat,on=['scene_id','level'],validate='many_to_one')
for model in sorted(merged.source_model.unique()):
    s=sorted(allscenes)[0]; d=merged[merged.source_model==model]
    train=d[d.scene_id!=s].sort_values(['scene_id','level']); test=d[d.scene_id==s].sort_values('level')
    expected=pr[(pr.source_model==model)&(pr.scene_id==s)].sort_values('level')
    for tag,fs in [('rf_mean',['mean']),('rf_five',['mean','std','dark_ratio','bright_ratio','gray_entropy'])]:
        for col,suffix in [('tau','tau'),('temperature','T')]:
            m=RandomForestRegressor(n_estimators=50,max_depth=3,random_state=42,n_jobs=1).fit(train[fs],train[col])
            assert np.allclose(m.predict(test[fs]),expected[tag+'_'+suffix],rtol=0,atol=1e-14)
results=pd.read_csv(R/'per_pair.csv'); assert len(results)==2160 and not results.duplicated(keys+['method']).any()
assert np.allclose(results.psnr_db,-10*np.log10(results.mse),rtol=0,atol=1e-10)
summ=results.groupby(['source_model','method']).psnr_db.mean().sort_index(); saved=pd.read_csv(R/'overall.csv').set_index(['source_model','method']).psnr_db.sort_index(); assert np.allclose(summ,saved,rtol=0,atol=1e-12)
sys.path.insert(0,str(P/'06_experiments/postprocess_gate_pilot_20261026'))
from fusion_gate import fuse
root=E/'MILLs-DSLR_train_validation'; source=E/'SCI_RUAS_comparison_v1'
pair=pd.read_csv(R/'paired.csv'); pixel_checks=0
for model in sorted(pair.source_model.unique()):
    part=pair[pair.source_model==model]; sample=part.nsmallest(1,'rf_five_minus_fixed').iloc[0]
    for scene,level in [(str(sample.scene_id),int(sample.level)),('Test1',1)]:
        def img(path):
            with Image.open(path) as im: return np.ascontiguousarray(np.asarray(im,dtype=np.float32)[:396]/np.float32(255))
        x=img(root/'validation/input'/f'{scene}-{level}.png'); ref=img(root/'validation/gt'/f'{scene}.png')
        if model=='ZeroDCEpp_official':
            with np.load(P/'06_experiments/baseline_validation/outputs'/f'{scene}.npz') as z: y=z[f'level_{level:02d}']
        else: y=np.load(source/model/'real'/f'{scene}-{level}.npy')
        row=pr[(pr.source_model==model)&(pr.scene_id==scene)&(pr.level==level)].iloc[0]
        out,_=fuse(x,y,threshold=np.float32(row.rf_five_tau),temperature=np.float32(row.rf_five_T),global_map=True)
        measured=results[(results.source_model==model)&(results.scene_id==scene)&(results.level==level)&(results.method=='rf_five')].iloc[0]
        mse=float(np.mean((out.astype(np.float64)-ref.astype(np.float64))**2))
        assert abs(mse-measured.mse)<1e-12 and hashlib.sha256(out.tobytes()).hexdigest()==measured.output_hash
        pixel_checks+=1
report={'status':'PASS','checks':['360 targets recomputed from stored grid','36 fold membership lists exclude heldout scene','two forests independently refitted for each of 3 models for first fold','2160 distinct rows and PSNR/MSE arithmetic','overall aggregation',f'{pixel_checks} independently rendered dynamic images against existing fuse implementation'], 'limits':'Pixel-level rerender is a six-image sample; primary run checked every raw, original and fixed metric against archived results. For exact refitting recompute full-precision targets from grid, since CSV parsing roundoff can change tied forest splits.'}
(R/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8'); print(json.dumps(report))
