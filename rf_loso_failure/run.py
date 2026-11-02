"""Predeclared CPU-only scene-held-out gate experiment using archived outputs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import sklearn

HERE=Path(__file__).resolve().parent
P=Path(os.environ['LLIE_PROJECT_ROOT'])
OLD=P/'06_experiments/postprocess_gate_ablation_20261026/results/analysis'
COMP=P/'06_experiments/SCI_RUAS_comparison_20261026'
ZERO=P/'06_experiments/baseline_validation'
DATA=Path(os.environ['LLIE_DATA_ROOT'])
ROOT=DATA/'MILLs-DSLR_train_validation'
FEATURES=DATA/'content_features_oracle_20261102/X_features.csv'
LABELDIR=DATA/'content_gate_loso_20261102'
OUT=HERE/'results'
KEY=['source_model','scene_id','level']
FEATURE_NAMES=['mean','std','dark_ratio','bright_ratio','gray_entropy']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def ahash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def guard():
    while psutil.virtual_memory().available < .5*1024**3:
        print('waiting for available RAM >= 0.5 GiB',flush=True); time.sleep(10)
def read_image(path,expected):
    assert sha(path)==expected, str(path)
    with Image.open(path) as im:
        assert im.mode=='RGB' and im.size==(600,400) and im.getexif().get(274,1)==1
        return np.ascontiguousarray(np.asarray(im,dtype=np.float32)[:396]/np.float32(255))
def alpha(mu,tau,temp):
    z=np.clip((np.float32(mu)-np.float32(tau))/np.float32(temp),np.float32(-60),np.float32(60))
    return np.float32(1)/(np.float32(1)+np.exp(z))
def render(x,y,a): return np.clip(a*y+(np.float32(1)-a)*x,0,1)
def metric(x,ref):
    d=x.astype(np.float64)-ref.astype(np.float64); mse=float(np.mean(d*d,dtype=np.float64))
    assert mse>0 and math.isfinite(mse)
    return mse,-10*math.log10(mse)

def main():
    guard(); assert not OUT.exists() and not LABELDIR.exists(), 'Do not overwrite prior run'
    grid_path=OLD/'grid_per_pair.csv'; old_summary=json.loads((OLD/'summary.json').read_text(encoding='utf8'))
    assert sha(grid_path)==old_summary['artifacts']['grid_per_pair.csv']
    grid=pd.read_csv(grid_path); feat=pd.read_csv(FEATURES)
    assert len(grid)==75600 and len(feat)==120 and not feat.duplicated(['scene_id','level']).any()
    assert not grid.duplicated(KEY+['threshold','temperature']).any() and (grid.groupby(KEY).size()==210).all()
    scene_ids=sorted(feat.scene_id.unique()); assert len(scene_ids)==12 and feat.groupby('scene_id').size().eq(10).all()
    best=grid.groupby(KEY).psnr_db.transform('max'); near=grid[grid.psnr_db >= best-.05-1e-10]
    target=near.groupby(KEY).agg(tau=('threshold','mean'),temperature=('temperature','mean'),near_count=('threshold','size')).reset_index()
    assert len(target)==360 and target.tau.between(.1,.3).all() and target.temperature.between(.01,.1).all()
    tab=target.merge(feat,on=['scene_id','level'],validate='many_to_one'); assert len(tab)==360
    LABELDIR.mkdir(parents=True); OUT.mkdir(parents=True); (OUT/'cells').mkdir()
    target.to_csv(LABELDIR/'smooth_targets_005db.csv',index=False)
    params=[]; fold_manifest=[]
    for model in sorted(tab.source_model.unique()):
        d=tab[tab.source_model==model]
        for s in scene_ids:
            guard(); train=d[d.scene_id!=s].sort_values(['scene_id','level']); test=d[d.scene_id==s].sort_values('level')
            assert len(train)==110 and len(test)==10 and s not in set(train.scene_id)
            g=grid[(grid.source_model==model)&(grid.scene_id!=s)]
            avg=g.groupby(['threshold','temperature']).psnr_db.mean().reset_index().sort_values(['psnr_db','threshold','temperature'],ascending=[False,True,True],kind='stable')
            assert len(avg)==210; constant=avg.iloc[0]
            pred={}
            for name,fs in [('rf_mean',['mean']),('rf_five',FEATURE_NAMES)]:
                pred[name]=[]
                for col in ['tau','temperature']:
                    clf=RandomForestRegressor(n_estimators=50,max_depth=3,random_state=42,n_jobs=1)
                    clf.fit(train[fs],train[col]); pred[name].append(clf.predict(test[fs]))
            fold_manifest.append({'model':model,'heldout':s,'train_scenes':sorted(train.scene_id.unique().tolist()),'train_count':len(train),'test_count':len(test),'constant_tau':float(constant['threshold']),'constant_T':float(constant['temperature'])})
            for i,r in enumerate(test.itertuples()):
                row={'source_model':model,'scene_id':s,'level':int(r.level),'mean':float(r.mean),
                     'constant_tau':float(constant['threshold']),'constant_T':float(constant['temperature'])}
                for name in pred:
                    row[name+'_tau']=float(pred[name][0][i]); row[name+'_T']=float(pred[name][1][i])
                    assert .1-1e-10<=row[name+'_tau']<=.3+1e-10 and .01-1e-10<=row[name+'_T']<=.1+1e-10
                params.append(row)
    predictions=pd.DataFrame(params); assert len(predictions)==360
    predictions.to_csv(OUT/'predicted_parameters.csv',index=False)
    (OUT/'fold_manifest.json').write_text(json.dumps(fold_manifest,indent=2)+'\n',encoding='utf8')
    print('predictions complete; starting streamed rendering',flush=True)
    prior=pd.read_csv(OLD/'primary_per_pair.csv'); selected=prior[(prior.method.isin(['identity','fixed_global']))]
    lookup={(r.source_model,r.method,r.scene_id,int(r.level)):r for r in selected.itertuples()}
    raw_ref=pd.read_csv(COMP/'results/analysis/real_per_pair.csv')
    zerometa=json.loads((ZERO/'run_metadata.json').read_text(encoding='utf8'))
    cfg=json.loads((COMP/'protocol.json').read_text(encoding='utf8')); source_root=Path(cfg['execution']['output_data_root'])
    records=[]
    for s in scene_ids:
        guard(); archive=ZERO/'outputs'/f'{s}.npz'; assert sha(archive)==zerometa['output_npz_sha256'][archive.name]
        with np.load(archive,allow_pickle=False) as zero:
            for level in range(1,11):
                guard(); rawold=lookup['raw','identity',s,level]
                x=read_image(ROOT/'validation/input'/f'{s}-{level}.png',str(feat[(feat.scene_id==s)&(feat.level==level)].iloc[0].input_sha256))
                assert ahash(x)==rawold.input_hash
                rr=raw_ref[(raw_ref.model=='raw')&(raw_ref.scene_id==s)&(raw_ref.level==level)].iloc[0]
                ref=read_image(ROOT/'validation/gt'/f'{s}.png',str(rr.reference_sha256))
                mu=float((x@np.array([.2126,.7152,.0722],dtype=np.float32)).mean())
                for model in sorted(tab.source_model.unique()):
                    old=lookup[model,'identity',s,level]; fixed=lookup[model,'fixed_global',s,level]
                    if model=='ZeroDCEpp_official': y=zero[f'level_{level:02d}']
                    else:
                        yp=source_root/model/'real'/f'{s}-{level}.npy'; proof=json.loads(yp.with_suffix('.json').read_text(encoding='utf8'))
                        assert proof['input_hash']==ahash(x); y=np.load(yp,allow_pickle=False); assert proof['output_hash']==ahash(y)
                    assert y.dtype==np.float32 and y.shape==x.shape
                    y=np.clip(y,0,1); assert ahash(y)==old.source_hash
                    pr=predictions[(predictions.source_model==model)&(predictions.scene_id==s)&(predictions.level==level)].iloc[0]
                    assert abs(mu-float(pr['mean']))<1e-6
                    variants=[('raw',x,None,None),('original',y,None,None),('fixed',render(x,y,alpha(mu,.18,.04)),.18,.04)]
                    for nm in ['constant','rf_mean','rf_five']:
                        t=float(pr[nm+'_tau']); T=float(pr[nm+'_T']); variants.append((nm,render(x,y,alpha(mu,t,T)),t,T))
                    for nm,img,t,T in variants:
                        mse,psnr=metric(img,ref)
                        if nm=='raw': assert abs(mse-float(rawold.mse))<1e-12
                        if nm=='original': assert abs(mse-float(old.mse))<1e-12
                        if nm=='fixed': assert abs(mse-float(fixed.mse))<1e-12 and ahash(img)==fixed.output_hash
                        records.append({'source_model':model,'scene_id':s,'level':level,'method':nm,'tau':t,'T':T,'mse':mse,'psnr_db':psnr,'output_hash':ahash(img)})
        print('rendered',s,'rows',len(records),flush=True)
    assert len(records)==2160
    result=pd.DataFrame(records); result.to_csv(OUT/'per_pair.csv',index=False)
    result.groupby(['source_model','method'],as_index=False).psnr_db.mean().to_csv(OUT/'overall.csv',index=False)
    result.groupby(['source_model','method','level'],as_index=False).psnr_db.mean().to_csv(OUT/'by_level.csv',index=False)
    pivot=result.pivot(index=['source_model','scene_id','level'],columns='method',values='psnr_db').reset_index()
    for nm in ['constant','rf_mean','rf_five']:
        pivot[nm+'_minus_fixed']=pivot[nm]-pivot.fixed
        pivot[nm+'_minus_original']=pivot[nm]-pivot.original
    pivot['rf_five_minus_rf_mean']=pivot.rf_five-pivot.rf_mean
    pivot.to_csv(OUT/'paired.csv',index=False)
    scene=pivot.groupby(['source_model','scene_id'],as_index=False)[['constant_minus_fixed','rf_mean_minus_fixed','rf_five_minus_fixed','rf_five_minus_rf_mean']].mean()
    scene.to_csv(OUT/'scene_deltas.csv',index=False)
    meta={'grid_sha256':sha(grid_path),'features_sha256':sha(FEATURES),'targets_sha256':sha(LABELDIR/'smooth_targets_005db.csv'),'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'folds':len(fold_manifest),'rows':len(result),'seed':42,'n_jobs':1,'forwards':0,'test_access':False}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    print('complete',json.dumps(meta),flush=True)
if __name__=='__main__': main()
