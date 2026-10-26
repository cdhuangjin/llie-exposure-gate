"""Rebuild numerical summaries and plots from archived measurements, without images."""
from pathlib import Path
import argparse,csv,hashlib,json,collections
import numpy as np

ROOT=Path(__file__).resolve().parent
MODELS=['ZeroDCEpp_official','SCI_CVPR_medium','RUAS_journal_lol']
NAMES=['Zero-DCE++','SCI medium','RUAS journal']
METRICS=['mse','psnr_db','ssim','mean_deltaE00','mean_chroma_ab_error']
def read(n):return list(csv.DictReader((ROOT/'results'/n).open(encoding='utf-8-sig')))
def grouped(rows,keys):
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    return groups
def mean(rows,k):return float(np.mean([float(r[k]) for r in rows]))
def scene_mean(rows,k):return float(np.mean([mean(g,k) for g in grouped(rows,['scene_id']).values()]))
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',default='reproduced');ap.add_argument('--no-plots',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for m in json.loads((ROOT/'results/manifest.json').read_text()):
        assert hashlib.sha256((ROOT/m['file']).read_bytes()).hexdigest()==m['sha256'],m['file']
    p=read('primary_per_pair.csv');expected=read('primary_by_group.csv')
    assert len(p)==6600 and len({(r['source_model'],r['method'],r['scene_id'],r['level']) for r in p})==6600
    summary=[]
    for (model,method),g in grouped(p,['source_model','method']).items():
        e=next(x for x in expected if x['source_model']==model and x['method']==method and x['group']=='overall')
        row={'source_model':model,'method':method,**{k:scene_mean(g,k) for k in METRICS}}
        for k in METRICS:assert np.isclose(row[k],float(e[k]),rtol=0,atol=1e-10),(model,method,k)
        summary.append(row)
    for r in p:
        assert np.isclose(float(r['psnr_db']),-10*np.log10(float(r['mse'])),atol=1e-10,rtol=0)
    write_csv(out/'overall_recomputed.csv',summary)
    regressions=[]
    for model in MODELS:
        g=[r for r in p if r['source_model']==model and r['method']=='fixed_global']
        regressions.append({'model':model,'below_original':sum(float(r['delta_PSNR_vs_source'])<0 for r in g),
            'below_input':sum(float(r['delta_PSNR_vs_raw'])<0 for r in g),
            'worst_change_vs_original':min(float(r['delta_PSNR_vs_source']) for r in g)})
    write_csv(out/'regressions_recomputed.csv',regressions)
    grid=read('grid_by_scene.csv');scenes=sorted({r['scene_id'] for r in grid})
    points=sorted({(float(r['threshold']),float(r['temperature'])) for r in grid})
    assert len(scenes)==12 and len(points)==210 and len(grid)==3*12*210
    cube=np.empty((3,12,210))
    for r in grid:cube[MODELS.index(r['source_model']),scenes.index(r['scene_id']),points.index((float(r['threshold']),float(r['temperature'])))]=float(r['psnr_db'])
    folds=json.loads((ROOT/'results/selections.json').read_text())['folds']
    choices=[]
    for i,s in enumerate(scenes):
        train=[j for j in range(12) if j!=i]
        selected=points[int(np.argmax(cube[:,train,:].mean(axis=(0,1))))]
        assert selected==tuple(folds[s]['shared_gate'])
        assert s not in folds[s]['train_scene_ids'] and len(folds[s]['train_scene_ids'])==11
        choices.append({'held_scene':s,'threshold':selected[0],'temperature':selected[1]})
    write_csv(out/'shared_LOSO_choices.csv',choices)
    sel=read('selection_per_pair.csv');selexp=read('selection_overall.csv');loso=[]
    for (model,method),g in grouped(sel,['evaluation_model','diagnostic_method']).items():
        e=next(r for r in selexp if r['evaluation_model']==model and r['diagnostic_method']==method)
        row={'evaluation_model':model,'diagnostic_method':method,**{k:scene_mean(g,k) for k in METRICS}}
        for k in METRICS:assert np.isclose(row[k],float(e[k]),rtol=0,atol=1e-10)
        loso.append(row)
    write_csv(out/'LOSO_overall_recomputed.csv',loso)
    sensitivity=[]
    for i,model in enumerate(MODELS):
        v=cube[i].mean(axis=0);gap=float(v.max()-v[points.index((.18,.04))]);near=int(np.count_nonzero(v>=v.max()-.3-1e-10))
        assert near==[9,7,4][i] and gap>.3
        sensitivity.append({'model':model,'default_gap_db':gap,'near_top_points':near})
    write_csv(out/'sensitivity_recomputed.csv',sensitivity)
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
        levels=read('real_by_level.csv');fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        for model,name in [('raw','Input')]+list(zip(MODELS,NAMES)):
            g=sorted([r for r in levels if r['model']==model],key=lambda r:int(r['level']))
            ax.plot([int(r['level']) for r in g],[float(r['psnr_db']) for r in g],'-o',label=name)
        ax.set(xlabel='MILL intensity level',ylabel='Scene-equal mean PSNR (dB)');ax.legend();fig.savefig(out/'exposure_profiles.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
        for i,ax in enumerate(axes):
            v=cube[i].mean(axis=0).reshape(21,10).T
            im=ax.imshow(v,origin='lower',aspect='auto',extent=[.095,.305,.005,.105],vmin=15,vmax=27)
            ax.scatter(.18,.04,marker='*',c='red',edgecolor='white',s=80);ax.set(title=NAMES[i],xlabel='Threshold')
        axes[0].set_ylabel('Temperature');fig.colorbar(im,ax=axes,label='Mean PSNR (dB)');fig.savefig(out/'parameter_grid.pdf');plt.close(fig)
    report={'status':'PASS','source_hashes_verified':True,'primary_rows':len(p),'selection_rows':len(sel),'grid_scene_rows':len(grid),
            'LOSO_folds_verified':len(choices),'scope':'Archived-score aggregation and gate application; not independent image/model rerun','no_network_access':True}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
