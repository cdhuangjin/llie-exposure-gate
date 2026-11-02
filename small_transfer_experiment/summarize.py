"""Convert per-pair JSONL into long-form CSV and equal-pair/scene summaries."""
import argparse
import csv
import json
from pathlib import Path
import statistics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    records=[json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines()]
    if len(records)!=50 or len({r['pair_key'] for r in records})!=50:
        raise ValueError('Expected 50 unique pairs')
    names=['Zero-DCE++','SCI medium','RUAS journal']
    if any(set(r['models'])!=set(names) for r in records):
        raise ValueError('Missing/extra model')
    args.out_dir.mkdir(parents=True,exist_ok=False)
    with (args.out_dir/'per_pair_psnr.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['pair_key','scene_id','model','input_psnr_db','original_psnr_db','gated_psnr_db','delta_db','alpha'])
        for r in records:
            for name in names:
                m=r['models'][name]
                writer.writerow([r['pair_key'],r['scene_id'],name,r['input_psnr'],m['output_psnr'],
                                 m['gated_psnr'],m['gated_psnr']-m['output_psnr'],m['alpha']])
    summaries=[]
    for name in names:
        original=[r['models'][name]['output_psnr'] for r in records]
        gated=[r['models'][name]['gated_psnr'] for r in records]
        groups={s:[r for r in records if r['scene_id']==s] for s in {r['scene_id'] for r in records}}
        summaries.append({'model':name,'original_pair_mean':statistics.mean(original),'gated_pair_mean':statistics.mean(gated),
            'delta_pair_mean':statistics.mean(b-a for a,b in zip(original,gated)),
            'regressions':sum(b<a for a,b in zip(original,gated)),
            'worst_delta':min(b-a for a,b in zip(original,gated)),
            'original_scene_mean':statistics.mean(statistics.mean(r['models'][name]['output_psnr'] for r in g) for g in groups.values()),
            'gated_scene_mean':statistics.mean(statistics.mean(r['models'][name]['gated_psnr'] for r in g) for g in groups.values())})
    with (args.out_dir/'summary.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    report={'pairs':len(records),'scene_ids':len(groups),'input_pair_mean_psnr':statistics.mean(r['input_psnr'] for r in records),
            'models':{s['model']:{k:v for k,v in s.items() if k!='model'} for s in summaries}}
    (args.out_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
