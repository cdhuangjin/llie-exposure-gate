"""Time the global gate only; no model inference or image I/O is timed."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate import fuse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new JSON output file')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261102)
    raw = rng.random((400,600,3),dtype=np.float32)
    enhanced = rng.random((400,600,3),dtype=np.float32)
    for _ in range(10):
        fuse(raw,enhanced,global_map=True)
    samples=[]
    for _ in range(100):
        start=time.perf_counter_ns()
        fuse(raw,enhanced,global_map=True)
        samples.append((time.perf_counter_ns()-start)/1e6)
    result={'cpu':platform.processor(),'platform':platform.platform(),'numpy':np.__version__,
            'opencv':cv2.__version__,'shape_hwc':[400,600,3],'seed':20261102,
            'warmups':10,'iterations':100,'median_ms':float(np.median(samples)),
            'p95_ms':float(np.percentile(samples,95)),'mean_ms':float(np.mean(samples)),
            'all_ms':samples,'threads':1,'scope':'gate.fuse global_map=True; no IO or model'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='all_ms'},indent=2))

if __name__=='__main__':
    main()
