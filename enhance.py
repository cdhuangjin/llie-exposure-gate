"""Apply the archived gate to RGB images or float32 .npy arrays."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab,deltaE_ciede2000
from gate import fuse

def read(path):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        a=np.load(path,allow_pickle=False)
        if not np.issubdtype(a.dtype,np.floating):
            raise ValueError('NPY arrays must be floating point RGB in [0,1]')
        return a.astype(np.float32)
    with Image.open(path) as im:
        if im.mode!='RGB':
            raise ValueError('Provide an 8-bit RGB image; no implicit color conversion')
        return np.asarray(im,dtype=np.float32)/255

def metrics(pred,ref):
    if pred.shape!=ref.shape or min(pred.shape[:2])<7:
        raise ValueError('Reference/output must match and be at least 7x7')
    if not np.isfinite(ref).all() or ref.min()<0 or ref.max()>1:
        raise ValueError('Reference must be finite RGB in [0,1]')
    pred=np.clip(pred,0,1)
    mse=float(np.mean((pred.astype(np.float64)-ref.astype(np.float64))**2))
    lab=rgb2lab(pred.astype(np.float64),illuminant='D65',observer='2')
    rlab=rgb2lab(ref.astype(np.float64),illuminant='D65',observer='2')
    return {'mse':mse,'psnr_db':float(-10*np.log10(mse)) if mse else None,
            'perfect_match':mse==0,
            'ssim':float(structural_similarity(ref,pred,data_range=1,win_size=7,
                 channel_axis=2,gaussian_weights=False,use_sample_covariance=True)),
            'mean_deltaE00':float(deltaE_ciede2000(lab,rlab).mean()),
            'mean_chroma_ab_error':float(np.linalg.norm(lab[...,1:]-rlab[...,1:],axis=-1).mean())}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',required=True);ap.add_argument('--enhanced',required=True)
    ap.add_argument('--out',required=True,help='New .npy (scientific float output) or .png (display output)')
    ap.add_argument('--mode',choices=['global','local'],default='global')
    ap.add_argument('--threshold',type=float,default=.18);ap.add_argument('--temperature',type=float,default=.04)
    ap.add_argument('--reference',help='Optional evaluation only; never enters the gate')
    ap.add_argument('--paper-crop',action='store_true',help='Common top-left 396x600 crop; no resize')
    args=ap.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if out.suffix.lower() not in ['.png','.npy']:raise ValueError('Output extension must be .png or .npy')
    raw=read(args.raw);enh=read(args.enhanced);ref=read(args.reference) if args.reference else None
    if args.paper_crop:
        for a in [raw,enh]+([ref] if ref is not None else []):
            if a.shape[0]<396 or a.shape[1]<600:raise ValueError('Image smaller than paper crop')
        raw=raw[:396,:600];enh=enh[:396,:600]
        if ref is not None:ref=ref[:396,:600]
    result,alpha=fuse(raw,enh,threshold=args.threshold,temperature=args.temperature,global_map=args.mode=='global')
    record={'mode':args.mode,'threshold':args.threshold,'temperature':args.temperature,
            'alpha_mean':float(alpha.mean()),'shape':list(result.shape),'metrics_before_output_quantization':True}
    if ref is not None:record['metrics']=metrics(result,ref)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.suffix.lower()=='.npy':np.save(out,result,allow_pickle=False)
    else:Image.fromarray(np.rint(result*255).astype(np.uint8)).save(out)
    print(json.dumps(record,indent=2,allow_nan=False))
if __name__=='__main__':main()
