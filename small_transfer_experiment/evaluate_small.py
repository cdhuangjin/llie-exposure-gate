"""Re-evaluate the predeclared SMALL test split with three frozen checkpoints.

Obtain upstream source, weights and SMALL separately; see README.md. Results go
to a new output directory. No image, model or parameter is selected by score.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np
import psutil
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate import fuse

PREFIX = 'SMALL_v1.0.0/'
WEIGHT_SHA = {
    'zero': 'ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84',
    'sci': '4ff1e7e66e8f18114f790e5085ca4c936f0858e329e7ad1cb65b8aa09a9c6916',
    'ruas': '28eb6bcb94f62c9dcdb1cc96196ca9c2fd902ea729ff3d9d15e7c8534e9645ac',
}

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_models(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2)
    torch.use_deterministic_algorithms(True)
    weights = {'zero': args.zero_weight, 'sci': args.sci_weight, 'ruas': args.ruas_weight}
    for name, path in weights.items():
        if file_sha(path) != WEIGHT_SHA[name]:
            raise ValueError(f'{name} checkpoint SHA256 mismatch')
    sources = [
        (args.zero_root/'model.py', 'cceac0f63902df17d19eeb2dc2ca1e84bcf6b6ed0751c121a8c3fff5288924f2'),
        (args.sci_root/'model.py', 'f39b47832fcb92c9bbc3ed2c66acbcd2dc6d78aac1a7dbcc35517c3edfd77fc7'),
        (args.sci_root/'loss.py', '585166901a5e4c859a7d1661264ff94c1bae89b6474502385bf0f176e64ca3f4'),
        (args.ruas_root/'model_enhance/model.py', '62229865de84784acc76ed5a115c81500359ef9a4a1f485c73f7e45f4e6bd008'),
        (args.ruas_root/'model_enhance/operations.py', '708258fd4075cdb86df383179ac2cb15256e9f4511e6da48c7b2717712b0f648'),
        (args.ruas_root/'model_enhance/genotypes.py', '89281876e1b159aa781d191d03ab08e37b4e076fa1a141407c240cdf9990ea17'),
    ]
    for path, expected in sources:
        if file_sha(path) != expected:
            raise ValueError('Upstream source SHA256 mismatch: ' + str(path))
    zero = load_module('zero_dce_official', args.zero_root / 'model.py').enhance_net_nopool(scale_factor=12)
    zero.load_state_dict(torch.load(weights['zero'], map_location='cpu', weights_only=True), strict=True)
    previous_loss = sys.modules.pop('loss', None)
    sys.path.insert(0, str(args.sci_root))
    try:
        sci_source = load_module('sci_official', args.sci_root / 'model.py')
    finally:
        sys.path.remove(str(args.sci_root))
        sys.modules.pop('loss', None)
        if previous_loss is not None:
            sys.modules['loss'] = previous_loss
    sci = sci_source.Finetunemodel.__new__(sci_source.Finetunemodel)
    torch.nn.Module.__init__(sci)
    sci.enhance = sci_source.EnhanceNetwork(layers=1, channels=3)
    sci._criterion = sci_source.LossFunction()
    state = torch.load(weights['sci'], map_location='cpu', weights_only=True)
    needed = set(sci.state_dict())
    if not needed.issubset(state) or any(not k.startswith('calibrate.') for k in set(state)-needed):
        raise ValueError('Unexpected SCI checkpoint keys')
    sci.load_state_dict({k: state[k] for k in needed}, strict=True)
    sys.path.insert(0, str(args.ruas_root))
    try:
        from model_enhance.model import Network
        ruas = Network()
    finally:
        sys.path.remove(str(args.ruas_root))
    ruas.load_state_dict(torch.load(weights['ruas'], map_location='cpu', weights_only=True), strict=True)
    nets = [('Zero-DCE++', zero, lambda x: x[0]),
            ('SCI medium', sci, lambda x: x[1]),
            ('RUAS journal', ruas, lambda x: x[0][-1])]
    for _, net, _ in nets:
        net.cpu().eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
    return nets

def load_pair(archive, row, checks):
    arrays, sizes, hashes = [], [], []
    for key in ('low_path', 'reference_path'):
        rel = row[key]
        data = archive.read(PREFIX + rel)
        digest = hashlib.sha256(data).hexdigest()
        if digest != checks[rel]:
            raise ValueError('Image checksum mismatch: ' + rel)
        with Image.open(io.BytesIO(data)) as im:
            sizes.append(im.size)
            rgb = im.convert('RGB')
            rgb.thumbnail((600, 600), Image.Resampling.LANCZOS)
            w, h = rgb.size
            rgb = rgb.crop((0, 0, w // 12 * 12, h // 12 * 12))
            arrays.append(np.asarray(rgb, dtype=np.float32) / np.float32(255))
        hashes.append(digest)
    if sizes[0] != sizes[1] or arrays[0].shape != arrays[1].shape:
        raise ValueError('Pair geometry mismatch: ' + row['pair_key'])
    return arrays[0], arrays[1], hashes, sizes[0]

def psnr(image, reference):
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return -10 * math.log10(mse) if mse else float('inf')

def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    nets = load_models(args)
    with zipfile.ZipFile(args.archive) as z, (args.out / 'per_pair.jsonl').open('x', encoding='utf-8') as f:
        rows = list(csv.DictReader(io.StringIO(z.read(PREFIX + 'metadata/metadata.csv').decode('utf-8-sig'))))
        selected = sorted((r for r in rows if r['split'] == 'test'), key=lambda r:r['pair_key'])
        keys = z.read(PREFIX+'splits/test.txt').decode('utf-8-sig').splitlines()
        if len(selected) != 50 or set(keys) != {r['pair_key'] for r in selected}:
            raise ValueError('Unexpected test split')
        checks = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in
                  z.read(PREFIX+'SHA256SUMS.txt').decode().splitlines() if '  ' in line}
        for i, row in enumerate(selected, 1):
            if psutil.virtual_memory().available < 1024**3:
                raise RuntimeError('Stopped: less than 1 GiB RAM available; partial output retained')
            x, ref, hashes, size = load_pair(z, row, checks)
            record = {'pair_key':row['pair_key'], 'scene_id':row['scene_id'], 'native_size':size,
                      'scoring_shape':list(x.shape), 'input_reference_sha256':hashes,
                      'input_psnr':psnr(x,ref), 'models':{}}
            for name, net, choose in nets:
                with torch.inference_mode():
                    tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)
                    y = choose(net(tensor))[0].permute(1, 2, 0).numpy()
                if y.shape != x.shape or not np.isfinite(y).all():
                    raise ValueError('Invalid model output')
                y = np.clip(y, 0, 1).astype(np.float32)
                gated, alpha = fuse(x, y, global_map=True)
                record['models'][name] = {'output_psnr':psnr(y,ref), 'gated_psnr':psnr(gated,ref),
                                          'alpha':float(alpha[0,0])}
            f.write(json.dumps(record, allow_nan=False)+'\n')
            f.flush()
            print(f'{i}/50 {row["pair_key"]}', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','zero-root','zero-weight','sci-root','sci-weight','ruas-root','ruas-weight','out'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())

if __name__ == '__main__':
    main()
