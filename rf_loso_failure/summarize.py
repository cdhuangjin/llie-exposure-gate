"""Validate and summarize archived LOSO scores without image-level inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent / 'archived_results'
d = pd.read_csv(root / 'per_pair.csv')
assert len(d) == 2160 and not d.duplicated(['source_model', 'scene_id', 'level', 'method']).any()
assert np.allclose(d.psnr_db, -10 * np.log10(d.mse), atol=1e-10, rtol=0)
summary = d.groupby(['source_model', 'method']).psnr_db.mean().sort_index()
archived = pd.read_csv(root / 'overall.csv').set_index(['source_model', 'method']).psnr_db.sort_index()
assert np.allclose(summary, archived, atol=1e-12, rtol=0)
print(summary.unstack().round(4).to_string())
print('PASS: archived numerical summaries; no image rerender or model refit')
