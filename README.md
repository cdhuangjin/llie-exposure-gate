# LLIE Exposure Gate

Minimal code and numerical-result reproduction for **A Training-Free Exposure Gate for Lightweight Low-Light Enhancement: An Empirical Study of Over-Enhancement and Its Trade-offs**. This is research-in-progress code; no journal acceptance is implied.

The gate blends an existing enhancement output with its input using mean encoded-RGB brightness. Default threshold: 0.18; temperature: 0.04. No neural network is trained or downloaded by this package.

## What can be reproduced?

1. Apply the archived global or local gate to your own RGB image/output pairs.
2. Score a supplied output against a supplied reference with the paper's metric settings.
3. Recompute overall scores, regressions, parameter sensitivity, and shared LOSO selection from archived numerical measurements; regenerate exposure and parameter plots.

This is **not** an end-to-end rerun of the three enhancement networks. No dataset pixels, model weights, manuscript, author declarations, credentials, or local machine paths are distributed. Numeric measurements are not a substitute for independent image-level validation.

## Install and reproduce

Python 3.12 is recommended. Use an isolated environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python reproduce.py --out reproduced
```

The exact tested dependency set is in `requirements-lock.txt`; `requirements.txt` lists the direct dependencies. A clean Python 3.12 environment passed dependency checks, four gate/metric tests, and the archived-result reproduction.

Reproduction is offline once dependencies are installed. `results/manifest.json` records file hashes and project-relative provenance. Generated summaries are checked against archived results. Output CSVs and plots go to `reproduced/`.

## Apply to an image pair

```bash
python enhance.py --raw input.png --enhanced model_output.png --out fused.png
python enhance.py --raw input.npy --enhanced model_output.npy --out fused.npy --reference reference.npy --paper-crop
python enhance.py --raw input.png --enhanced model_output.png --out local.png --mode local
```

The CLI defaults to **global** mode. The archived Python function retains its historical local default, so call `fuse(raw, enhanced, global_map=True)` explicitly for global gating. Global brightness uses the spatial mean; Gaussian filtering with sigma=7 and reflected boundaries is only used for the local control. Input coefficients are (0.2126, 0.7152, 0.0722), applied without inverse gamma correction. Enhanced values are clipped before blending.

Input/output NPY arrays must be floating-point H脳W脳3 RGB. PNG images must be 8-bit RGB. No automatic resizing, registration or reference-mean normalization occurs. `--paper-crop` requests the original top-left 396脳600 crop. A reference is optional and used only after gating for scoring. Metrics printed for PNG output are measured before its display quantization; reproducing paper scores requires the original floating-point enhancement outputs. Existing output files are not overwritten.

## Fixed-parameter results

| Checkpoint | Original PSNR | Global gate PSNR | Global gate SSIM |
|---|---:|---:|---:|
| Zero-DCE++ official | 14.538 | 24.837 | 0.8630 |
| SCI public medium | 9.284 | 24.726 | 0.8508 |
| RUAS journal extension | 9.175 | 25.276 | 0.8543 |

The input baseline is 20.043 dB / 0.7732. Population: **120 validation pairs = 12 published scene IDs 脳 10 intensity levels**, previously exposed during development. IDs `Test1`鈥揱Test4` in this release are part of the selected validation group; their names do not imply access to the sealed test split.

## Limits that must accompany reuse

- These are three specific public checkpoints, not original-paper score reproduction or all LLIE models.
- Fixed-gate PSNR regresses relative to original enhancement on 17/10/12 pairs; the worst observed loss is 11.122 dB.
- The default is outside every model's 0.30-dB near-optimal grid region. There is no shared near-optimal point in this grid.
- LOSO is post-hoc reference-assisted calibration on already exposed scenes, not independent validation or zero-calibration performance.
- Lower CIEDE2000 does not imply lower Lab chromaticity error. Input V-gamma 0.8 has better chromaticity error than the fixed gates.
- Real cross-capture reference correspondence is not independently certified. No universal non-degradation or shared causal-mechanism claim is made.

## Data and upstream models

- MILL dataset/paper: https://color.cvc.uab.es/mill/ and https://github.com/CVC-Color/MILL
- Zero-DCE++: https://github.com/Li-Chongyi/Zero-DCE_extension
- SCI: https://github.com/vis-opt-group/SCI (CVPR implementation, public `medium.pt`)
- RUAS: https://github.com/vis-opt-group/RUAS (journal extension, `lol.pt`)

Obtain any needed images or checkpoints from their authors under their original terms. SCI's public medium checkpoint differs from the original paper-table model; the tested RUAS is not the original CVPR architecture. See `docs/protocol.md` and `docs/environment.json` for scoring and environment details.

## Software assistance and release status

OpenAI Codex assisted with code, result analysis, and documentation. Numerical values derive from retained executable experiment records. The package includes verification checks; it does not claim external independent replication. Repository: https://github.com/cdhuangjin/llie-exposure-gate. No DOI or journal acceptance is asserted. The original software is distributed under the MIT License; third-party terms are described in `NOTICE.md`.
