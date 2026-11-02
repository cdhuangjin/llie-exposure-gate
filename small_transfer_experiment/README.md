# SMALL fixed-gate transfer and CPU timing

This directory records a negative transfer result for the unchanged `(tau,T)=(0.18,0.04)` global gate. It contains the frozen protocol, 50-pair numeric records, 150 model-level PSNR rows, summary, timing samples and executable scripts. It does not distribute image pixels, checkpoints or third-party code.

| Checkpoint | Original pair-mean PSNR | Gated pair-mean PSNR | Delta (dB) | Regressions |
|---|---:|---:|---:|---:|
| Zero-DCE++ | 9.040 | 8.785 | -0.255 | 50/50 |
| SCI medium | 8.262 | 8.168 | -0.094 | 50/50 |
| RUAS journal | 7.512 | 7.464 | -0.048 | 48/50 |

Input pair-mean PSNR is 5.693 dB. The release uses all 50 test pairs (37 scene IDs) of [SMALL v1.0.0](https://doi.org/10.17632/vnz4gcw8wk.1), not used in selecting the gate in this study. Inputs and references are resized with aspect ratio preserved to maximum dimension 600 using Pillow Lanczos, then cropped at the lower/right edges to multiples of 12; see `PROTOCOL.md`. The two separately captured images are not certified pixel-aligned. Scene ID independence and full training exposure of the pretrained models have not been established. Results are descriptive and are not official native-resolution benchmark scores.

## Reproduce numeric summaries without datasets or PyTorch

From the repository root with Python 3.12:

```sh
python small_transfer_experiment/summarize.py --input small_transfer_experiment/results/per_pair.jsonl --out-dir outputs/small_summary
```

The output directory must not already exist. `results/per_pair_psnr.csv` has one row per pair and checkpoint; the JSONL also includes input/reference checksums, image dimensions and gate weights. The archived `summary.json` matches `summary.csv` and the paper. These are numeric outputs, not images.

## Rerun inference

Obtain the authors' [SMALL v1.0.0 zip](https://data.mendeley.com/datasets/vnz4gcw8wk/1) (CC BY 4.0); its expected SHA-256 is `2044f537a752f0b2b52f534584062e584bda6e22ed7ba04b33abb984731c80d2`. The runner uses the release's image checksums. Obtain source and checkpoints separately under their original licenses:

* [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension), revision `09f202b690f82da939b8e6ec8535960ae97ad8bd`, source folder `Zero-DCE++`, checkpoint `snapshots_Zero_DCE++/Epoch99.pth`.
* [SCI](https://github.com/vis-opt-group/SCI), revision `f6f88fd73cd614dbeee17d61a0dbde3678b7e183`, source folder `CVPR`, checkpoint `weights/medium.pt`. The upstream README says this checkpoint differs from the model used for the original paper tables.
* [RUAS](https://github.com/vis-opt-group/RUAS), revision `461172c4d05b54155d8f818af0543eca457a1bb5`, journal-extension source root and `weights/lol.pt`.

The runner checks the checkpoint and inference-source SHA-256 values before inference and uses `weights_only=True`. The original run used Python 3.12, PyTorch `2.13.0+cu130` on CPU, NumPy `2.4.4`, Pillow `12.2.0`, OpenCV `4.12.0`, and psutil `7.2.2`; install appropriate builds for your platform. The root `requirements-lock.txt` is for the earlier archived-summary workflow and does not contain PyTorch or match this later run's NumPy version. A separate clean environment and other platforms have not been tested.

```sh
python small_transfer_experiment/evaluate_small.py \
  --archive /path/to/SMALL_v1.0.0.zip \
  --zero-root /path/to/Zero-DCE_extension/Zero-DCE++ \
  --zero-weight /path/to/Epoch99.pth \
  --sci-root /path/to/SCI/CVPR --sci-weight /path/to/medium.pt \
  --ruas-root /path/to/RUAS --ruas-weight /path/to/lol.pt \
  --out outputs/small_run
python small_transfer_experiment/summarize.py --input outputs/small_run/per_pair.jsonl --out-dir outputs/small_run_summary
```

The first command requires an output path that does not already exist and retains partial records on failure. On the original machine, a rerun of all 50 pairs using this portable runner, including its source-hash guards, matched all archived output PSNR, gate PSNR, and alpha values exactly. This remains an internal verification, not independent replication. Other hardware and library versions may differ numerically.

## Gate-only CPU timing

```sh
python small_transfer_experiment/benchmark_cpu.py --out outputs/gate_cpu_timing.json
```

`results/cpu_timing.json` contains all 100 observed timings on the original Intel Core Ultra 7 270K Plus desktop, after ten warmups, using float32 synthetic 400×600 RGB arrays. Median 9.3874 ms and 95th percentile 10.4492 ms include validation, luminance, allocation, blending and clipping, but exclude model inference and file I/O. A second run using the portable script yielded median 11.6282 ms and 95th percentile 13.2314 ms; its 100 samples are in `results/cpu_timing_repeat.json`. The machine was shared with other workloads; timing is illustrative, not mobile or end-to-end latency. The timing script reproduces the measurement procedure, not the exact observed latency.
