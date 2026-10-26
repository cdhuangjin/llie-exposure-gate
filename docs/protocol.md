# Reproduction scope and protocol

RGB float32 input scaled to [0,1]; common top-left 396×600 crop. Predictions clipped to [0,1]; no resize, registration, GT-mean normalization, or per-output maximum normalization. RGB MSE accumulated in float64; PSNR = -10 log10(MSE), range 1. SSIM: uniform 7×7, sample covariance, RGB average. Color: assumed sRGB/D65/2-degree observer, mean CIEDE2000 (unit factors), and mean Euclidean Lab ab error.

Overall means average levels within a published scene ID and then equally average scene IDs. 12 IDs are not certified independent physical scenes. Full-frame metrics precede PNG quantization. Source-level array hashes in the score CSVs identify the scientific outputs, but those arrays are not included in this minimal repository.

The default (.18,.04) is retained separately from all grid maxima and calibrated choices. The 210-point grid spans threshold .10–.30 and temperature .01–.10 in .01 steps. Archived grid-by-scene scores enable recalculation of shared LOSO choices. Each fold excludes all ten levels of its held scene and uses equal model/scene weights over the remaining eleven. Exact ties resolve by ascending threshold then temperature.

The original full grid used direct rendered float32 images and float64 MSE. This minimal package reaggregates archived scores; it does not rerender all 75,600 outputs. Its summaries therefore verify numerical consistency and selection rules rather than prove the image-level measurements. `primary_per_pair.csv` contains 6,600 evaluations, not 6,600 independent images. Selected-calibration comparisons contain 1,800 rows, with reused fixed and traditional controls. None adds new scenes.

SCI source revision: `f6f88fd73cd614dbeee17d61a0dbde3678b7e183`; public medium weight SHA-256: `4ff1e7e66e8f18114f790e5085ca4c936f0858e329e7ad1cb65b8aa09a9c6916`.

RUAS journal source revision: `461172c4d05b54155d8f818af0543eca457a1bb5`; lol weight SHA-256: `28eb6bcb94f62c9dcdb1cc96196ca9c2fd902ea729ff3d9d15e7c8534e9645ac`.

Zero-DCE++ uses the official Epoch99 checkpoint. Historical real-image outputs originated on GPU; SCI/RUAS inference and the postprocessing comparisons were on CPU. These results are not a same-device speed comparison.
