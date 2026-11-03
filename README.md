# EdgeLight-GAN

**An edge aware lightweight multiresolution generative adversarial network for low light image enhancement on resource constrained devices**

T. Kavitha¹\*, Dr. A. Kathirvel²
¹ Research Scholar, Saveetha Engineering College, Chennai, India
² Department of Computer Science and Engineering, Saveetha Engineering College, Chennai, India
\* Correspondence: kavithat@saveetha.ac.in

[![CI](https://github.com/iitrky/EdgeLight-GAN_ScientificReports_GitHub/actions/workflows/ci.yml/badge.svg)](https://github.com/iitrky/EdgeLight-GAN_ScientificReports_GitHub/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23917544.svg)](https://doi.org/10.5281/zenodo.23917544)


This repository is the official code companion to the manuscript submitted to
*Scientific Reports* (revision R2). It provides a complete, working PyTorch
implementation of **EdgeLight-GAN**: model definitions, training and
evaluation scripts, dataset loaders, loss functions, and the computational
benchmarking utilities used to produce the paper's reported parameter counts,
FLOPs, latency, FPS, memory and power measurements.

---

## Table of contents

- [Abstract](#abstract)
- [Headline results](#headline-results)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Single-image / folder inference](#single-image--folder-inference)
  - [Computational benchmarking](#computational-benchmarking)
- [Results](#results)
- [Ablation studies](#ablation-studies)
- [Computational efficiency](#computational-efficiency)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Citation](#citation)
- [Code availability statement](#code-availability-statement)
- [License](#license)

---

## Abstract

Low-light image enhancement (LLIE) is important for reliable visual
perception because insufficient illumination can reduce visibility,
structural contrast, color fidelity, and perceptual quality. Existing
enhancement methods may improve brightness while increasing noise,
degrading edges, or introducing substantial computational overhead. This
study presents EdgeLight-GAN, a lightweight multi-resolution generative
adversarial framework that integrates fine-, mid-, and global-context
feature extraction with edge-aware feature fusion, residual attention,
adaptive feature processing, learnable channel gating, illumination
reconstruction, and multi-scale adversarial supervision. The principal
methodological contribution is a coordinated edge-conditioned
resource-control pathway that couples structural feature emphasis with
adaptive feature-channel modulation, rather than claiming novelty for
established operators individually. On the principal LOL evaluation,
EdgeLight-GAN achieves **31.92 dB PSNR, 0.956 SSIM, 0.087 LPIPS, and 2.81
NIQE**. Implementation auditing identifies 0.748 M generator parameters and
0.616 M discriminator parameters (1.364 M total training parameters). A
generator-only PyTorch profiler measurement at 256×256 resolution gives
approximately 89.32 GFLOPs, with a hardware-specific inference measurement
of 17.5 ms/image (~57 FPS). Deployment feasibility is additionally evaluated
on Raspberry Pi 5, Jetson Xavier NX, and Jetson Orin Nano. Medical-image
experiments are interpreted strictly as image-quality enhancement and do
not establish diagnostic efficacy or clinical safety.

## Headline results

| Metric (LOL dataset) | EdgeLight-GAN | Best prior baseline (Retinexformer) |
|---|---|---|
| PSNR ↑ | **31.92 dB** | 31.12 dB |
| SSIM ↑ | **0.956** | 0.953 |
| LPIPS ↓ | **0.087** | 0.091 |
| NIQE ↓ | **2.81** | 2.86 |
| Parameters ↓ | **1.364 M** (0.748 M generator-only) | 9.76 M |
| Generator FLOPs (256×256) | **≈89.32 GFLOPs** | — |
| Inference speed | **≈57 FPS** (17.5 ms/image, RTX 4090) | 12 FPS |
| Inference memory ↓ | **198 MB** | 1784 MB |
| Avg. power draw ↓ | **12 W** | 51 W |

All figures above are hardware/software-specific measurements reported by
the paper (Sec. 6.7) and reproduced in this repository via
[`scripts/benchmark.py`](scripts/benchmark.py) -- they are not universal
model properties. See [Computational efficiency](#computational-efficiency)
for the full comparison tables.

## Architecture

EdgeLight-GAN (Sec. 3) is composed of five coordinated modules:

1. **Multi-Resolution Feature Extraction Module** (`models/feature_extraction.py`) --
   fine-scale (edges/textures), mid-scale (dilated-conv object contours), and
   global-context (pooled illumination/scene) branches, fused with an
   edge-aware attention step.
2. **Lightweight Generator Network** (`models/generator.py`) --
   depthwise-separable convolutions, residual attention blocks, edge-aware
   fusion, and a feature-compression/decoder pipeline.
3. **Learnable Channel Gating / Adaptive Feature Resource Control**
   (`models/blocks.py::ChannelGating`, Sec. 3.5 & Eq. 21-24) -- a
   squeeze-and-excitation-style gate that modulates channel responses by
   learned importance while preserving a residual pathway. The paper is
   explicit that this performs *feature-response modulation*, not physical
   parameter pruning or dynamic kernel replacement (Sec. 3.5.1, 4.9, 7).
4. **Multi-Scale Discriminator Network** (`models/discriminator.py`) --
   local (PatchGAN-style), global, and texture-preservation branches fused
   into a single adversarial score. **Training-only**: the discriminator is
   never executed at inference time (Sec. 3.6).
5. **Illumination Reconstruction Module** (`models/blocks.py::IlluminationReconstruction`) --
   estimates a learnable illumination map applied to the reconstructed
   reflectance to produce the final enhanced image (Sec. 3.9).

The implementation follows the per-layer configuration audited in Table 3 of
the paper (kernel sizes, channel widths, number of layers per stage). With
the default configuration, parameter counts closely match the paper's
audited figures:

| Component | Paper (Table 4) | This implementation |
|---|---|---|
| Generator | 0.748 M | 0.738 M |
| Discriminator | 0.616 M | 0.625 M |
| **Total** | **1.364 M** | **1.363 M** |

Verify on your own machine with:

```bash
python -c "
from models import EdgeLightGenerator, MultiScaleDiscriminator
g, d = EdgeLightGenerator(), MultiScaleDiscriminator()
print('generator (M):', sum(p.numel() for p in g.parameters())/1e6)
print('discriminator (M):', sum(p.numel() for p in d.parameters())/1e6)
"
```

The hybrid training objective (`models/losses.py`, Sec. 4.3-4.8) combines
six terms: reconstruction (L1), adversarial (standard GAN, Eq. 11-12), VGG perceptual,
structural-similarity, Sobel edge-preservation, and illumination-consistency
losses.

## Repository structure

```
EdgeLight-GAN/
├── configs/
│   ├── default.yaml          # full training configuration (Sec. 5.1-5.4)
│   └── smoke_test.yaml       # tiny config used by CI / local smoke tests
├── data/
│   ├── datasets.py           # LOL / LOL-v2 / ExDark / SID / paired-folder loaders
│   └── __init__.py
├── models/
│   ├── blocks.py             # depthwise-sep conv, residual attention, channel gating, edge fusion, illumination module
│   ├── feature_extraction.py # multi-resolution (fine/mid/global) extraction module
│   ├── generator.py          # full EdgeLight-GAN generator
│   ├── discriminator.py      # multi-scale discriminator (local/global/texture)
│   ├── losses.py             # hybrid loss (rec/adv/perceptual/ssim/edge/illumination)
│   └── __init__.py
├── utils/
│   ├── common.py             # seeding, checkpointing, logging
│   ├── metrics.py            # PSNR, SSIM, LPIPS, NIQE, BRISQUE
│   ├── profiling.py          # parameter/FLOP/latency/FPS/memory benchmarking
│   └── __init__.py
├── scripts/
│   ├── train.py               # adversarial training loop
│   ├── evaluate.py            # quantitative evaluation (Tables 5-11)
│   ├── inference.py           # single image / folder enhancement (generator-only)
│   └── benchmark.py           # computational efficiency report (Tables 21-27)
├── tests/
│   └── test_models.py         # shape / forward-pass / loss unit tests
├── .github/workflows/ci.yml   # GitHub Actions: unit tests + training smoke test
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/iitrky/EdgeLight-GAN_ScientificReports_GitHub.git
cd EdgeLight-GAN

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or: pip install -e .
```

Tested with Python 3.9-3.11 and PyTorch 2.x. The paper's original
experiments (Sec. 5.1) used Python 3.11, PyTorch 2.2, CUDA 12.1, cuDNN 8.9
on an Intel Core i9-13900K / RTX 4090 (24 GB) workstation; this
implementation targets any CUDA GPU, Apple Silicon (MPS), or CPU-only
environment, with edge deployment additionally validated on Raspberry Pi 5,
Jetson Xavier NX, and Jetson Orin Nano platforms in the paper.

Run the test suite to verify your installation:

```bash
python -m pytest tests/ -v
# or, without pytest:
python tests/test_models.py
```

## Dataset preparation

The paper evaluates on the following public benchmarks (Sec. 5.2, "Data
Availability Statement"):

| Dataset | Link | Notes |
|---|---|---|
| LOL | https://daooshee.github.io/BMVC2018website/ | 500 paired images (485 train / 15 test) |
| LOL-v2 | https://github.com/flyywh/CVPR-2020-Semi-Low-Light | Real + synthetic subsets |
| ExDark | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | Unpaired, no-reference evaluation only |
| SID (See-in-the-Dark) | https://github.com/cchen156/Learning-to-See-in-the-Dark | Raw sensor, extreme low-light |

The medical-imaging, remote-sensing, and surveillance collections used for
the application-oriented experiments (Sec. 5.2.5-5.2.7) are not
redistributed by this repository; substitute your own licensed collections
organized in the same folder layout.

All loaders in `data/datasets.py` expect:

```
<dataset_root>/
├── train/
│   ├── low/    *.png            # degraded input images
│   └── high/   *.png            # paired ground-truth images (omit for unpaired sets)
├── val/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

Re-organize the officially downloaded datasets into this structure (a
`scripts/prepare_dataset.py` helper is a good place to add dataset-specific
conversion logic if the official release layout differs).

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

Edit `configs/default.yaml` to point `data.train_root` / `data.val_root` at
your prepared dataset. Key hyperparameters (Sec. 5.4): Adam optimizer,
initial learning rate `1e-4` (selected via the sensitivity sweep in Table
18), batch size 16, 300 epochs, cosine-annealing schedule, fixed seed 42.
Checkpoints (`best.pt`, periodic `checkpoint_epochN.pt`, and `final.pt`) are
written to `train.output_dir`.

Resume a run with:

```bash
python scripts/train.py --config configs/default.yaml --resume runs/edgelight_gan_lol/checkpoint_epoch50.pt
```

### Evaluation

Reproduce the quantitative tables of Sec. 6.1 (PSNR/SSIM/LPIPS on paired
sets; NIQE/BRISQUE additionally for no-reference sets such as ExDark):

```bash
python scripts/evaluate.py \
    --checkpoint runs/edgelight_gan_lol/best.pt \
    --data_root ./datasets/LOL/test \
    --dataset lol \
    --output_json results/lol_test.json
```

LPIPS/NIQE/BRISQUE require the optional `lpips` and `pyiqa` packages
(already listed in `requirements.txt`); PSNR/SSIM have no extra
dependencies.

### Single-image / folder inference

Only the trained generator is required at inference time (Sec. 3.6, 5.5):

```bash
python scripts/inference.py \
    --checkpoint runs/edgelight_gan_lol/best.pt \
    --input path/to/low_light_image.png \
    --output_dir ./outputs
# or run on an entire folder:
python scripts/inference.py --checkpoint runs/edgelight_gan_lol/best.pt \
    --input path/to/folder --output_dir ./outputs
```

### Computational benchmarking

Reproduce the Sec. 6.7 measurement protocol (parameter counts, generator-only
FLOPs, latency, FPS, peak memory) on your own hardware:

```bash
python scripts/benchmark.py --checkpoint runs/edgelight_gan_lol/best.pt --device cuda
```

Numbers will differ from the paper's Table 21-27 unless run on identical
hardware/software (Sec. 6.7, 7.1) -- this script reproduces the
*measurement procedure*, not the exact reported values. Edge-device figures
(Raspberry Pi 5 / Jetson Xavier NX / Jetson Orin Nano, Table 28) require
running this same script directly on those boards.

## Results

### LOL dataset (Table 5)

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ |
|---|---|---|---|---|
| LLNet | 20.12 | 0.792 | 0.281 | 5.82 |
| MBLLEN | 22.84 | 0.841 | 0.236 | 5.14 |
| Zero-DCE | 25.43 | 0.892 | 0.174 | 4.28 |
| EnlightenGAN | 26.91 | 0.911 | 0.152 | 3.94 |
| EFINet | 28.34 | 0.932 | 0.121 | 3.42 |
| LLFlow | 29.85 | 0.944 | 0.102 | 3.08 |
| PairLIE | 30.24 | 0.947 | 0.098 | 2.97 |
| DNF | 30.71 | 0.950 | 0.094 | 2.91 |
| Retinexformer | 31.12 | 0.953 | 0.091 | 2.86 |
| **EdgeLight-GAN (proposed)** | **31.92** | **0.956** | **0.087** | **2.81** |

### Other benchmarks

| Dataset | PSNR ↑ | SSIM ↑ | Other |
|---|---|---|---|
| LOL-v2 (Table 6) | 30.74 dB | 0.948 | LPIPS 0.093, NIQE 2.82 |
| ExDark (Table 8, no-reference) | -- | -- | NIQE 2.81, BRISQUE 17.46 |
| SID (Table 9) | 33.22 dB | 0.947 | -- |
| Medical imaging (Table 10) | 32.67 dB | 0.953 | LPIPS 0.074 |
| Remote sensing (Table 11) | 31.76 dB | 0.951 | NIQE 2.71 |
| Surveillance (Table 12) | 31.58 dB | 0.949 | BRISQUE 17.24 |

Full per-dataset breakdowns and baseline comparisons are in Sec. 6.1 of the
manuscript; the corresponding evaluation config for each dataset can be
built from `configs/default.yaml` by changing `data.dataset`/`data.*_root`.

## Ablation studies

Component-wise contributions on the LOL dataset (Sec. 6.5):

| Configuration | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Single-scale learning | 29.84 | 0.936 | 0.118 |
| + Dual-scale learning | 30.76 | 0.946 | 0.101 |
| + Multi-resolution learning (full) | 31.92 | 0.956 | 0.087 |
| Without edge-aware optimization | 30.58 | 0.944 | 0.106 |
| Basic edge fusion (no attention) | 31.17 | 0.950 | 0.095 |
| Smart edge optimization (full) | 31.92 | 0.956 | 0.087 |
| Without attention | 30.43 | 0.942 | 0.109 |
| Channel attention only | 31.08 | 0.949 | 0.098 |
| Residual attention (full) | 31.92 | 0.956 | 0.087 |
| Without edge loss | 30.86 | 0.947 | 0.101 |
| Without illumination loss | 31.12 | 0.950 | 0.096 |
| **Full model** | **31.92** | **0.956** | **0.087** |

Learning-rate sensitivity (Table 18):

| Learning rate | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|
| 1e-5 | 30.84 | 0.947 |
| 5e-5 | 31.47 | 0.952 |
| **1e-4 (selected)** | **31.92** | **0.956** |
| 5e-4 | 31.28 | 0.950 |
| 1e-3 | 30.61 | 0.944 |

## Computational efficiency

FPS / latency / parameters / power (Tables 21, 22, 26, 29):

| Method | FPS ↑ | Inference time (ms) ↓ | Parameters (M) ↓ | Power (W) ↓ |
|---|---|---|---|---|
| LLNet | 48 | 20.8 | 4.31 | 18 |
| Zero-DCE | 42 | 23.6 | 0.79 | 15 |
| EnlightenGAN | 28 | 35.7 | 8.64 | 28 |
| LLFlow | 22 | 45.5 | 11.83 | 42 |
| DNF | 18 | 55.6 | 9.14 | 35 |
| Retinexformer | 12 | 83.3 | 9.76 | 51 |
| **EdgeLight-GAN** | **57** | **17.5** | **1.364** | **12** |

Edge-device deployment (Table 28):

| Device | FPS ↑ | Memory (MB) ↓ | Power (W) ↓ |
|---|---|---|---|
| Raspberry Pi 5 | 18 | 245 | 6.2 |
| Jetson Xavier NX | 42 | 212 | 12.8 |
| Jetson Orin Nano | 57 | 198 | 9.5 |

All figures are implementation- and hardware-specific measurements (Sec.
6.7, 7.1) and should not be treated as universal, hardware-independent
model properties.

## Reproducibility notes

- **Seed / protocol**: fixed random seed 42, three repeated runs averaged
  where applicable (Sec. 5.1, 5.6), 300-epoch training with periodic
  checkpointing and best-model selection on the validation split.
- **Training settings**: `configs/default.yaml` uses the values stated in Sec. 5.4 of the revised manuscript: Adam with initial learning rate 1e-4 (selected in Table 18), betas 0.5 and 0.999, batch size 16, 300 epochs, cosine annealing, and loss coefficients 1.0 (reconstruction), 0.1 (adversarial), 0.5 (perceptual), 0.3 (SSIM), 0.4 (edge) and 0.2 (illumination). The adversarial loss is the standard GAN loss of Eq. (11)-(12) (`adversarial_mode: vanilla`).
- **Parameter audit**: this implementation's parameter counts (0.738 M
  generator / 0.625 M discriminator / 1.363 M total) closely match the
  paper's audited figures (0.748 M / 0.616 M / 1.364 M, Table 4) under the
  default configuration in `configs/default.yaml`, which mirrors the
  per-layer channel/kernel widths audited in Table 3. Minor differences are
  expected since not every implementation detail (e.g. exact discriminator
  fully-connected fusion width) is fully specified in the manuscript text.
- **FLOPs / latency / memory / power**: reproduced via
  `scripts/benchmark.py` using `torch.profiler`; absolute values are
  hardware- and software-dependent (Sec. 6.7.3-6.7.6) and will differ from
  the paper's RTX 4090 / Raspberry Pi 5 / Jetson measurements when run on
  different hardware.
- **Channel gating semantics**: the paper is explicit (Sec. 3.5.1, 4.9, 7)
  that the "resource control" mechanism performs *learnable feature-channel
  modulation*, not literal runtime parameter pruning, dynamic kernel
  replacement, or TensorRT/ONNX-based acceleration. `models/blocks.py`
  implements it accordingly as a squeeze-and-excite-style gate with a
  residual pathway (Eq. 21-24).

## Limitations

As discussed in Sec. 7.1 of the paper: performance may degrade under
extremely severe underexposure, heavy sensor noise, strong motion blur, or
degradation patterns not represented in training data; edge-device
measurements are specific to the tested Raspberry Pi 5 / Jetson Xavier NX /
Jetson Orin Nano configurations; the evaluation focuses on image-quality
metrics rather than downstream detection/recognition/segmentation
performance; and the medical-imaging experiments evaluate visual
enhancement only and must **not** be interpreted as evidence of clinical
diagnostic efficacy or patient-level safety.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{kavitha2026edgelightgan,
  title   = {An edge aware lightweight multiresolution generative adversarial network
             for low light image enhancement on resource constrained devices},
  author  = {Kavitha, T. and Kathirvel, A.},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Manuscript revision R2},
  url     = {https://github.com/iitrky/EdgeLight-GAN_ScientificReports_GitHub},
  doi     = {10.5281/zenodo.23917544}
}
```

The `doi` field above is the Zenodo software DOI. Update `year`, `volume`, and `pages` once the article is published.

## Code availability statement

> The source code, model configuration files, dataset-loader configurations,
> evaluation scripts, reproducibility utilities, and deployment benchmarking
> scripts for EdgeLight-GAN are publicly available in the project
> repository **`https://github.com/iitrky/EdgeLight-GAN_ScientificReports_GitHub`** and are
> archived on Zenodo at **`https://doi.org/10.5281/zenodo.23917544`**. The
> repository contains a PyTorch implementation of the method described in the
> paper, together with instructions for training, evaluation, computational
> profiling, and edge-device benchmarking.

## License

Released under the [MIT License](LICENSE). If your target journal or
institution requires a different license (e.g. Apache-2.0, or a
non-commercial research license), replace `LICENSE` accordingly before
publishing the final repository link in the manuscript.

## Authors

- **T. Kavitha** -- conceptualization, architecture design, experiments, analysis, original draft.
- **Dr. A. Kathirvel** -- supervision, methodology validation, review & editing.

## Acknowledgements

The authors thank their respective institution (Saveetha Engineering
College, Chennai, India) for providing the necessary facilities and
research support.
