# VisDerm: Split Vision Transformers with Privacy for Teledermatology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)

This repository provides the implementation, trained models, and reproduction scripts for the paper:

> **VisDerm: Split Vision Transformers with Privacy for Teledermatology**  
> Shaik Rukhsar Begum and N. Mallikharjuna Rao  
> *IAES International Journal of Artificial Intelligence (IJ-AI)*, Manuscript 32906

## Overview

VisDerm is a privacy-preserving split-inference framework for resource-constrained teledermatology. Key contributions:

1. **Split-ViT** — A parameter-free split of DeiT-Tiny at transformer block 6, balancing client/server computation while introducing zero additional trainable parameters.
2. **STP-DP** — Stochastic Token Pruning with Differential Privacy, retaining 25% of patch tokens uniformly at random and applying calibrated Gaussian noise. Reduces transmission to 38 KB (3.9× reduction) and enables a smaller noise scale at fixed (ε, δ) under the analytic Gaussian mechanism.
3. **Noise Dimensionality Heuristic** — An empirical design heuristic from systematic comparison of seven privacy mechanisms: on compact Vision Transformers (≤192-d embeddings), reducing the number of noisy dimensions dominates redistributing noise across dimensions.
4. **Defense-in-Depth Empirical Validation** — Membership inference (AUC 0.543 → 0.514) and feature-inversion attacks (SSIM 0.752 → 0.278) on HAM10000, plus cross-dataset assessment on DDI.

## Quick Start: Reproduce Table 1

The single-file script `reproduce.py` downloads HAM10000, applies the canonical 70/15/15 patient-grouped split (`GroupShuffleSplit` on `lesion_id` with `random_state=42`), loads the released checkpoint, and reproduces the Split-ViT row of Table 1.

```bash
git clone https://github.com/ShaikRukhsarBegum/visderm.git
cd visderm
pip install -r requirements.txt

# Download the trained checkpoint (~21 MB) from the GitHub Release
wget https://github.com/ShaikRukhsarBegum/visderm/releases/download/v1.0-ijai/model1.pth

python reproduce.py
```

Expected output:
```
Loading HAM10000 metadata (n=10015) ...
Patient-grouped split: train=6963 val=1525 test=1527
Loading checkpoint: model1.pth (sha256: c32d8680d8a565...)
Evaluating on n=1527 test images ...
  Test accuracy:    73.87%
  Melanoma recall:  79.57%
  Per-inference payload: 148 KB (148 KB without STP-DP, 38 KB with)
```

The checkpoint `model1.pth` is attached to [GitHub Release v1.0-ijai](https://github.com/ShaikRukhsarBegum/visderm/releases/tag/v1.0-ijai) (~21 MB).

## Repository Structure

```
visderm/
├── README.md                 ← this file
├── LICENSE                   ← MIT
├── requirements.txt          ← pinned package versions
├── reproduce.py              ← single-file reproduction of Table 1
├── src/
│   ├── model.py              ← SplitViT class (DeiT-Tiny, split at layer 6)
│   ├── privacy.py            ← STP-DP and seven privacy mechanism implementations
│   ├── federated.py          ← FedAvg / AGM v6 / History AGM aggregators
│   ├── attacks.py            ← Byzantine attacks (negate / flip / ALIE)
│   ├── data.py               ← HAM10000 dataset and patient-grouped split
│   └── train.py              ← Training loop with cosine schedule, mel-class boost
└── scripts/
    ├── train_baseline.py     ← Section 4.1, Table 1 (Split-ViT training)
    ├── eval_privacy.py       ← Section 4.2, Table 2 (seven mechanisms)
    ├── run_federated.py      ← Section 4.6, Table 4 (Byzantine robustness)
    └── ddi_evaluation.py     ← Section 4.5 (cross-dataset DDI)
```

## Datasets

### HAM10000

**Source:** Tschandl et al. (2018), "The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Sci. Data*, 5, 180161.

**Download:** [https://doi.org/10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)

The repository expects the dataset at `data/HAM10000/` with structure:
```
data/HAM10000/
├── HAM10000_metadata.csv     (10015 rows, columns: lesion_id, image_id, dx, ...)
└── images/                    (10015 .jpg files, named <image_id>.jpg)
```

### DDI (for Section 4.5 cross-dataset evaluation)

**Source:** Daneshjou et al. (2022), "Disparities in dermatology AI performance on a diverse, curated clinical image set," *Sci. Adv.*, 8(31), eabq6147.

**Download:** [https://ddi-dataset.github.io/](https://ddi-dataset.github.io/) (registration required)

## Reproducing Each Table / Figure

| Paper Element | Script | Approximate Runtime |
|---|---|---|
| Table 1 (Method comparison) | `python reproduce.py` | 2 min (CPU) / 30 sec (GPU) |
| Table 2 (Seven privacy mechanisms) | `python scripts/eval_privacy.py` | 3 hours (single GPU) |
| Table 3 (5-seed reproducibility) | `python scripts/train_baseline.py --seeds 42 123 777 2024 31415` | 3.5 hours (5× 40 min) |
| Table 4 (Federated, Byzantine attacks) | `python scripts/run_federated.py` | 8 hours (single GPU) |
| Section 4.5 DDI evaluation | `python scripts/ddi_evaluation.py` | 5 min |
| Figure 3 (feature inversion) | `python scripts/inversion_attack.py` | 1 hour |

Runtimes assume an NVIDIA A100. Smaller GPUs work but take proportionally longer.

## Citation

If you use this code or the released checkpoint, please cite:

```bibtex
@article{begum2026visderm,
  author    = {Shaik Rukhsar Begum and N. Mallikharjuna Rao},
  title     = {{VisDerm}: {Split} {Vision} {Transformers} with Privacy for Teledermatology},
  journal   = {IAES International Journal of Artificial Intelligence (IJ-AI)},
  year      = {2026},
  note      = {Forthcoming}
}
```

## License

This code is released under the **MIT License** (see `LICENSE`). The released checkpoint (`model1.pth`) is also released under MIT.

The HAM10000 and DDI datasets are governed by their own licenses; please refer to their original publications.

## Acknowledgments

We thank the IJ-AI reviewers whose detailed feedback materially improved the work. Computational experiments were conducted using Google Colaboratory GPU resources. AI-assisted tools (Claude, Gemini) were used for code review, writing assistance, and reproducibility verification; the authors take full responsibility for scientific content, methodology, and results.

## Contact

For questions about reproduction, please open an issue on this repository. For other inquiries:

**Shaik Rukhsar Begum** — Research Scholar, Department of Computer Science, Annamacharya University, Rajampeta, India  
Email: `shaik.rukhsar.begum@gmail.com`
