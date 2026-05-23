# ProOOD (VoxDet Backbone) 🚧

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.01081)
[![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/7uHeng/ProOOD)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **🚧 This branch is under development.** The ProOOD + VoxDet implementation is not yet complete.  
> For the ready-to-use version, switch to the [`main`](https://github.com/7uHeng/ProOOD/tree/main) branch.

## What is ProOOD?

**ProOOD** is a lightweight plug-and-play method for 3D semantic occupancy prediction that tackles long-tailed class bias and out-of-distribution (OOD) detection through three components:

1. **Prototype-Guided Semantic Imputation** — fills occluded regions
2. **Prototype-Guided Tail Mining** — strengthens rare-class representations
3. **EchoOOD Score** — training-free OOD detection fusing local logit coherence with prototype matching

State-of-the-art on five benchmarks across both in-distribution occupancy prediction and OOD detection.

---

| [🛠️ Installation](#installation) | [📂 Data](#data-preparation) | [🏋️ Training](#training--evaluation) | [📜 Citation](#citation) |
|:---:|:---:|:---:|:---:|

---

## 🛠️ Installation

See **[docs/install.md](docs/install.md)** — Conda, PyTorch 1.9.1 + CUDA 11.1, mmdet3d, spconv, etc.

## 📂 Data Preparation

See **[docs/dataset.md](docs/dataset.md)** for SemanticKITTI, KITTI-360, Vaakitti, Vaakitti360, and STU setup.

## 🏋️ Training & Evaluation

See **[docs/run.md](docs/run.md)** for the full command list. ⚠️ VoxDet-specific configs and weights coming soon.

```bash
# Switch to the stable SGN branch for now:
git checkout main
./tools/dist_train.sh projects/configs/sgn/proood-semkitti.py 4
```

## 📜 Citation

```bibtex
@article{zhang2026proood,
  title   = {ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction},
  author  = {Zhang, Yuheng and Duan, Mengfei and Peng, Kunyu and Wang, Yuhang and
             Wen, Di and Paudel, Danda Pani and Van Gool, Luc and Yang, Kailun},
  journal = {arXiv preprint arXiv:2604.01081},
  year    = {2026}
}
