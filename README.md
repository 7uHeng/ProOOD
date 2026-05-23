# ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.01081)
[![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/7uHeng/ProOOD)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="asserts/pipeline.png" width="800" alt="ProOOD Pipeline">
</p>

## 📰 Abstract
3D semantic occupancy prediction is central to autonomous driving, yet current methods are vulnerable to **long-tailed class bias** and **out-of-distribution (OOD)** inputs, often over-confidently assigning anomalies to rare classes. 

We present **ProOOD**, a lightweight, plug-and-play method that couples prototype-guided refinement with training-free OOD scoring. ProOOD comprises three key components:
1. **Prototype-Guided Semantic Imputation**: Fills occluded regions with class-consistent features to enhance structural completeness.
2. **Prototype-Guided Tail Mining**: Strengthens rare-class representations to curb OOD absorption, improving long-tail performance.
3. **EchoOOD Score**: A novel OOD detection mechanism that fuses local logit coherence with local and global prototype matching to produce reliable voxel-level OOD scores.

Extensive experiments on five datasets demonstrate that ProOOD achieves state-of-the-art performance on both **in-distribution 3D occupancy prediction** and **OOD detection**.

---

## 🛠️ Installation

Please refer to **[docs/install.md](docs/install.md)** for step-by-step installation instructions, including:
- Conda environment setup (Python 3.8)
- PyTorch 1.9.1 + CUDA 11.1
- mmcv-full, mmdet, mmseg, mmdet3d
- Additional dependencies (timm, spconv, torch-scatter, torchmetrics)

## 📂 Data Preparation

Please refer to **[docs/dataset.md](docs/dataset.md)** for detailed instructions on:
- **SemanticKITTI**: Download, label generation, depth estimation (MobileStereoNet/SQL), pseudo point cloud generation
- **SSCBench-KITTI-360**: Download, label preprocessing, depth to pseudo point cloud
- **OccOoD Dataset**: OOD benchmarks (Vaakitti, Vaakitti360, STU)

## 🏋️ Training & Evaluation

Please refer to **[docs/run.md](docs/run.md)** for all training and evaluation commands.

### Quick Start

```shell
# Train ProOOD on SemanticKITTI with 4 GPUs
./tools/dist_train.sh projects/configs/sgn/proood-semkitti.py 4

# Evaluate
./tools/dist_test.sh projects/configs/sgn/proood-semkitti.py ./path/to/ckpts.pth 4

# OOD Evaluation
./tools/dist_test_ood.sh projects/configs/sgn/proood-ood-vaakitti.py ./path/to/ckpts.pth 4
```

### Available Configs

| Config | Dataset | Depth Model | OOD? |
|--------|---------|-------------|------|
| `proood-semkitti.py` | SemanticKITTI | MobileStereoNet | No |
| `proood-sql-semkitti.py` | SemanticKITTI | SQL | No |
| `proood-kitti360.py` | KITTI-360 | MobileStereoNet | No |
| `proood-sql-kitti360.py` | KITTI-360 | SQL | No |
| `proood-ood-vaakitti.py` | Vaakitti | SQL | Yes |
| `proood-ood-vaakitti360.py` | Vaakitti360 | SQL | Yes |
| `proood-ood-stu.py` | STU | SQL | Yes |

---

## 🌿 Branches & Baselines

ProOOD is designed as a **plug-and-play** module. We provide implementations integrated with two distinct mainstream 3D occupancy prediction backbones:

| Branch Name | Description | Status |
| :--- | :--- | :--- |
| [`main`](https://github.com/7uHeng/ProOOD/tree/main) | **ProOOD + [SGN]** <br> The primary implementation integrating ProOOD with **[SGN]**. Contains training/evaluation code for SemanticKITTI, KITTI-360, and OOD benchmarks. | ✅ Available |
| [`baseline_b`](https://github.com/7uHeng/ProOOD/tree/baseline_b) | **ProOOD + [VoxDet]** <br> The implementation integrating ProOOD with **[VoxDet]**. | 🚧 Updating |

> **💡 Note on Plug-and-Play Design:**
> ProOOD does not rely on specific backbone structures. The core modules (Semantic Imputation, Tail Mining, and EchoOOD scoring) can be easily adapted to other 3D occupancy frameworks. We provide these two branches as representative examples.

> **How to switch between baselines:**
> ```bash
> # Checkout the implementation based on [SGN]
> git checkout main
> 
> # Checkout the implementation based on [VoxDet]
> git checkout baseline_b
> ```

---

## 📜 Citation
If you find this work helpful in your research, please consider citing:

```bibtex
@article{zhang2026proood,
  title={ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction},
  author={Zhang, Yuheng and Duan, Mengfei and Peng, Kunyu and Wang, Yuhang and Wen, Di and Paudel, Danda Pani and Van Gool, Luc and Yang, Kailun},
  journal={arXiv preprint arXiv:2604.01081},
  year={2026}
}
```

## Acknowledgement
This project is based on the following open-source projects. We thank their authors for making the source code publically available.
* [SGN](https://github.com/Jieqianyu/SGN)
* [ProtoOcc](https://github.com/SPA-junghokim/ProtoOcc)
* [VoxDet](https://github.com/vita-epfl/VoxDet)
* [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
