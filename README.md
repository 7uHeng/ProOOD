# ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.01081)
[![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/7uHeng/ProOOD)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="https://github.com/7uHeng/ProOOD/blob/main/asserts/pipeline.png" width="800" alt="ProOOD Pipeline">
</p>

## 📰 Abstract
3D semantic occupancy prediction is central to autonomous driving, yet current methods are vulnerable to **long-tailed class bias** and **out-of-distribution (OOD)** inputs, often over-confidently assigning anomalies to rare classes. 

We present **ProOOD**, a lightweight, plug-and-play method that couples prototype-guided refinement with training-free OOD scoring. ProOOD comprises three key components:
1. **Prototype-Guided Semantic Imputation**: Fills occluded regions with class-consistent features to enhance structural completeness.
2. **Prototype-Guided Tail Mining**: Strengthens rare-class representations to curb OOD absorption, improving long-tail performance.
3. **EchoOOD Score**: A novel OOD detection mechanism that fuses local logit coherence with local and global prototype matching to produce reliable voxel-level OOD scores.

Extensive experiments on five datasets demonstrate that ProOOD achieves state-of-the-art performance on both **in-distribution 3D occupancy prediction** and **OOD detection**.

---

## 🚀 Updates

* **[2026/04]** 🎉 Repository initialized! We are actively preparing the final release.
    * **Current Status**:
        - [ ] Update & Clean OCC Training Code
        - [ ] Release Pre-trained OCC Weights (ID Performance)
        - [ ] Release OccOoD Dataset (Benchmark for OOD Detection)
        - [ ] Release OOD Detection Weights & Scripts
    
    * **Note**: Please check the **Branches** section below for baseline implementations.

---

## 🌿 Branches & Baselines

ProOOD is designed as a **plug-and-play** module. We provide implementations integrated with two distinct mainstream 3D occupancy prediction backbones:

| Branch Name | Description | Status |
| :--- | :--- | :--- |
| [`main`](https://github.com/7uHeng/ProOOD/tree/main) | **ProOOD + [SGN]** <br> The primary implementation integrating ProOOD with **[SGN]**.  | 🚧 Updating |
| [`baseline_b`](https://github.com/7uHeng/ProOOD/tree/baseline_b) | **ProOOD + [VoxDet]** <br> The implementation integrating ProOOD with **[VoxDet]**.  | 🚧 Updating |

> **💡 Note on Plug-and-Play Design:**
> ProOOD does not rely on specific backbone structures. The core modules (Semantic Imputation, Tail Mining, and EchoOOD scoring) can be easily adapted to other 3D occupancy frameworks. We provide these two branches as representative examples.

> **How to switch between baselines:**
> ```bash
> # Checkout the implementation based on [Baseline A]
> git checkout main
> 
> # Checkout the implementation based on [Baseline B]
> git checkout baseline_b
> ```

---

## 🛠️ Installation
*(Coming Soon)*
We will provide detailed instructions for environment setup, including CUDA versions and dependency installation.

## 📂 Data Preparation
*(Coming Soon)*
Instructions for downloading SemanticKITTI/SSCBench-KITTI360 and processing the **OccOoD** dataset will be released shortly.

## 🏋️ Training & Evaluation
*(Coming Soon)*
Scripts for training ProOOD and evaluating both ID and OOD metrics will be provided.

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
