# FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering

Yunji Seo*, Young Sun Choi*, Hyun Seung Son, [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor)

[![arXiv](https://img.shields.io/badge/arXiv-2408.128894-b31b1b.svg)](https://arxiv.org/abs/2408.12894) 
[![Project Page](https://img.shields.io/badge/Visit-Project_Page-007ec6.svg)](https://3dgs-flod.github.io/flod/)

## Overview
<img src="https://github.com/3DGS-FLoD/flod/blob/main/assets/overall.jpeg" alt="Overview" width="100%" />

We introduce integrating a Flexible Level of Detail (FLoD) to 3DGS, to allow a scene to be rendered at varying levels of detail according to hardware capabilities.  

## Installation
Our code was tested Ubuntu 20.04.6 LTS, on conda environment installed with environment.yml and the submodules below.

```bash
git clone https://github.com/3DGS-FLoD/flod.git
cd flod
```

Setup conda environment
```bash
conda env create -f environment.yml
conda activate flod
```

Clone submodules
```bash
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
```

Install dependencies
```bash
sudo apt install libglm-dev
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Training and Evaluation
To reproduce, run...
(Link to datasets used in the paper: [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Tanks&Temples](https://www.tanksandtemples.org/download/), [DL3DV-10K](https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#dataset-download))
```bash
train.sh 
```

Render and evaluate by...
```bash
render.sh 
```

## Viewer(Demo on Laptop, GeForce MX250)
https://github.com/user-attachments/assets/04de9f26-b25d-4aa9-bed8-5fd3060f0b49

To run viewer as demonstrated on our project page
```bash
convert4viewer.sh
SIBR_viewers/install/bin/SIBR_flodViewer_app -m /path/to/your/model
```

## Licencse
We build our code for FLoD on top of the open-source code of 3D Gaussian Splatting.  
Hence our licencse follows [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)


## Acknowledgement
We would like to express our gratitude to the authors of the 3D Gaussian Splatting.  
Their work has laid the foundation for this research.  
Our code is largely based on their open-source project: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)


## Citation
```bibtex
@article{seo2025flod,
      author  = {Yunji Seo and Young Sun Choi and Hyun Seung Son and Youngjung Uh},
      title   = {FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering},
      journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH)},
      number  = {4},
      volume  = {44},
      year    = {2025},
      doi     = {10.1145/3731430}
}
```
