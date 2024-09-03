# FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering

Yunji Seo*, Young Sun Choi, Hyun Seung Son, [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor)

[![arXiv](https://img.shields.io/badge/arXiv-2408.128894-b31b1b.svg)](https://arxiv.org/pdf/2408.12894v1) 
[![Project Page](https://img.shields.io/badge/Visit-Project_Page-007ec6.svg)](https://3dgs-flod.github.io/flod.github.io/)

## Overview

## Installation
Our code was tested on conda environment installed with environment.yml and the submodules below.

```bash
# Setup conda environement
(help fill in here for installing environment.yml)

# Clone submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn

# Install dependencies
sudo apt install libglm-dev
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Training and Evaluation
To reproduce, run...
```bash
train_{dataset_name}.sh to train 3DGS-FLoD
```

Render and evaluate by...
```bash
render_single.sh # for individual level rendering of 3DGS-FLoD
render_selective.sh #for selective rendering of 3DGS-FLoD
```

## Viewer(Demo)
To run viewer as demonstrated on our project page
```bash
convert4viewer.sh
SIBR_viewers/install/bin/SIBR_flodViewer_app /path/to/your/model
```

## Licencse
We build our code for FLoD on top of the open-source code of 3D Gaussian Splatting.
Hence our licencse follows [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
