Paper: FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering
Authors: Yunji Seo*, Young Sun Choi*, Hyun Seung Son, Youngjung Uh

Our code was tested Ubuntu 20.04.6 LTS, on conda environment installed with environment.yml and the submodules below.

1. Setting up running environment for reproduction. 
Download datasets from:
Mip-NeRF 360: https://jonbarron.info/mipnerf360
Tanks&Temples: https://www.tanksandtemples.org/download
DL3DV-10K: https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#dataset-download

In terminal:
git clone https://github.com/3DGS-FLoD/flod.git
cd flod

conda env create -f environment.yml
conda activate flod

git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn

sudo apt install libglm-dev
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn


2. Training and Evaluation.
In terminal:
train.sh 
render.sh 

* adjust path to dataset in the .sh files
