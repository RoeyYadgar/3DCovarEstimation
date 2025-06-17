
## Installation Guide

COV3D uses RELION for conesus reconstruction and RECOVAR for heterogneous reconstruction from the latent embedding produced by COV3D.
Follow the instructions bellow for installing REOCVAR and COV3D
```
conda create --name cov3d python=3.11
conda activate cov3d

#Install RECOVAR
git clone git@github.com:ma-gilles/recovar.git
cd recovar
pip install recovar
cd ..


conda install -c nvidia -c conda-forge cuda=12.1 cudnn
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```
Additionally, make sure RELION is installed and `relion_reconstruct`
