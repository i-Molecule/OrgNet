CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -y -n orgnet python=3.10
conda activate orgnet

conda install -y pytorch==2.1 torchvision==0.16 -c pytorch -c nvidia

conda install -y -c conda-forge torchmetrics

conda install -y pandas numpy=1.24 tqdm