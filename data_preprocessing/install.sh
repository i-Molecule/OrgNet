CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -y -n preprocessing
conda activate preprocessing

conda install -y python=3.10 htmd -c acellera -c conda-forge