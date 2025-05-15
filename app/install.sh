set -e
set -x

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

export ENVNAME=orgnet_st
export PYTHON_VERSION=3.10
conda create -y -n ${ENVNAME} python=${PYTHON_VERSION}
conda activate ${ENVNAME}

conda install -y -c acellera htmd
conda install -y pytorch=2.1 torchvision=0.16 -c pytorch -c nvidia
conda install -y -c conda-forge torchmetrics scikit-learn

conda install -y pandas numpy=1.24 tqdm

conda install -y streamlit
conda install -y py3dmol

git clone https://github.com/bkmi/e3nnet.git
python3 -m pip install lie_learn --no-cache-dir
sed -i -e 's/torch.qr/torch.linalg.qr/g' ./e3nnet/se3cnn/kernel.py
