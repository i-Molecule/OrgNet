CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -y -n orgnet python=3.10
conda activate orgnet

conda install -y pytorch==2.1 torchvision==0.16 -c pytorch -c nvidia

conda install -y -c conda-forge torchmetrics scikit-learn

conda install -y pandas numpy=1.24 tqdm

git clone https://github.com/bkmi/e3nnet.git
python3 -m pip install lie_learn --no-cache-dir
sed -i -e 's/torch.qr/torch.linalg.qr/g' ./e3nnet/se3cnn/kernel.py