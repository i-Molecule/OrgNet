CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -y -n tf-thermonet python=3.9
conda activate tf-thermonet

conda install -y pandas numpy tqdm scikit-learn scipy matplotlib

python3 -m pip install 'tensorflow[and-cuda]'