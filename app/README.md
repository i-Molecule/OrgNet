
# Installation

## Install Rosetta
Follow instructions for Rosetta installation in `data_preprocessing/README.md`.

```bash
cd data_preprocessing/
# download
wget https://downloads.rosettacommons.org/downloads/academic/3.13/rosetta_bin_linux_3.13_bundle.tgz
# extract files
tar -xvzf rosetta_bin_linux_3.13_bundle.tgz
# rename folder
mv rosetta_bin_linux_2021.16.61629_bundle rosetta
```

## Setup conda environment

This will create `orgnet_st` conda environment and install required packages.
```bash
bash app/install.sh
```

# Inference

Running streamlit app
```bash
conda install orgnet_st
python3 -m streamlit run app/main.py
```