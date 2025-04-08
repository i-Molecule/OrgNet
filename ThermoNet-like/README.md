# ThermoNet-like Models: Training and Inference

This repository provides Jupyter Notebooks for training and inference with ThermoNet-like models. The workflow is organized into two Jupyter Notebooks:

- **`Thermonet_like_models_training.ipynb`**: Defines the model architecture, training routines, and evaluation metrics.
- **`Thermonet_like_models_inference.ipynb`**: Scripts for running inference using trained models.

## Setup Instructions

To prepare your datasets for Thermonet-like models, follow instructions in /data_preprocessing/README.md.

### Create the Environment

**Prerequisites:**
- Anaconda/Miniconda
- GPU with CUDA support 
(tested on Ubuntu 22.04.5 LTS, AMD Ryzen 9 7950X, NVIDIA GeForce RTX 4090)

Run the `install.sh` script to create the `tf-thermonet` environment within `$CONDA_BASE/envs/`:
```bash
cd ./ThermoNet-like
bash -l install.sh