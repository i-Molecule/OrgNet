# OrgNet

OrgNet is a 3D-CNN model designed to predict ddG values upon single amino acid substitutions in PDB structures.

## Features
- **Data Preprocessing**: Scripts for preparing datasets for OrgNet and ThermoNet-like models.
- **Training**: Training scripts for ThermoNet-like models.
- **Inference**: Scripts for predicting ddG values using OrgNet and ThermoNet-like models.

## Repository Structure
- **`data_preprocessing/`**: Contains data preprocessing scripts and the main data processing notebook: `Dataset_processing.ipynb`.
- **`OrgNet/`**: PyTorch implementation of OrgNet for inference.
- **`ThermoNet-like/`**: TensorFlow implementation of ThermoNet-like models for training and inference.

## Installation
Refer to the `README.md` file in each folder for detailed environment setup instructions.

## Usage
- **Data Preprocessing**: Use `data_preprocessing/Dataset_processing.ipynb` notebook to prepare datasets for OrgNet and ThermoNet-like models. See `data_preprocessing/README.md` for details.
- **Training**: Follow the full pipeline of preprocessing, training scripts, and inference guidelines in `ThermoNet-like/README.md`.
- **Inference**:
  - For OrgNet: Use the inference script in `OrgNet/`. See `OrgNet/README.md` for details.
  - For ThermoNet-like models: Use the inference scripts in `ThermoNet-like/`.
