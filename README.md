# OrgNet

OrgNet is a 3D-CNN model designed to predict ddG values upon single amino acid substitutions in PDB structures.

## Features
- **Data Preprocessing**: Scripts for preparing datasets for OrgNet and ThermoNet-like models.
- **Training**: Training scripts for ThermoNet-like models.
- **Inference**: Scripts for predicting ddG values using OrgNet and ThermoNet-like models.

## Repository Structure
- **`data_preprocessing/`**: TensorFlow implementation for data preprocessing.
- **`OrgNet/`**: PyTorch implementation of OrgNet for inference.
- **`ThermoNet-like/`**: TensorFlow implementation of ThermoNet-like models for training and inference.

## Installation
Refer to `README.md` in each folder for environment setup instructions. Note that `data_preprocessing/` and `ThermoNet-like/` share the same environment.

## Usage
- **Data Preprocessing**: Instructions for preparing datasets, see `data_preprocessing/README.md`.
- **Training**: Preprocessing, training scripts, inference and guidelines, see `ThermoNet-like/README.md`.
- **Inference**: Inference scripts for predicting ddG values are available in both  `OrgNet/` and `ThermoNet-like/`.
