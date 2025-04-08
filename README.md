# OrgNet

OrgNet is a 3D-CNN model designed to predict ddG values upon single amino acid substitutions in PDB structures.

## Features
- **Data Preprocessing**: Scripts for preparing datasets for OrgNet and ThermoNet-like models.
- **Training**: Training scripts for OrgNet and ThermoNet-like models.
- **Inference**: Scripts for predicting ddG values using OrgNet and ThermoNet-like models.

## Repository Structure
- **`data_preprocessing/`**: Contains data preprocessing scripts and the main data processing notebook: `Dataset_processing.ipynb`.
- **`OrgNet/`**: PyTorch implementation of OrgNet for training and inference. PyTorch implementation of ThermoNet-like models (including steerable) for training and inference.
- **`ThermoNet-like/`**: TensorFlow implementation of ThermoNet-like models for training and inference.

## Installation
Refer to the `README.md` file in each folder for detailed environment setup instructions.

## Usage
- **Data Preprocessing**: Use `data_preprocessing/Dataset_processing.ipynb` notebook to prepare datasets for OrgNet and ThermoNet-like models. See `data_preprocessing/README.md` for details.
- **Inference**:
  - For OrgNet: Use the inference scripts as described in `Running OrgNet Inference` section within `OrgNet/README.md`.
  - For ThermoNet-like models: 
    - For steerable and augmented versions, see section `Running ThermoNet-like Models Inference` in `OrgNet/README.md`.
    - For other cases, use the inference workflow in `ThermoNet-like/Thermonet_like_models_inference.ipynb`.
- **Training**: 
  - For OrgNet: Use the training scripts as described in `Training OrgNet` section within `OrgNet/README.md`.
  - For ThermoNet-like models: 
    - For steerable and augmented versions, see section `Training ThermoNet-like Models` in `OrgNet/README.md`.
    - For other cases, use the inference workflow in `ThermoNet-like/Thermonet_like_models_training.ipynb`.
