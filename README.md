# OrgNet

OrgNet is a 3D-CNN model designed to predict ddG values upon single amino acid substitutions in PDB structures.

## Repository Structure

The repository is organized into two distinct environments for TensorFlow and PyTorch implementations.

- **`tensorflow/`**: Contains the TensorFlow implementation of ThermoNet-like and OrgNet-like models, including data preprocessing, training, and inference scripts. Refer to the `README.md` inside this folder and `Readme.md` inside its subfolders for detailed instructions.
- **`pytorch/`**: Contains the PyTorch implementation of OrgNet, focusing on inference. Refer to the `README.md` inside this folder for more details.

## Installation

Each folder (`tensorflow/` and `pytorch/`) includes its own environment configuration file (refer to `.yml` and `.sh`) and setup instructions. Follow the specific instructions in the respective folder's `README.md` to set up the required environment.

## Usage

- **Data Preprocessing**: Instructions for preparing datasets are provided in the `tensorflow/` folder.
- **Training**: Training scripts and guidelines are available in the `tensorflow/` folder.
- **Inference**: Inference scripts for predicting ddG values are available in both the `tensorflow/` and `pytorch/` folders.
