# OrgNet

OrgNet is a 3D-CNN model designed to predict ddG values upon single amino acid substitutions in PDB structures. This repository provides inference and training scripts for OrgNet and ThermoNet-like models.


## Setting Up a Virtual Environment

**Prerequisites:**
- Anaconda/Miniconda
- Optional: GPU with CUDA support

Run the `install.sh` script to create the `orgnet` environment within `$CONDA_BASE/envs/`:

```bash
bash -l install.sh
```


## Downloading Datasets

You can download the datasets either manually or via the terminal:

### Option 1: Manual Download
Visit [Zenodo](https://zenodo.org/records/15098791) and download `OrgNet_data.tar.gz`. Place the file in the `data` directory.

### Option 2: Terminal Download
Run the following commands:
```bash
cd ./data

wget https://zenodo.org/records/15098791/files/OrgNet_data.tar.gz?download=1 -O OrgNet_data.tar.gz
```

After downloading, extract the archive:
```bash
cd ./data

tar -xvzf OrgNet_data.tar.gz
```


## Running OrgNet Inference

Activate the `orgnet` environment and run the inference script:
```bash
conda activate orgnet

python predict.py -X /path/to/X.npy -y /path/to/y.npy --save_to /path/to/predictions.csv --device cpu --trained_on S2648_V
```

### Required Arguments:
- `-X`: Path to the `.npy` file containing voxels.
- `-y`: Path to the `.npy` file containing values.

### Optional Arguments:
- `--save_to`: Path to save predictions as a `.csv` file. If not provided, predictions are not saved.
- `--device`: Device to run inference (`cpu` or `cuda`). Default is `cpu`.
- `--trained_on`: Model trained on `Q3214` or `S2648_V`. Default is `S2648_V`.

### Example Commands:
#### Table 2:
```bash
python predict.py -X data/Datasets/OrgNet/test/S669_X_direct.npy -y data/Datasets/OrgNet/test/S669_y_direct.npy --device cpu --trained_on S2648_V

python predict.py -X data/Datasets/OrgNet/test/S669_X_reverse.npy -y data/Datasets/OrgNet/test/S669_y_reverse.npy --device cpu --trained_on S2648_V
```

#### Table 3:
```bash
python predict.py -X data/Datasets/OrgNet/test/Ssym_X_direct.npy -y data/Datasets/OrgNet/test/Ssym_y_direct.npy --device cpu --trained_on S2648_V

python predict.py -X data/Datasets/OrgNet/test/Ssym_X_reverse.npy -y data/Datasets/OrgNet/test/Ssym_y_reverse.npy --device cpu --trained_on S2648_V

python predict.py -X data/Datasets/OrgNet/test/Ssym_X_direct.npy -y data/Datasets/OrgNet/test/Ssym_y_direct.npy --device cpu --trained_on Q3214

python predict.py -X data/Datasets/OrgNet/test/Ssym_X_reverse.npy -y data/Datasets/OrgNet/test/Ssym_y_reverse.npy --device cpu --trained_on Q3214
```

## Running ThermoNet-like Models Inference

Activate the `orgnet` environment and run the inference script:
```bash
conda activate orgnet

python predict.py -X /path/to/X.npy -y /path/to/y.npy --model_name ThermoNet --model_weights_dir models/weights/thermonet_augmented/Q1744 --save_to /path/to/predictions.csv --device cuda
```

### Required Arguments:
- `-X`: Path to the `.npy` file containing voxels.
- `-y`: Path to the `.npy` file containing values.
- `--model_name`: Choose `ThermoNet` for ThermoNet trained on augmented data or `ThermoNet_steerable`.
- `--model_weights_dir`: Directory containing model weights. With `{train_data}` set to either `Q1744` or `Q3214`, use:
  - `models/weights/thermonet_augmented/{train_data}` for ThermoNet.
  - `models/weights/thermonet_steerable/{train_data}` for ThermoNet steerable.

### Optional Arguments:
- `--save_to`: Path to save predictions as a `.csv` file. If not provided, predictions are not saved.
- `--device`: Device to run inference (`cpu` or `cuda`). Default is `cpu`. Set to `cuda` for ThermoNet steerable.

### Example Commands:
#### Table 1:
```bash
python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet --model_weights_dir models/weights/thermonet_augmented/Q1744

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_reverse.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_reverse.npy --model_name ThermoNet --model_weights_dir models/weights/thermonet_augmented/Q1744

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet_steerable --model_weights_dir models/weights/thermonet_steerable/Q1744 --device cuda

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_reverse.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_reverse.npy --model_name ThermoNet_steerable --model_weights_dir models/weights/thermonet_steerable/Q1744 --device cuda

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet --model_weights_dir models/weights/thermonet_augmented/Q3214

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_reverse.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_reverse.npy --model_name ThermoNet --model_weights_dir models/weights/thermonet_augmented/Q3214

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet_steerable --model_weights_dir models/weights/thermonet_steerable/Q3214 --device cuda

python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_reverse.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_reverse.npy --model_name ThermoNet_steerable --model_weights_dir models/weights/thermonet_steerable/Q3214 --device cuda
```


## Training OrgNet

Activate the `orgnet` environment and run the training script:
```bash
conda activate orgnet

python train.py -data_json /path/to/data_dictionary.json
```

### Required Arguments:
- `-data_json`: Path to the `.json` file specifying paths to data, folds, and other settings.

### Optional Arguments:
- `--device`: Device to run training (`cpu` or `cuda`). Default is `cuda`.

### Example Commands:
#### Training on S2648:
```bash
python train.py -data_json train_res/orgnet/S2648_V/train_S2648_V.json
```
Run inference:
```bash
python predict.py -X data/Datasets/OrgNet/test/Ssym_X_direct.npy -y data/Datasets/OrgNet/test/Ssym_y_direct.npy --model_name OrgNet --model_weights_dir train_res/orgnet/S2648_V --trained_on S2648_V
```

#### Training on Q3214:
```bash
python train.py -data_json train_res/orgnet/Q3214/train_Q3214.json
```
Run inference:
```bash
python predict.py -X data/Datasets/OrgNet/test/Ssym_X_direct.npy -y data/Datasets/OrgNet/test/Ssym_y_direct.npy --model_name OrgNet --model_weights_dir train_res/orgnet/Q3214 --trained_on Q3214
```


## Training ThermoNet-like Models

Activate the `orgnet` environment and run the training script:
```bash
conda activate orgnet

python train.py -data_json /path/to/data_dictionary.json
```

### Required Arguments:
- `-data_json`: Path to the `.json` file specifying paths to data, folds, and other settings.

### Optional Arguments:
- `--device`: Device to run training (`cpu` or `cuda`). Default is `cuda`.

### Example Commands:
#### ThermoNet (Augmented Data):
##### Q1744
```bash
python train.py -data_json train_res/thermonet_augmented/Q1744/train_Q1744.json
```
Run inference:
```bash
python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet --model_weights_dir train_res/thermonet_augmented/Q1744
```

##### Q3214
```bash
python train.py -data_json train_res/thermonet_augmented/Q3214/train_Q3214.json
```
Run inference:
```bash
python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet --model_weights_dir train_res/thermonet_augmented/Q3214
```

#### ThermoNet Steerable:
##### Q1744
```bash
python train.py -data_json train_res/thermonet_steerable/Q1744/train_Q1744.json
```
Run inference:
```bash
python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet_steerable --model_weights_dir train_res/thermonet_steerable/Q1744 --device cuda
```

##### Q3214
```bash
python train.py -data_json train_res/thermonet_steerable/Q3214/train_Q3214.json
```
Run inference:
```bash
python predict.py -X data/Datasets/ThermoNet-like/test/Ssym_X_direct.npy -y data/Datasets/ThermoNet-like/test/Ssym_y_direct.npy --model_name ThermoNet_steerable --model_weights_dir train_res/thermonet_steerable/Q3214 --device cuda
```