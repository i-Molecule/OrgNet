# OrgNet inference

## Setting Up a Virtual Environment with `conda`

**Prerequisites:**
- Anaconda/Miniconda
- Optional: GPU with CUDA support

Run the `install.sh` script to create the `orgnet` environment within `$CONDA_BASE/envs/`:

```bash
cd .../OrgNet
bash -l install.sh
```

## Running OrgNet inference

```bash
conda activate orgnet
python main.py -X /path/to/X.npy -y /path/to/y.npy --save_to /path/to/predictions.csv --device cpu --trained_on S2648_V
```

**Required arguments:**
* -X: Path to the .npy file with voxels.
* -y: Path to the .npy file with values.

**Optional arguments:**
* --save_to: .csv path where predictions will be saved. If not provided, predictions are not saved.
* --device: Device to run the inference on (`cpu` or `cuda`). Default is `cpu`.
* --trained_on: Select model which was trained on `Q3214` or `S2648_V`. Default is `S2648_V`.

**Example:**
```bash
python main.py -X data/datasets/p53_X_reverse.npy -y data/datasets/p53_y_reverse.npy --save_to predictions.csv --device cpu --trained_on S2648_V
```
