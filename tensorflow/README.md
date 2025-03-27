# OrgNet
OrgNet is a 3D-CNN model capable of predicting ddG upon single amino acid substitution in pdb structue.

## Installation

You can create an environment from .yml file:

```bash
conda env create -f OrgNet_conf.yml
conda activate OrgNet_conf
```

All required packages and versions can be found in requirements.txt file.

GPCRapa was tested on Ubuntu 22.04.5 LTS, AMD® Ryzen 9 7950x 16-core processor × 32, NVIDIA Corporation NVIDIA GeForce RTX 4090.

## Usage

To look at usage example of OrgNet see Tutorial.ipynb in /Notebooks/

## Training

To train your own model on your own data use script Train.py in /Notebooks/ . It directly uses .npy voxel grids prepared from the training datasets, thus it may require adjustment for your type of data preparation.
```
python Train.py -evdirect /path_to_direct_eval_features/ -evreverse  /path_to_reverse_eval_features/ -evds  /path_to_csv_eval_ds/ -trdirect /path_to_train_direct_ds/ -trreverse /path_to_train_reverse_ds/ -trds /path_to_csv_train_ds/ -mod /path_to_save_models/ -log /path_to_save_logs/
```
All options:

```
usage: Train.py [-h] [- evdirect PATH_TO_EVAL_DATASET_DIRECT_FEATURES] [-evreverse PATH_TO_EVAL_DATASET_REVERSE_FEATURES] [-evds PATH_TO_EVAL_DATASET_CSV] [-trdirect PATH_TO_TRAIN_DATASET_DIRECT_FEATURES] [-trreverse PATH_TO_TRAIN_DATASET_REVERSE_FEATURES] [-trds PATH_TO_TRAIN_DATASET_CSV]
                             [-mod PATH_TO_SAVE_MODELS] [-log PATH_TO_SAVE_LOGS]

Training script for OrgNet.

optional arguments:
  -h, --help            show this help message and exit
  -evdirect INPUT_DIRECT, --input_direct INPUT_DIRECT
                        Path to calculated direct features.
  -evreverse INPUT_REVERSE, --input_reverse INPUT_REVERSE
                        Path to calculated reverse features.
  -evds INPUT_DATASET, --input_dataset INPUT_DATASET
                        Path to dataset csv file.
  -trdirect TRAIN_DIRECT, --train_direct TRAIN_DIRECT
                        Path to calculated train direct features.
  -trreverse TRAIN_REVERSE, --train_reverse TRAIN_REVERSE
                        Path to calculated train reverse features.
  -trds TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        Path to train dataset csv file.
  -mod MODEL_PATH, --model_path MODEL_PATH
                        Path to save models.
  -log PATH_TO_SAVE_LOGGING, --path_to_save_logging PATH_TO_SAVE_LOGGING
                        Path to save logs.
```

To prepare your dataset for OrgNet use   
use custom splitting function to split your data, use gends_2prots_1_TOTAL1.py:
```
gends_2prots_1_TOTAL1.py  -iwt /path_to_wt/ -imut /path_to_mut/ -o /path_to_save/ --boxsize 16 --voxelsize 1
```
All options:
```
usage: gends_2prots_1_TOTAL1.py [-h] [-iwt INPUT_WILDTYPE_PROTEIN] [-imut INPUT_MUTANT_PROTEIN] [-o OUTPUT]
                                [--boxsize BOXSIZE] [--voxelsize VOXELSIZE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -iwt INPUT_WILDTYPE_PROTEIN, --input_wildtype_protein INPUT_WILDTYPE_PROTEIN
                        Pdb file for WT.
  -imut INPUT_MUTANT_PROTEIN, --input_mutant_protein INPUT_MUTANT_PROTEIN
                        Pdb file for Mut.
  -o OUTPUT, --path_to_save OUTPUT
                        Path to save output features.
  --boxsize BOXSIZE     Size of the bounding box around the mutation site.
  --voxelsize VOXELSIZE
                        Size of the voxel.
  -v, --verbose         Whether to print verbose messages from HTMD function calls.
```



## Inference

To predict the ddG for a set of processed proteins, use the  Inference script.

Create the conda environment (remember to change the prefix directory in OrgNet_conf.yml). 
```
conda env create -f OrgNet_conf.yml
```

After this, activate the env and run the Inference.py script.
```
usage: Inference.py [-h] -evdirect INPUT_DIRECT -evreverse INPUT_REVERSE -evds INPUT_DATASET -o OUTPUT -mod MODEL_DIR
                    [-flag W]

optional arguments:
  -h, --help            show this help message and exit
  -evdirect INPUT_DIRECT, --input_direct INPUT_DIRECT
                        Path to calculated direct features.
  -evreverse INPUT_REVERSE, --input_reverse INPUT_REVERSE
                        Path to calculated reverse features.
  -evds INPUT_DATASET, --input_dataset INPUT_DATASET
                        Path to dataset csv file.
  -o OUTPUT, --output OUTPUT
                        Path to save the evaluation dataframe.
  -mod MODEL_DIR, --model_dir MODEL_DIR
                        Path to models.
  -flag W, --w W        Sting to flag the models.

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
