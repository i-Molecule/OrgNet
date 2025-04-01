# Dataset Processing for OrgNet and ThermoNet-like Models

This repository contains tools and scripts for preparing datasets, generating mutant structures, and calculating features for OrgNet and ThermoNet-like models. The pipeline includes dataset preprocessing, orientation standardization (for OrgNet), and voxel-based feature calculation.

---

## Overview

- **`Dataset_processing.ipynb`**: Prepares datasets by generating mutants, performing orientation standardization (for OrgNet), and calculating features for OrgNet and ThermoNet-like models.
- **Input**: `.csv` files located in the `/datasets/` folder.
- **Output**: Processed datasets ready for model training.

---

## Setup Instructions

### Install Rosetta 3.13
1. Obtain an academic license for Rosetta from [here](https://els2.comotion.uw.edu/product/rosetta).
2. Download Rosetta 3.13 (source + binaries for Linux) from [this link](https://www.rosettacommons.org/software/license-and-download) or run:
```bash
cd ./data_preprocessing
wget https://downloads.rosettacommons.org/downloads/academic/3.13/rosetta_bin_linux_3.13_bundle.tgz
```
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their FULL path.
```bash
tar -xvzf rosetta_bin_linux_3.13_bundle.tgz
```
4. Rename the extracted folder `rosetta_bin_linux_2021.16.61629_bundle` to `rosetta`:
```bash
mv rosetta_bin_linux_2021.16.61629_bundle rosetta
```
5. Ensure `parallel` is installed:
```bash
parallel --version
```
If not installed, run:
```bash
sudo apt install parallel
```

### Setting Up a Virtual Environment with `conda`

**Prerequisites:**
- Anaconda/Miniconda

1. Run the `install.sh` script to create the `preprocessing` environment within `$CONDA_BASE/envs/`:
```bash
cd ./data_preprocessing
bash -l install.sh
```
2. Activate the environment and register HTMD (free for non-commercial use):
```bash
conda activate preprocessing
htmd_register
```

## OrgNet Dataset Preparation

OrgNet dataset `.csv` files for Ssym, Q1744, Q3214, S669 and S2648 are located in `./data_preprocessing/datasets/` folder.
Use these files as input `.csv` data for `Dataset_processing.ipynb` to create voxel datasets.

Note, that because OrgNet and Thermonet-like models use the following notation for destabilizing (ddg>0) and stabilizing (ddg<0) mutations, the signs for S2648 and S669 were changed accordingly. The correct datasets to reproduce OrgNet models are S669_fixed_sign.csv and S2648_fixed_sign.csv. 

## Important Note

Always specify **absolute paths** for all paths and directories.

## Mutant structures generation

To generate mutant structures use run_rosetta_relax function in Dataset_processing.ipynb. 

Parameters:
    - pdb_id (str): PDB identifier.
    - wt (str): Wild-type amino acid.
    - mut (str): Mutant amino acid.
    - pos (str or int): Position of the mutation.
    - path_to_relaxed_chains (str): Directory with relaxed chain files.
    - PDBDIR (str): Not used directly here, but expected for input structure directory.
    - OUTDIR (str): Output directory for Rosetta relax results.
    - ROSETTA_PATH (str): Path to the Rosetta relax binary.

## Voxels calculation
 
To calculate voxels using HTMD library we have used two different scripts, calculate_features_for_thermonet.py and calculate_features_for_orgnet.py. The first script calculate_features_for_thermonet.py is used to calculate features for reproduced Thermonet models, while calculate_features_for_orgnet.py is used to calculate features for OrgNet models. calculate_features_for_orgnetd.py provides a GLY correction, for GLY residues to be correctly positioned in the center of voxel grid. 

Both of them are designed to generate voxel-based feature datasets from two protein structure files — one corresponding to the wildtype protein and the other to its mutant version. 

Example usage:
```bash
python calculate_features_for_thermonet.py -iwt absolute/path/to/wildtype.pdb -imut absolute/path/to/mutant.pdb -o absolute/path/to/save/output --boxsize --voxelsize -v
```

- **-iwt/--input_wildtype_protein: Path to the wildtype PDB file.

- **-imut/--input_mutant_protein: Path to the mutant PDB file.

- **-o/--path_to_save: Directory where the output feature files will be stored.

- **--boxsize: Size of the bounding box around the mutation site.

- **--voxelsize: Size of each voxel.

- **-v/--verbose: Enable verbose logging for more detailed messages during execution.

Note, that:

- **1) output directory should contain several directories with your dataset name and following suffixes "_def_direct", "_defdif_direct", "_dif_direct", "_defdif_reverse", "_def_reverse", "_dif_reverse" and the "--path_to_save" should have your dataset name as a prefix "datasetname_" 
- **2) The script parses the mutant file name for wild-type residue, position, mutation, so the input mutant pdb filename should include this information.   


In the output creates several feature combinations:

- **Default Direct: Concatenates the wildtype and mutant features.

- **Default Reverse: Concatenates the mutant and wildtype features (order reversed).

- **Default + Difference Direct: Concatenates wildtype features with the difference between wildtype and mutant.

- **Default + Difference Reverse: Concatenates mutant features with the negative difference.

- **Differential Direct: Saves only the computed difference.

- **Differential Reverse: Saves the negative of the difference.

Its example usage is also included in /Dataset_processing/Dataset_processing.ipynb

## Orientation standardization

To orient the protein structures using orientation standardization use the following script - /orientation_standardization/orient_protein.py. This script is designed to perform a orientation standardization of a protein structure based on a specified mutation site. It parses a PDB file to extract the protein's coordinates and associated data, computes a normalized basis from the protein structure, and then aligns this basis with a predefined reference basis using a rotation transformation. The reoriented coordinates are then saved into a new PDB file with an updated naming convention.

Example usage:
```bash
python orient_protein.py -i absolute/path/to/input.pdb -o absolute/path/to/output_directory -mp mutation_position -fl structure_tag
```

-i/--input_file: Path to the input PDB file.

-o/--output_directory: Directory where the reoriented PDB file will be saved.

-mp/--mut_pos: Position of the mutation in the PDB file.

-fl/--flag: A tag to label the structure.

Its example usage is also included in /Dataset_processing/Dataset_processing.ipynb
