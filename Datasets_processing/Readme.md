# Dataset processing and Thermonet-like models training

This repository contains Jupyter Notebooks for dataset processing, training, and running inference with Thermonet-like models. The project is organized into three 4 primary folders:

- **/Dataset_processing/Dataset_processing.ipynb:** Prepares the dataset (generates mutants, performs standardized orientation and calculates features for OrgNet and Thermonet-like models) for model training.
- **/Training_Thermonet_like/Thermonet_like_models_training.ipynb:** Contains the model architecture, training routines, and evaluation metrics.
- **/Inference_thermonet_like/Thermonet_like_models_inderence.ipynb:** Provides scripts for running inference on new data using the trained model.

## Prerequisites

Make sure you have Python 3.10 or later installed. The following packages are required (you may have additional dependencies based on your specific project):

- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow or pytorch (depending on your implementation)
- jupyter
- etc

Full python packages and their version can be found in requirements.txt

### Install Rosetta 3.13
1. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
2. Download Rosetta 3.13 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.

### Install HTMD - the main library for voxels calculation
The free version of HTMD is free to non-commercial users although it does not come with full functionality. But to use it with ThermoNet, the free version is sufficient. You can either install it by following the instructions listed [here](https://software.acellera.com/install-htmd.html), or by running
```bash
conda env create --name ds_processing --file Thermonet_like.yaml
```
The above command will create a conda environment and install all dependencies so that one can run scripts to make input tensors.

## Mutant structures generation

To generate mutant structures use run_rosetta_relax function in /Dataset_processing/Dataset_processing.ipynb. 

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
 
To calculate voxels using HTMD library we have used two different scripts, /Dataset_processing/Scripts/Thermonet_like_default.py and /Dataset_processing/Scripts/Thermonet_like_modified.py. The first script Thermonet_like_default.py is used to calculate features for reproduced Thermonet models, while Thermonet_like_modified.py is used to calculate features for OrgNet models. Thermonet_like_modified.py provides a GLY correction, for GLY residues to be correctly positioned in the center of voxel grid. 

Both of them are designed to generate voxel-based feature datasets from two protein structure files — one corresponding to the wildtype protein and the other to its mutant version. 

Example usage:
```bash
python script_name.py -iwt path/to/wildtype.pdb -imut path/to/mutant.pdb -o path/to/save/output --boxsize --voxelsize -v
```

- **-iwt/--input_wildtype_protein: Path to the wildtype PDB file.

- **-imut/--input_mutant_protein: Path to the mutant PDB file.

- **-o/--path_to_save: Directory where the output feature files will be stored.

- **--boxsize: Size of the bounding box around the mutation site.

- **--voxelsize: Size of each voxel.

- **-v/--verbose: Enable verbose logging for more detailed messages during execution.

Note, that:

- **1) output directory should contain several directories with following suffixes "_def_direct", "_defdif_direct", "_dif_direct", "_defdif_reverse", "_def_reverse", "_dif_reverse"
- **2) The script parses the mutant file name for wild-type residue, position, mutation, so it should include this information.   


In the output creates several feature combinations:

- **Default Direct: Concatenates the wildtype and mutant features.

- **Default Reverse: Concatenates the mutant and wildtype features (order reversed).

- **Default + Difference Direct: Concatenates wildtype features with the difference between wildtype and mutant.

- **Default + Difference Reverse: Concatenates mutant features with the negative difference.

- **Differential Direct: Saves only the computed difference.

- **Differential Reverse: Saves the negative of the difference.

Its example usage is also included in /Dataset_processing/Dataset_processing.ipynb

## Orientation standardization

To orient the protein structures using orientation standardization /Dataset_processing/Scripts/proteinorientation/orient_protein.py. This script is designed to perform a orientation standardization of a protein structure based on a specified mutation site. It parses a PDB file to extract the protein's coordinates and associated data, computes a normalized basis from the protein structure, and then aligns this basis with a predefined reference basis using a rotation transformation. The reoriented coordinates are then saved into a new PDB file with an updated naming convention.

Example usage:
```bash
python orient_protein.py -i path/to/input.pdb -o path/to/output_directory -mp mutation_position -fl structure_tag
```

-i/--input_file: Path to the input PDB file.

-o/--output_directory: Directory where the reoriented PDB file will be saved.

-mp/--mut_pos: Position of the mutation in the PDB file.

-fl/--flag: A tag to label the structure.

Its example usage is also included in /Dataset_processing/Dataset_processing.ipynb
