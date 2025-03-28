#!/usr/bin/env python3

import os
import sys
from utils import pdb_utils_mod
import time
from argparse import ArgumentParser
import numpy as np

"""
This script is a slightly modified version of Thermonet script for feature calculation. The only difference is in careful GLY management for its grid to be identical for reverse and direct mutations. Sсript uses htmd library to calculate the grids.

Note that if you use it via Notebook, it is better to include "%env HTMD_NONINTERACTIVE=1"

Required arguments:
-iwt : path to input .pdb file
-iwt : path to input .pdb file
-o : path to output directory
--boxsize : size of the bounding box around the mutation site # default 16
--voxelsize : size of the voxel #default 1

Optional arguments:
-v : True/False, enable the verbose messages

Note that this script utilizes input parsing, so the input MT files should have the following format {pdb_chain}_{WT}{POS}{MT}_ ... for them to be processed correctly. 

This script calculates 3 voxel grids, which were noted in the article: def - [WT, MUT], defdif - [WT,(WT-MUT)], dif - [WT-MUT]. It outputs features to the directories that should be created beforehand. So the output directory should have the following structure: {dataset}_.../, than in this directory should be 6 directories (if you want only one feature type please comment the line accordingly), 3 for direct features and 3 for reverse, which is used for data augmentation:

/{dataset_name}_def_direct/,
/{dataset_name}_defdif_direct/,
/{dataset_name}_dif_direct/,
/{dataset_name}_def_direct/,
/{dataset_name}_defdif_direct/,
/{dataset_name}_dif_direct/,

And in each of those directories the directory /{pdb_chain}/ should be created.

Example script to run this script and create the following directories:

ptc = "/home/nata/work/Projects/Protein_stability_prediction/S669/relaxed_chains_total_ori/"
ptofeat = "/home/nata/work/Projects/Protein_stability_prediction/S669/features/S669_ori/"
bad_fold = []

for folder in os.listdir(ptc):
    
    try:
        
        for ff in os.listdir(ptofeat):
            if os.path.exists(ptofeat+"/"+ff+"/"+folder+"/") == False:
                os.mkdir(ptofeat+"/"+ff+"/"+folder+"/")

        folder_path = ptc+folder+"/"
        mut_prots = [folder_path+f for f in os.listdir(folder_path) if "_relaxed_0_oriented.pdb" in f]

        for pdb_file in mut_prots:

            pos = pdb_file.split("/")[-1].split("_")[2][1:-1]
            path_to_wt = ptc+f"{folder}/{folder}_relaxed_{pos}_wt_oriented.pdb"#1A0FA_fixed_relaxed_11_wt_oriented.pdb
            path_to_mut = pdb_file
            print(pos, path_to_wt, path_to_mut)
            print(os.path.exists(path_to_mut), os.path.exists(path_to_wt))
            pts = ptofeat

            os.system(f"python /home/nata/work/Programs/ThermoNet/ThermoNet/gends_2prots_1_TOTAL1.py  -iwt {path_to_wt} -imut {path_to_mut} -o {pts} --boxsize 16 --voxelsize 1")

    except:
        
        bad_fold.append(folder)



Example usage:
gends_2prots_1_TOTAL1.py  -iwt {path_to_wt} -imut {path_to_mut} -o {pathtosave} --boxsize 16 --voxelsize 1")
"""


def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-iwt', '--input_wildtype_protein',
                        help='Pdb file for WT.')
    parser.add_argument('-imut', '--input_mutant_protein',
                        help='Pdb file for Mut.')
    parser.add_argument('-o', '--path_to_save', dest='output', type=str,
                        help='Path to save output features.')
    parser.add_argument('--boxsize', dest='boxsize', type=int,
                        help='Size of the bounding box around the mutation site.')
    parser.add_argument('--voxelsize', dest='voxelsize', type=int,
                        help='Size of the voxel.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Whether to print verbose messages from HTMD function calls.')

    args = parser.parse_args()
    # do any necessary argument checking here before returning
    return args


def main():
    
    args = parse_cmd()
    print(args)
    wt_rot = args.input_wildtype_protein
    mut_rot = args.input_mutant_protein
    rotations = None
    print(args.verbose)
    
    info = mut_rot.split("/")[-1].split("_")[2]#-2
    pdb_chain = mut_rot.split("/")[-1].split("_")[0]
    wt, mt, pos = info[0], info[-1], info[1:-1]
    
    if wt == "G":
            wt_type = "GLY"
    else:
            wt_type = None     
    
    print("args parsed")
    print(pos, wt, mt)
    
    
    features_wt_all = {}

    if wt_type=="GLY":
            
            features_mt, center = pdb_utils_mod.compute_voxel_features(pos, mut_rot, boxsize=args.boxsize,
                        voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations)
            print("Mut features calculated")

            features, _ = pdb_utils_mod.compute_voxel_features(pos, wt_rot, boxsize=args.boxsize,
                            voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations, center_ref = center)
            print("Wt features calculated")

    else:

            features, center = pdb_utils_mod.compute_voxel_features(pos, wt_rot, boxsize=args.boxsize,
                            voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations)
            print("Wt features calculated")

            features_mt, center = pdb_utils_mod.compute_voxel_features(pos, mut_rot, boxsize=args.boxsize,
                        voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations, center_ref = center)
            print("Mut features calculated")


    features_wt_all[pdb_chain + pos] = features
    
    
    features_wt = features_wt_all[pdb_chain + pos]
    features_wt = np.delete(features_wt, obj=6, axis=0)
    features_mt = np.delete(features_mt, obj=6, axis=0)

    dif = features_wt - features_mt
    
    dataset_name = args.output.split("/")[-2].split("_")[0]
    
    #default features
    features_combined_1 = np.concatenate((features_wt, features_mt), axis=0)
    np.save(args.output+f'{dataset_name}_def_direct/{pdb_chain}/{pdb_chain}_{wt}{pos}{mt}.npy', features_combined_1)
    print("Default direct(wt, mt) features saved")
    
    features_combined_2 = np.concatenate((features_mt, features_wt), axis=0)
    np.save(args.output+f'{dataset_name}_def_reverse/{pdb_chain}/{pdb_chain}_{wt}{pos}{mt}.npy', features_combined_2)
    print("Default reverse(mt, wt) features saved")
    
    #Default+difference features
    features_combined_3 = np.concatenate((features_wt, dif), axis=0)
    np.save(args.output+f'{dataset_name}_defdif_direct/{pdb_chain}/{pdb_chain}_{wt}{pos}{mt}.npy', features_combined_3)
    print("Default+dif direct(wt, dif) features saved")
    
    features_combined_4 = np.concatenate((features_mt, -dif), axis=0)
    np.save(args.output+f'{dataset_name}_defdif_reverse/{pdb_chain}/{pdb_chain}_{wt}{pos}{mt}.npy', features_combined_4)
    print("Reverse+-dif reverse(mt, -dif) features saved")
    
    #Difference features
    features_combined_5 = dif
    np.save(args.output+f'{dataset_name}_dif_direct/{pdb_chain}/{pdb_chain}_{wt}{pos}{mt}.npy', features_combined_5)
    print("Differential direct(dif) features saved")
    
    features_combined_6 = -dif
    np.save(args.output+f'{dataset_name}_dif_reverse/{pdb_chain}/{pdb_chain}_{wt}{pos}{mt}.npy', features_combined_6)
    print("Differential reverse(-dif) features saved")
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print('gends_2prots_1_TOTAL.py took', elapsed, 'seconds to generate the dataset.')
