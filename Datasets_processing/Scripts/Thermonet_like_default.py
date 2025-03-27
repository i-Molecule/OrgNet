#!/usr/bin/env python3

import os
import sys
from utils import pdb_utils
import time
from argparse import ArgumentParser
import numpy as np


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
    
    info = mut_rot.split("/")[-1].split("_")[-2]#-2
    pdb_chain = mut_rot.split("/")[-1].split("_")[0]
    wt, mt, pos = info[0], info[-1], info[1:-1]
    
    print("args parsed")
    print(pos, wt, mt)
    
    
    features_wt_all = {}
    features = pdb_utils.compute_voxel_features(pos, wt_rot, boxsize=args.boxsize,
                            voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations)
    features_wt_all[pdb_chain + pos] = features
    print("Wt features calculated")
    features_mt = pdb_utils.compute_voxel_features(pos, mut_rot, boxsize=args.boxsize,
                        voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations)
    print("Mut features calculated")
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
