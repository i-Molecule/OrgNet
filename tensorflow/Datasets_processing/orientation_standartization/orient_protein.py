#!/usr/bin/env python3

import os, sys
import time
from argparse import ArgumentParser
import numpy as np
import scipy as sc
from scipy.spatial.transform import Rotation as R
import pandas as pd
import params as pr
import protein_orientation as PO

def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', dest='input', type=str,
                        help='Pdb file.')
    parser.add_argument('-o', '--output_directory', dest='output', type=str,
                        help='Path to output directory.')
    parser.add_argument('-mp', '--mut_pos', dest='mp', type=str,
                        help='Position of the mutation in pdb file')
    parser.add_argument('-fl', '--flag', dest='flag', type=str,
                        help='Flag to tag the structure')
    args = parser.parse_args()
    # do any necessary argument checking here before returning
    return args
    
    
    
def main():
    """

    Returns
    -------

    """
    args = parse_cmd()
    
    
    coords = PO.parse_pdb_to_array(args.input)
    coords_dict = PO.parse_pdb_to_dict(args.input)
    
    basis_normed = PO.get_protein_basis(coords_dict, args.mp)
    
    rot = R.align_vectors(basis_normed, pr.basis_ref)
    rotation_matrix = rot[0].as_matrix()
    print(rotation_matrix)
    
    
    conformation = np.matmul(coords, rotation_matrix)
    print(conformation)
    conformation = conformation + (pr.Nn_ref_coords - conformation[PO.get_Nn_coords_after_transf(coords, coords_dict, args.mp)])
    
    
    new_f_name = PO.restore_pdb(args.input, conformation, args.output, args.flag)
    print(new_f_name)

    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print('orient_protein.py took', elapsed, 'seconds to generate the oriented protein.')
