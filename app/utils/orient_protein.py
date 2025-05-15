import numpy as np
from scipy.spatial.transform import Rotation as R


from data_preprocessing.orientation_standardization.protein_orientation import (
    parse_pdb_to_array, parse_pdb_to_dict, get_protein_basis, get_Nn_coords_after_transf
)
from data_preprocessing.orientation_standardization.params import (
    Nn_ref_coords, basis_ref
)


def restore_pdb(file_name, coords, filename_out):
    
    with open(file_name, "r") as file:
        pdb_lines = file.readlines()
    
    
    before_atom = []
    for line in pdb_lines:
        if line[0:6] != 'ATOM  ':
            before_atom.append(line)
        else:
            break

    after_atom = []
    for line in pdb_lines:
        if line not in before_atom and line[0:6] != 'ATOM  ':
            after_atom.append(line)


    atom_lines = [l for l in pdb_lines if l[0:6] == 'ATOM  ']
    new_atom = []
    for line, crd in zip(atom_lines, coords):
        new_line = line[:30]+str(round(crd[0], 4)).rjust(8)+str(round(crd[1], 4)).rjust(8)+str(round(crd[2], 4)).rjust(8)+line[54:]
        new_atom.append(new_line)
    
    
    print(f"Writing orient to {filename_out}")
    with open(filename_out, "w")as f:
        for line in before_atom+new_atom+after_atom:
            print(line.rstrip("\n"), file=f)

    return filename_out


def orient_protein(
        input: str,
        output: str,
        mp: str,
        ):
    coords = parse_pdb_to_array(input)
    coords_dict = parse_pdb_to_dict(input)
    
    basis_normed = get_protein_basis(coords_dict, mp)
    
    rot = R.align_vectors(basis_normed, basis_ref)
    rotation_matrix = rot[0].as_matrix()
    print(rotation_matrix)
    
    conformation = np.matmul(coords, rotation_matrix)
    print(conformation)
    conformation = conformation + (Nn_ref_coords - conformation[get_Nn_coords_after_transf(coords, coords_dict, mp)])
    
    new_f_name = restore_pdb(input, conformation, output)
    print(new_f_name)
    
    return