import os, sys
import numpy as np
import scipy as sc
import pandas as pd
import warnings

def parse_pdb_to_dict(file_name):
    
    with open(file_name, "r") as file:
        pdb_lines = file.readlines()
    
    atom_lines = [l for l in pdb_lines if l[0:6] == 'ATOM  ']
    dict_coord = {}
    
    for line in atom_lines:
        
        atom_name = line[12:16].replace(' ', '')
        res_name = line[17:20]
        res_num = int(line[22:26].strip())
        
        if (res_name, res_num) not in dict_coord.keys():
            dict_coord[(res_name, res_num)] = {}
         
        
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        
        
        dict_coord[(res_name, res_num)][atom_name] = np.array([x,y,z])
    
    return dict_coord
    
    
def parse_pdb_to_array(file_name):
    
    with open(file_name, "r") as file:
        pdb_lines = file.readlines()
    
    atom_lines = [l for l in pdb_lines if l[0:6] == 'ATOM  ']
    dict_coord = {}
    tot_coords = []
    
    for line in atom_lines:
        
        atom_name = line[12:16].replace(' ', '')
        res_name = line[17:20]
        
        
        res_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        
        tot_coords.append(np.array([x,y,z]))
    
    
    tot_coords = np.stack((tot_coords))
    points_coords = tot_coords
    
    return points_coords

def restore_pdb(file_name, coords, out_dir, flag):
    
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
    
    
    new_f_name = file_name.split("/")[-1].split(".")[0] 
    with open(out_dir+f"{new_f_name}_{flag}_oriented.pdb", "w")as f:
        for line in before_atom+new_atom+after_atom:
            print(line.rstrip("\n"), file=f)

    return out_dir+f"{new_f_name}_{flag}_oriented.pdb"

def get_Cn_1_coords(coords_dict, pos):
    res_num = int(pos) -1
    for item in coords_dict.keys():
        if int(item[1]) == int(res_num):
            res = item
    #print(coords_dict[res])
    return coords_dict[res]["C"]

def get_Calphan_coords(coords_dict, pos):
    res_num = pos
    for item in coords_dict.keys():
        if int(item[1]) == int(res_num):
            res = item
    #print(coords_dict[res])
    return coords_dict[res]["CA"]

def get_Nn_coords(coords_dict, pos):
    res_num = pos
    for item in coords_dict.keys():
        if int(item[1]) == int(res_num):
            res = item
    #print(coords_dict[res])
    return coords_dict[res]["N"]
    
def get_Nn_coords_after_transf(coords, coords_dict, pos):
    for indx, crd in enumerate(coords):
        if np.all(crd == get_Nn_coords(coords_dict, pos)):
            Nn_coord_ind = indx
    return Nn_coord_ind

def vector_projection(v,w): return (np.dot(v, w)/(np.linalg.norm(w))**2)*w

def check_vector_dir(vector, vector_x):
    
    if np.sign(vector.dot(vector_x)) == 1:
        return True
    else:
        return False

def get_protein_basis(coords_dict, pos):
    
    vector_x = get_Nn_coords(coords_dict, pos) - get_Cn_1_coords(coords_dict, pos)
    
    
    vector_l = get_Calphan_coords(coords_dict, pos) - get_Nn_coords(coords_dict, pos)
    
    if check_vector_dir(vector_l, vector_x) == False:
        warnings.warn("Smth wrong with basis orientation. Check the structure visually!")
    
    proj_l_x = vector_projection(vector_l,vector_x)
    vector_y = vector_l - proj_l_x
    
    vector_z = np.cross(vector_x, vector_y)
    
    vector_x_normed = vector_x/np.linalg.norm(vector_x)
    vector_y_normed = vector_y/np.linalg.norm(vector_y)
    vector_z_normed = vector_z/np.linalg.norm(vector_z)

    basis_normed = np.stack([
        vector_x_normed,
        vector_y_normed,
        vector_z_normed
    ])

    return basis_normed


def dots_to_pdb(file_name, coords):
    
    with open(file_name, "w") as file:

        elements = [
            "HETATM    1 NA    NA A   1     -13.301  34.777  39.670  1.00  0.00          NA",
            "HETATM    1 AU    AU A   1     -13.301  34.777  39.670  1.00  0.00          NA",
#           "HETATM    1 CL    CL A   1     -13.301  34.777  39.670  1.00  0.00          CL",
            "HETATM    1 BR    BR A   1     -13.301  34.777  39.670  1.00  0.00          BR",
            "HETATM    1 ZN    ZN A   1     -13.301  34.777  39.670  1.00  0.00          ZN",
#           "HETATM    1 ZN    ZN A   1     -13.301  34.777  39.670  1.00  0.00          ZN"
        ]

        for crd, line in zip(coords, elements):
            new_line = line[:30]+str(round(crd[0], 4)).rjust(8)+str(round(crd[1], 4)).rjust(8)+str(round(crd[2], 4)).rjust(8)+line[54:]
            
            
            print(new_line.rstrip("\n"), file=file)

        print("ENDMDL", file = file)
        print("END", file = file)

    return file_name

def get_basis_points(basis_normed, coords_dict, pos, filename):
    
    vector_x , vector_y, vector_z = basis_normed[0], basis_normed[1], basis_normed[2]
    
    #end of y vector
    Q = vector_y
    #end of the vector z
    M = vector_z
    #end of the vector x
    S = vector_x
    
    dots_coords = np.stack([
        Q,
        M,
        S,
        np.array([0,0,0])
    ])
    dots_to_pdb(filename, dots_coords)
    
    return filename
    
    
