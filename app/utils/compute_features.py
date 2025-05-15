from pathlib import Path
import numpy as np

from data_preprocessing.utils.pdb_utils_mod import compute_voxel_features


def calculate_features(
        filename_wt: Path,
        filename_mut: Path,
        wt: str,
        pos: str,
        prefix_out: Path,
        boxsize : int = 16,
        voxelsize : float = 1,
        ):

    if wt == "G":
        wt_type = "GLY"
    else:
        wt_type = None
        
    if wt_type == "GLY":
        features_mt, center = compute_voxel_features(pos, str(filename_mut), boxsize=boxsize, voxelsize=voxelsize)
        features, _ = compute_voxel_features(pos, str(filename_wt), 
            boxsize=boxsize, voxelsize=voxelsize, center_ref=center)
    else:
        features, center = compute_voxel_features(pos, str(filename_wt), boxsize=boxsize, voxelsize=voxelsize)
        features_mt, center = compute_voxel_features(pos, str(filename_mut), 
            boxsize=boxsize, voxelsize=voxelsize, center_ref=center)
        
    features_wt = np.delete(features, obj=6, axis=0)
    features_mt = np.delete(features_mt, obj=6, axis=0)
    
    dif = features_wt - features_mt
    
    features_combined_def_direct = np.concatenate((features_wt, features_mt), axis=0)
    features_combined_def_reverse = np.concatenate((features_mt, features_wt), axis=0)
    features_combined_defdif_direct = np.concatenate((features_wt, dif), axis=0)
    features_combined_defdif_reverse = np.concatenate((features_mt, -dif), axis=0)
    features_combined_dif_direct = dif
    features_combined_dif_reverse = -dif
    
    filename_def_direct = prefix_out.with_name(prefix_out.name + "_def_direct.npy")
    filename_def_reverse = prefix_out.with_name(prefix_out.name + "_def_reverse.npy")
    filename_defdif_direct = prefix_out.with_name(prefix_out.name + "_defdif_direct.npy")
    filename_defdif_reverse = prefix_out.with_name(prefix_out.name + "_defdif_reverse.npy")
    filename_dif_direct = prefix_out.with_name(prefix_out.name + "_dif_direct.npy")
    filename_dif_reverse = prefix_out.with_name(prefix_out.name + "_dif_reverse.npy")
    
    np.save(filename_def_direct, features_combined_def_direct)
    np.save(filename_def_reverse, features_combined_def_reverse)
    np.save(filename_defdif_direct, features_combined_defdif_direct)
    np.save(filename_defdif_reverse, features_combined_defdif_reverse)
    np.save(filename_dif_direct, features_combined_dif_direct)
    np.save(filename_dif_reverse, features_combined_dif_reverse)
    
    return