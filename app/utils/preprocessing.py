from typing import Dict, List, Optional
from pathlib import Path
import tempfile
from Bio.PDB import PDBParser, PDBIO, Select
import logging
import subprocess
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd

from .orient_protein import orient_protein
from .compute_features import calculate_features

file_dir = Path(__file__).parent
root_dir = file_dir.parent.parent
data_prep_dir = root_dir / "data_preprocessing"
ROSETTA_BIN = data_prep_dir / "rosetta/main/source/bin"
ROSETTA_PATH = ROSETTA_BIN / "relax.static.linuxgccrelease"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class ChainSelect(Select):
    def __init__(self, chain_id, remove_hetatm):
        self.chain_id = chain_id
        self.remove_hetatm = remove_hetatm

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        hetfield = residue.id[0]
        if self.remove_hetatm and hetfield != ' ':
            return False
        return True


def split_pdb_to_chains(
        filename: Path, 
        chain_ids: List[str],
        folder_out: Path,
        remove_hetatm : bool = True,
        ) -> Dict[str, Path]:
    structure = PDBParser(QUIET=True).get_structure("structure", filename)
    folder_out.mkdir(exist_ok=True)
    
    chain_filenames = {}
    for chain_id in chain_ids:
        io = PDBIO()
        io.set_structure(structure)
        filename_chain = folder_out / f"{chain_id}.pdb"
        io.save(str(filename_chain), ChainSelect(chain_id, remove_hetatm))
        chain_filenames[chain_id] = filename_chain
        logger.debug(f"Writing chain {chain_id} to file {filename_chain}")
        
    return chain_filenames


def run_relax(
        filename: Path,
        filename_out: Path,
        folder: Path,
        ):
    folder.mkdir(exist_ok=True)
    filename_log = folder / "rosetta_relax.log"
    cmd = (
        f"{ROSETTA_PATH.resolve()} "
        f"-in:file:s {filename.resolve()} "
        f"-out:path:pdb {filename_out.resolve()} "
        "-relax:constrain_relax_to_start_coords "
        "-out:no_nstruct_label "
        "-relax:ramp_constraints false "
        "-run:constant_seed "
        "-nstruct 1 "
        "-relax:fast "
    )
    logger.info(f"Executing: {cmd}")
    with open(filename_log, "w") as logfile:
        subprocess.run(cmd, shell=True, cwd=folder, check=True, stdout=logfile, stderr=subprocess.STDOUT)
    return


def run_relax_list(
        folder_in: Path, 
        names: List[str], 
        folder_out: Path, 
        folder_working: Path,
        threads : Optional[int] = None,
        skip_existing : bool = True,
        ):
    
    args_list = []
    for name in names:
        filename_out = folder_out / f"{name}.pdb"
        if skip_existing and filename_out.exists():
            logger.warning(f"Skipping relax, file exists: {filename_out}")
            continue
        args_list.append((
            folder_in / f"{name}.pdb",
            folder_out,
            folder_working / f"{name}",
        ))
    if len(args_list) == 0:
        return
    
    if threads is None or threads <= 0:
        n_cpu = multiprocessing.cpu_count()
        threads = max(1, n_cpu - 1)
    threads = min(threads, len(args_list))
    
    folder_out.mkdir(exist_ok=True)
    folder_working.mkdir(exist_ok=True)
    
    with multiprocessing.Pool(threads) as pool:
        list(tqdm(pool.starmap(run_relax, args_list), total=len(args_list), desc="Rosetta relax"))

    return


def run_mutation(
        filename: Path,
        folder_out: Path,
        folder: Path,
        suffix: str,
        chain_id: str,
        mut: str,
        pos: str,
        ):
    folder.mkdir(exist_ok=True)
    
    variant_resfile = folder / "mutation.resfile"
    with open(variant_resfile, "wt") as opf:
        opf.write("NATAA\n")
        opf.write("start\n")
        opf.write(f"{pos} {chain_id} PIKAA {mut}\n")
        
    cmd = [
        str(ROSETTA_PATH.resolve()),
        "-in:file:s", str(filename.resolve()),
        "-out:path:pdb", str(folder_out.resolve()),
        "-out:suffix", suffix,
        "-in:file:fullatom",
        "-relax:constrain_relax_to_start_coords",
        "-out:no_nstruct_label",
        "-relax:ramp_constraints", "false",
        "-relax:respect_resfile",
        "-packing:resfile", str(variant_resfile.resolve()),
        "-default_max_cycles", "200",
        "-out:file:scorefile", str(folder.resolve() / "relaxed.sc"),
        "--run:constant_seed",
        "-nstruct", "1",
        "-relax:default_repeats", "5",
    ]
    
    cmd_str = " ".join(cmd)
    logger.info(f"Executing: {cmd_str}")
    filename_log = folder / "relax.log"
    with open(filename_log, "w") as logfile:
        subprocess.run(cmd, cwd=folder, stdout=logfile, stderr=subprocess.STDOUT)
    
    return 


def run_mutations_list(
        folder_in: Path,
        folder_out: Path,
        folder_working: Path,
        mutations: pd.DataFrame,
        threads : Optional[int] = None,
        skip_existing : bool = True,
        ):
    args_list = []
    out_names = []
    for _, v in mutations.iterrows():
        suffix = f"_{v.wild_type}{v.position}{v.mutant}"
        out_name = f"{v.chain_id}{suffix}"
        out_names.append(out_name)
        filename_out = folder_out / f"{out_name}.pdb"
        if skip_existing and filename_out.exists():
            logger.warning(f"Skipping mutation, file exists: {filename_out}")
            continue
        args_list.append((
            folder_in / f"{v.chain_id}.pdb",
            folder_out,
            folder_working / f"{out_name}",
            suffix,
            v.chain_id, 
            v.mutant, 
            v.position,
        ))
    if len(args_list) == 0:
        return out_names
    
    if threads is None or threads <= 0:
        n_cpu = multiprocessing.cpu_count()
        threads = max(1, n_cpu - 1)
    threads = min(threads, len(args_list))
    
    folder_out.mkdir(exist_ok=True)
    folder_working.mkdir(exist_ok=True)
    
    with multiprocessing.Pool(threads) as pool:
        list(tqdm(pool.starmap(run_mutation, args_list), 
            total=len(args_list), desc="Rosetta mutation"))
    
    return out_names


def orient_dataset(
        folder_in: Path, 
        names_in: List[str], 
        folder_out: Path,
        names_out: List[str],
        pos: List[str],
        threads : Optional[int] = None,
        skip_existing : bool = True,
        ):
    folder_out.mkdir(exist_ok=True)
    args_list = []
    for name, p, name_out in zip(names_in, pos, names_out):
        filename_out = folder_out / f"{name_out}.pdb"
        if skip_existing and filename_out.exists():
            continue
        args_list.append((
            folder_in / f"{name}.pdb",
            folder_out / f"{name_out}.pdb",
            p,
        ))
    if len(args_list) == 0:
        return
    
    if threads is None or threads <= 0:
        n_cpu = multiprocessing.cpu_count()
        threads = max(1, n_cpu - 1)
    threads = min(threads, len(args_list))
    
    with multiprocessing.Pool(threads) as pool:
        list(tqdm(pool.starmap(orient_protein, args_list), 
            total=len(args_list), desc="Orienting dataset"))
    
    return


def calculate_features_list(
        folder_in_wt: Path,
        folder_in_mut: Path,
        names_wt: List[Path],
        names_mut: List[Path],
        folder_out: Path,
        pos_list: List[str],
        wt_list: List[str],
        boxsize : int = 16,
        voxelsize : float = 1,
        threads : Optional[int] = None,
        skip_existing : bool = True,
        ):
    
    args_list = []
    for name_wt, name_mt, wt, pos in zip(names_wt, names_mut, wt_list, pos_list):
        prefix_out = folder_out / name_mt
        filenames_out = [prefix_out.with_name(f"{prefix_out.name}_{v}.npy")
            for v in ["def_direct", "def_reverse", "defdif_direct", "defdif_reverse", "dif_direct", "dif_reverse"]]
        if skip_existing and all(f.exists() for f in filenames_out):
            continue
        args_list.append((
            folder_in_wt / f"{name_wt}.pdb",
            folder_in_mut / f"{name_mt}.pdb",
            wt,
            pos,
            folder_out / name_mt,
            boxsize,
            voxelsize,
        ))
    if len(args_list) == 0:
        return
    
    if threads is None or threads <= 0:
        n_cpu = multiprocessing.cpu_count()
        threads = max(1, n_cpu - 1)
    threads = min(threads, len(args_list))

    folder_out.mkdir(exist_ok=True)
    
    with multiprocessing.Pool(threads) as pool:
        list(tqdm(pool.starmap(calculate_features, args_list), 
            total=len(args_list), desc="Calculating features"))
    return
        

def main(
        filename_pdb: Path,
        filename_out: Path,
        mutations: pd.DataFrame,
        folder : Optional[Path] = None,
        # 
        run_relax : bool = True,
        threads : Optional[int] = None,
        skip_existing : bool = True,
        ):
    if folder is None:
        tempdir = tempfile.TemporaryDirectory()
        folder = Path(tempdir.name)
    else:
        tempdir = None
    folder.mkdir(exist_ok=True)

    # extract chains
    chain_ids = np.unique(mutations.chain_id.values).tolist()
    folder_chains = folder / "chains"
    split_pdb_to_chains(filename_pdb, chain_ids, folder_chains, remove_hetatm=True)
    
    # run relax
    if run_relax:
        folder_relax = folder / "relaxed"
        folder_relax_working = folder / "relaxed_working"
        run_relax_list(folder_chains, chain_ids, folder_relax, 
            folder_relax_working, threads=threads, skip_existing=skip_existing)
    else:
        folder_relax = folder_chains
    
    # run mutations
    folder_mutations = folder / "mutations"
    folder_mutations_working = folder / "mutations_working"
    names_mut = run_mutations_list(folder_relax, folder_mutations, 
        folder_mutations_working, mutations, threads=threads, skip_existing=skip_existing)
    
    # orient dataset
    folder_oriented_wt = folder / "oriented_wt"
    folder_oriented_mt = folder / "oriented_mt"
    names_mt_ori = [f"{v.chain_id}_{v.wild_type}{v.position}{v.mutant}"
        for _, v in mutations.iterrows()]
    names_wt_ori = [f"{v.chain_id}_{v.position}" for _, v in mutations.iterrows()]
    orient_dataset(folder_relax, mutations.chain_id.values, 
        folder_oriented_wt, names_wt_ori, mutations.position.values,
        threads=threads, skip_existing=skip_existing,
    )
    orient_dataset(folder_mutations, names_mut, folder_oriented_mt, 
        names_mt_ori, mutations.position.values,
        threads=threads, skip_existing=skip_existing,
    )
    
    # calculate features
    folder_features = folder / "features"
    calculate_features_list(
        folder_in_wt=folder_oriented_wt,
        folder_in_mut=folder_oriented_mt,
        names_wt=names_wt_ori,
        names_mut=names_mt_ori,
        folder_out=folder_features,
        pos_list=mutations.position.values,
        wt_list=mutations.wild_type.values,
        threads=threads,
        skip_existing=skip_existing,
    )
    
    # save npy
    features = []
    for name in names_mut:
        f = np.load(folder_features / f"{name}_defdif_direct.npy")
        features.append(f)
    features = np.array(features)
    np.save(filename_out, features)
    return features