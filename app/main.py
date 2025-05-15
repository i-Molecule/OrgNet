import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import streamlit as st
import tempfile
import pandas as pd
from types import SimpleNamespace
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

file_dir = Path(__file__).parent
root_dir = file_dir.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "OrgNet"))

from app.utils.preprocessing import main as preprocess
from OrgNet.predict import call_predict
from app.utils.visualization import StructureVisualizer



standard_amino_acids = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
aa_3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def get_model_options():
    st.subheader("Model")
    
    model_name = st.selectbox("Model Name", 
        options=["OrgNet", "ThermoNet", "ThermoNet_steerable"],
        index=0, key="model_name",
    )
    training_data = st.selectbox("Training Data",
        options=["Q3214", "S2648_V"],
        index=0, key="training_data",
    )
    relax_input = st.checkbox("Relax Input Structure", value=False, key="relax_input")
    
    rotations = st.checkbox("Additional rotations", value=False, key="rotations")
    
    return {
        "model_name": model_name,
        "training_data": training_data,
        "relax_input": relax_input,
        "rotations": rotations,
    }
    

def get_list_of_residues(filename: Path) -> Dict[str, Tuple[str, str]]:
    structure = PDBParser(QUIET=True).get_structure("loaded", filename)
    model = structure[0]
    
    chain_residues = {}
    for chain in model:
        residues = []
        for residue in chain:
            if is_aa(residue, standard=True):
                res_id = residue.id[1]
                res_name = residue.get_resname()
                residues.append((res_id, res_name))
        if len(residues) > 0:
            chain_residues[chain.id] = residues
    
    return chain_residues


@st.fragment
def select_mut(chain_residues: Dict[str, List[Tuple[str, str]]]) -> Tuple[str, str, str, str]:
    if len(chain_residues) == 0:
        return None, None, None, None
    
    chain_ids = list(chain_residues.keys())
    
    col1, col2, col3 = st.columns([1, 2, 2])
    if st.session_state.orgnet_vars.edit_mutation is not None:
        mutation_edit = st.session_state.orgnet_vars.edit_mutation
        chain_edit, res_id_edit, res_name_edit, mut_edit = mutation_edit
    else:
        chain_edit, res_id_edit, res_name_edit, mut_edit = None, None, None, None
        
    with col1:
        selected_chain = st.selectbox("Select Chain", chain_ids,
            index=(0 if chain_edit is None else chain_ids.index(chain_edit)),
            key="selected_chain",
        )
    
    filtered_residues = chain_residues[selected_chain]
    residue_options = [f"{i} {t}" for i, t in filtered_residues]
    with col2:
        selected_residue = st.selectbox("Select Position", residue_options,
            index=(0 if res_id_edit is None else filtered_residues.index((res_id_edit, res_name_edit))),
            key="selected_residue",
        )
    
    res_index = residue_options.index(selected_residue)
    res_id, res_name = filtered_residues[res_index]
    mutation_options = [aa for aa in standard_amino_acids if aa != res_name]
    with col3:
        selected_mutation = st.selectbox("Select Mutation", mutation_options,
            format_func=lambda x: f"{x}, {aa_3to1[x]}",
            index=(0 if mut_edit is None else mutation_options.index(mut_edit)),
            key="selected_mutation",
        )
    return selected_chain, res_id, res_name, selected_mutation
    

def add_mutation(
        chain_residues: Dict[str, List[Tuple[str, str]]],
        ) -> List[Tuple[str, str, str, str]]:
    st.subheader("Add a mutation")
    
    selected_chain, res_id, res_name, selected_mutation = select_mut(chain_residues)        
    if selected_chain and res_id and selected_mutation:
        submit_button = st.button(label="Add Mutation")
        
        if submit_button:
            mutation_entry = (selected_chain, res_id, res_name, selected_mutation)
            if st.session_state.orgnet_vars.edit_index is not None:
                st.session_state.orgnet_vars.mutations[st.session_state.orgnet_vars.edit_index] = mutation_entry
            elif mutation_entry not in st.session_state.orgnet_vars.mutations:
                st.session_state.orgnet_vars.mutations.append(mutation_entry)
            st.session_state.orgnet_vars.edit_index = None
            st.session_state.orgnet_vars.edit_mutation = None
    return

def remove_mutation(index: int):
    st.session_state.orgnet_vars.mutations.pop(index)
    
def edit_mutation(index: int):
    st.session_state.orgnet_vars.edit_index = index
    st.session_state.orgnet_vars.edit_mutation = st.session_state.orgnet_vars.mutations[index]

def display_mutations():
    for idx, mut in enumerate(st.session_state.orgnet_vars.mutations):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(
                f"{idx+1}. Chain {mut[0]}, Residue {mut[1]} "
                f"({mut[2]}, {aa_3to1[mut[2]]} → {mut[3]}, {aa_3to1[mut[3]]})"
            )
        with col2:
            if st.button("Edit", key=f"edit_{idx}"):
                edit_mutation(idx)
        with col3:
            if st.button("Remove", key=f"remove_{idx}"):
                remove_mutation(idx)
                st.rerun()


@st.cache_data(hash_funcs={StructureVisualizer: lambda _: None})
def show_protein(
        filename: Path,
        mutations: List[Tuple[str, str, str, str]] = [],
        preds : Optional[pd.DataFrame] = None
        ):
            
    vis = StructureVisualizer()
    
    vis.show_structure(
        filename,
        show_cartoon=True,
        show_surface=True,
        show_sticks=False,
        cartoon_color="white",
    )
    
    vis.view.render()
    vis.view.zoomTo()
        
    preds = st.session_state.orgnet_vars.preds
    
    if preds is None:
        if len(mutations) > 0:
            res_mut = {}
            for chain_id, pos, wt, mut in mutations:
                res_mut.setdefault((chain_id, pos, aa_3to1[wt]), []).append(aa_3to1[mut])
            vis.show_sel_mutations(res_mut)
    else:
        min_val = preds.ddg_mean.values.min()
        max_val = preds.ddg_mean.values.max()
        res_scores_for_vis = {}
        for chain_id, pos, wt, mut, ddg, ddg_std in zip(preds.chain_id, 
                preds.position, preds.wild_type, preds.mutant, preds.ddg_mean, preds.ddg_std):
            res_scores_for_vis.setdefault((chain_id, pos, wt), {})[mut] = (ddg, ddg_std)
        vis.show_sel_residue_scores(res_scores_for_vis, min_value=min_val, max_value=max_val)
    
    html = vis.view._make_html()
    return html
    

def make_predictions(
        filename_input: Path,
        folder: Path,
        mutations_df: pd.DataFrame,
        config: Dict[str, Any],
        ):
    filename_input_csv = folder / "input_mutations.csv"
    mutations_df.to_csv(filename_input_csv)
    filename_features = folder / "features.npy"
    preprocess(
        filename_pdb=filename_input,
        filename_out=filename_features,
        mutations=mutations_df,
        folder=folder / "working",
        run_relax=config["relax_input"],
    )
    filename_out = folder / "output.csv"
    call_predict(
        path_to_X=filename_features,
        save_to=filename_out,
        device="cpu",
        training_data=config["training_data"],
        model_name=config["model_name"],
        samples_dim=0,
        channels_dim=1,
        random_rotations=False,
        fully_rotated=config["rotations"],
    )
    
    df_preds = pd.read_csv(filename_out)
    df_preds = df_preds.drop(columns=["id"])
    
    df_preds = pd.concat([mutations_df, df_preds], axis=1)
    df_preds = df_preds.rename(columns={
        "mean_predictions": "ddg_mean",
        "std_predictions_folds": "ddg_std",
    })
    return df_preds


def show_predictions(df_preds: pd.DataFrame):
    
    
    df_preds_styled = df_preds.style.format(
        {col: "{:.3f}" for col in ["ddg_mean", "ddg_std"]})
    
    ddg = df_preds.ddg_mean.values
    inds_pos = np.where(ddg > 0)[0]
    inds_neg = np.where(ddg < 0)[0]
    vals_for_colors = np.zeros(len(df_preds))
    if len(inds_pos) > 0:
        m = ddg[inds_pos].max()
        vals_for_colors[inds_pos] = ddg[inds_pos] / m
    if len(inds_neg) > 0:
        m = np.abs(ddg[inds_neg]).max()
        vals_for_colors[inds_neg] = ddg[inds_neg] / m
    df_preds_styled = df_preds_styled.background_gradient(
        cmap="bwr",
        axis=0,
        subset=["ddg_mean"],
        vmin=-1,
        vmax=1,
        gmap=vals_for_colors,
    )
    st.dataframe(df_preds_styled)
    
    csv_preds = df_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predicted ddG",
        data=csv_preds,
        file_name="orgnet_preds.csv",
        mime="text/csv",
        on_click="ignore",
    )
    return



def main():
    st.set_page_config(page_title="OrgNet: prediction of mutation ddG", layout="wide")
    st.title("OrgNet: prediction of mutation ddG")
    
    if "orgnet_vars" not in st.session_state:
        st.session_state.orgnet_vars = SimpleNamespace(
            config=None,
            temp_dir=None,
            file_path=None,
            mutations=[],
            edit_mutation=None,
            edit_index=None,
            uploaded_file=None,
            chain_residues=None,
            preds=None,
        )
    
    left_col, right_col = st.columns([2, 3])
    
    with left_col:
        st.header("Input Parameters")
        st.session_state.orgnet_vars.config = get_model_options()
        
        uploaded_file = st.file_uploader("Upload a file", type="pdb")
        if uploaded_file is not None:
            if st.session_state.orgnet_vars.uploaded_file != uploaded_file:
                temp_dir = tempfile.TemporaryDirectory()
                temp_dir_name = Path(temp_dir.name)
                
                file_path = temp_dir_name / uploaded_file.name
                print(f"writing uploaded_file to {file_path}")
                open(file_path, "wb").write(uploaded_file.getvalue())
                
                chain_residues = get_list_of_residues(file_path)
                
                if len(chain_residues) == 0:
                    st.error("Input structure does not contain protein chains")
                
                st.session_state.orgnet_vars.uploaded_file = uploaded_file
                st.session_state.orgnet_vars.chain_residues = chain_residues
                st.session_state.orgnet_vars.mutations = []
                st.session_state.orgnet_vars.edit_index = None
                st.session_state.orgnet_vars.edit_mutation = None
                
                st.session_state.orgnet_vars.temp_dir = temp_dir
                st.session_state.orgnet_vars.file_path = file_path
                st.session_state.orgnet_vars.preds = None
            
        else:
            st.session_state.orgnet_vars.mutations = []
            
        if st.session_state.orgnet_vars.chain_residues is not None:
            add_mutation(st.session_state.orgnet_vars.chain_residues)
            display_mutations()
            
        if st.button("Submit", 
                disabled=len(st.session_state.orgnet_vars.mutations) == 0):
            if st.session_state.orgnet_vars.config is None:
                st.info("Model config is not specified")
            elif st.session_state.orgnet_vars.uploaded_file is None:
                st.info("Upload a file")
            elif len(st.session_state.orgnet_vars.mutations) == 0:
                st.info("Specify at least one mutation")
            else:
                mutations = [
                    {"chain_id": c, "position": ri, "wild_type": aa_3to1[wt], "mutant": aa_3to1[mt]}
                    for c, ri, wt, mt in st.session_state.orgnet_vars.mutations
                ]
                df_mutations = pd.DataFrame(mutations)
                
                st.session_state.orgnet_vars.preds = make_predictions(
                    st.session_state.orgnet_vars.file_path,
                    Path(st.session_state.orgnet_vars.temp_dir.name),
                    df_mutations,
                    config=st.session_state.orgnet_vars.config,
                )
            
    
    with right_col:
        st.header("Output")
        if st.session_state.orgnet_vars.file_path is not None:
            html = show_protein(
                filename=st.session_state.orgnet_vars.file_path,
                mutations=(st.session_state.orgnet_vars.mutations if st.session_state.orgnet_vars.preds is None else []),
                preds=st.session_state.orgnet_vars.preds,
            )
            st.components.v1.html(html, width=800, height=600)
        
        if st.session_state.orgnet_vars.preds is not None \
                and st.session_state.orgnet_vars.file_path is not None:
            show_predictions(st.session_state.orgnet_vars.preds)
            
        else:
            st.info("Please complete the input parameters and click Submit")
    return


if __name__ == "__main__":
    main()