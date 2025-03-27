"""
Inference script that can be called via commandline using argparse.
Use the pre-trained model to predict the ddG for the prepared datasets.

Required arguments:
-evdirect : Path to calculated direct features
-evreverse : Path to calculated reverse features
-evds : Path to dataset csv file
-o : Path to save the evaluation dataframe
-mod : Path to models.



Optional arguments:
-flag : Sting to flag the models

Example usage:
python Inference.py -evdirect /home/nata/work/Projects/Protein_stability_prediction/Ssym/features/Ssym_ori/Ssym_{feature_type}_direct/ -evreverse  /home/nata/work/Projects/Protein_stability_prediction/Ssym/features/Ssym_ori/Ssym_{feature_type}_reverse/ -evds  /home/nata/work/Programs/ThermoNet/data/datasets/Ssym.csv -o /home/nata/work/Projects/Protein_stability_prediction/Thermonet_var/14656_Unique_Mutations_Voxel_Features_PDBs/Tables_for_article/ -mod /home/nata/work/Projects/Protein_stability_prediction/Thermonet_var/14656_Unique_Mutations_Voxel_Features_PDBs/Models_fin2/NA_5f_q1744/ -flag new_arch_nft
"""



import argparse
import pathlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as sc
from keras.models import load_model


params = ["model", "mae_dir", "mse_dir","rmse_dir", "pearson_r_dir","mae_rev","mse_rev", "rmse_rev","pearson_r_rev", "mae_tot", "mse_tot","rmse_tot","pearson_r_tot"]


def load_data_ssym_dir(evaluation_dataset_path, evaluation_features_dir_dir):
    
    print("Loading Ssym direct mutations")
    df = pd.read_csv(evaluation_dataset_path)
    print(f'Total unique mutations: {len(df)}')

    df['features'] = df.apply(lambda r: f'{evaluation_features_dir_dir}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
    df = df[df.features.apply(lambda v: os.path.exists(v))]
    print(f'Total mutations with features: {len(df)}')
    df.features = [np.load(f) for f in tqdm(df.features, desc="2. Loading features")]
    print(f'Total mutations after filtering: {len(df)}')

    df_train = df
    df_train.features = df_train.features.apply(lambda k: np.transpose(k, (1, 2, 3, 0)))
    
    return df_train

def load_data_ssym_rev(evaluation_dataset_path, evaluation_features_dir_rev):
    
    print('Loading Ssym reverse mutations')
    df_rev = pd.read_csv(evaluation_dataset_path)
    df_rev.ddg = -df_rev.ddg

        
    df_rev['features'] = df_rev.apply(lambda r: f'{evaluation_features_dir_rev}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
    df_rev = df_rev[df_rev.features.apply(lambda v: os.path.exists(v))]
    print(f'Total mutations with features: {len(df_rev)}')
        
        
    df_rev.features = [np.load(f) for f in tqdm(df_rev.features, desc="3. Loading features")]
    print(f'Total mutations after filtering: {len(df_rev)}')
    
    df_rev.features = df_rev.features.apply(lambda k: np.transpose(k, (1, 2, 3, 0)))
    
    return df_rev
    
def load_data_s669_dir(evaluation_dataset_path, evaluation_features_dir_dir):
    
    print("Loading Ssym direct mutations")
    df = pd.read_csv(evaluation_dataset_path)
    print(f'Total unique mutations: {len(df)}')

    df['features'] = df.apply(lambda r: f'{evaluation_features_dir_dir}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
    df = df[df.features.apply(lambda v: os.path.exists(v))]
    print(f'Total mutations with features: {len(df)}')
    df.features = [np.load(f) for f in tqdm(df.features, desc="2. Loading features")]
    print(f'Total mutations after filtering: {len(df)}')

    df_train = df
    df_train.features = df_train.features.apply(lambda k: np.transpose(k, (1, 2, 3, 0)))
    df_train.ddg = -df_train.ddg
    
    return df_train

def load_data_s669_rev(evaluation_dataset_path, evaluation_features_dir_rev):
    
    print('Loading Ssym reverse mutations')
    df_rev = pd.read_csv(evaluation_dataset_path)
    #df_rev.ddg = -df_rev.ddg

        
    df_rev['features'] = df_rev.apply(lambda r: f'{evaluation_features_dir_rev}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
    df_rev = df_rev[df_rev.features.apply(lambda v: os.path.exists(v))]
    print(f'Total mutations with features: {len(df_rev)}')
        
        
    df_rev.features = [np.load(f) for f in tqdm(df_rev.features, desc="3. Loading features")]
    print(f'Total mutations after filtering: {len(df_rev)}')
    
    df_rev.features = df_rev.features.apply(lambda k: np.transpose(k, (1, 2, 3, 0)))
    
    return df_rev
    
def rmse(y_val_direct, y_pred):

    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(y_val_direct) - tf.squeeze(y_pred))))
    
    return rmse

def pearson_r(y_val_direct, y_pred):

    if tf.shape(y_val_direct)[0] == 1:
        y_val_direct = tf.concat([y_val_direct, y_val_direct], axis=0)
        y_pred = tf.concat([y_pred, y_pred], axis=0)

        pr, _ = tf.py_function(sc.pearsonr, [y_val_direct, y_pred], [tf.float64, tf.float64])
        #tf.print("Pearson correlation coefficient:", pr)
    else:
        y_val_direct = tf.squeeze(y_val_direct)
        y_pred = tf.squeeze(y_pred)
    
        pr, _ = tf.py_function(sc.pearsonr, [y_val_direct, y_pred], [tf.float64, tf.float64])
        #tf.print("Pearson correlation coefficient:", pr)

    return pr

# Function to prepare datasets
def prepare_datasets(df_train_ssym_dir, df_train_ssym_rev):
    X_direct_ssym_dir = np.array(df_train_ssym_dir.features.to_list())
    y_direct_ssym_dir = df_train_ssym_dir.ddg.to_numpy()

    X_direct_ssym_rev = np.array(df_train_ssym_rev.features.to_list())
    y_direct_ssym_rev = df_train_ssym_rev.ddg.to_numpy()
    
    g = pd.concat([df_train_ssym_dir, df_train_ssym_rev])
    X_total_ssym = np.array(g.features.to_list())
    y_total_ssym = g.ddg.to_numpy()

    return X_direct_ssym_dir,y_direct_ssym_dir, X_direct_ssym_rev, y_direct_ssym_rev, X_total_ssym, y_total_ssym

# Function to evaluate a single model
def evaluate_model(model_path, X_dir, y_dir, X_rev, y_rev , X_tot, y_tot, model_name):
    
    model = load_model(model_path, custom_objects={"rmse": rmse, "pearson_r": pearson_r})
    
    y_pred_dir = model.predict(X_dir).reshape(-1)
    y_pred_rev = model.predict(X_rev).reshape(-1)
    y_pred_tot = model.predict(X_tot).reshape(-1)

    mae_dir = mean_absolute_error(y_dir, y_pred_dir)
    mse_dir = mean_squared_error(y_dir, y_pred_dir)
    rmse_dir = mean_squared_error(y_dir, y_pred_dir, squared=False)
    pr_dir = sc.pearsonr(y_dir, y_pred_dir)[0]

    mae_rev = mean_absolute_error(y_rev, y_pred_rev)
    mse_rev = mean_squared_error(y_rev, y_pred_rev)
    rmse_rev = mean_squared_error(y_rev, y_pred_rev, squared=False)
    pr_rev = sc.pearsonr(y_rev, y_pred_rev)[0]
    
    mae_tot = mean_absolute_error(y_tot, y_pred_tot)
    mse_tot = mean_squared_error(y_tot, y_pred_tot)
    rmse_tot = mean_squared_error(y_tot, y_pred_tot, squared=False)
    pr_tot = sc.pearsonr(y_tot, y_pred_tot)[0]

    return {
        "model_name": model_name,
        "mae_dir": mae_dir, "mse_dir": mse_dir, "rmse_dir": rmse_dir, "pearson_r_dir": pr_dir,
        "mae_rev": mae_rev, "mse_rev": mse_rev, "rmse_rev": rmse_rev, "pearson_r_rev": pr_rev,
        "mae_tot": mae_tot, "mse_tot": mse_tot, "rmse_tot": rmse_tot, "pearson_r_tot": pr_tot
    }

# Function to evaluate all models
def evaluate_models(df_train_ssym_dir, df_train_ssym_rev, model_dir, model_type, v):
    
    X_dir, y_dir, X_rev, y_rev, X_tot, y_tot = prepare_datasets(df_train_ssym_dir, df_train_ssym_rev)
    
    eval_results = []
    
    if model_type == "sing":
        for model_name in sorted(os.listdir(model_dir), key=lambda x: int(x.split(".")[0].split("_")[-1])):
        
            model_path = os.path.join(model_dir, model_name)
            results = evaluate_model(model_path, X_dir, y_dir, X_rev, y_rev , X_tot, y_tot, model_name)
            eval_results.append(results)
    
    if model_type == "ens":
        results = evaluate_ensemble(model_dir, X_dir, y_dir, X_rev, y_rev , X_tot, y_tot)
        eval_results.append(results)

    eval_df = pd.DataFrame(eval_results)
    #eval_df.to_csv(f"{evalpathsave}/{model_dir.split('/')[-2]}_Ssym_eval_results_{v}.csv", index=False)
    
    #print(f"Mean Pearson r (direct): {eval_df['pearson_r_dir'].mean()}")
    #print(f"Mean Pearson r (reverse): {eval_df['pearson_r_rev'].mean()}")
    
    return eval_df

def evaluate_ensemble(models_dir, X_dir, y_dir, X_rev, y_rev, X_tot, y_tot):
    
    model_files = sorted(os.listdir(models_dir), key=lambda x: int(x.split(".")[0].split("_")[-1]))
    ensemble_preds_dir = []
    ensemble_preds_rev = []
    ensemble_preds_tot = []

    for model_name in model_files:
        model_path = os.path.join(models_dir, model_name)
        model = load_model(model_path, custom_objects={"rmse": rmse, "pearson_r": pearson_r})

        y_pred_dir = model.predict(X_dir).reshape(-1)
        y_pred_rev = model.predict(X_rev).reshape(-1)
        y_pred_tot = model.predict(X_tot).reshape(-1)

        ensemble_preds_dir.append(y_pred_dir)
        ensemble_preds_rev.append(y_pred_rev)
        ensemble_preds_tot.append(y_pred_tot)

    # Average predictions across all models
    avg_pred_dir = np.mean(ensemble_preds_dir, axis=0)
    avg_pred_rev = np.mean(ensemble_preds_rev, axis=0)
    avg_pred_tot = np.mean(ensemble_preds_tot, axis=0)

    # Evaluate ensemble predictions
    mae_dir = mean_absolute_error(y_dir, avg_pred_dir)
    mse_dir = mean_squared_error(y_dir, avg_pred_dir)
    rmse_dir = mean_squared_error(y_dir, avg_pred_dir, squared=False)
    pr_dir = sc.pearsonr(y_dir, avg_pred_dir)[0]

    mae_rev = mean_absolute_error(y_rev, avg_pred_rev)
    mse_rev = mean_squared_error(y_rev, avg_pred_rev)
    rmse_rev = mean_squared_error(y_rev, avg_pred_rev, squared=False)
    pr_rev = sc.pearsonr(y_rev, avg_pred_rev)[0]
    
    
    mae_tot = mean_absolute_error(y_tot, avg_pred_tot)
    mse_tot = mean_squared_error(y_tot, avg_pred_tot)
    rmse_tot = mean_squared_error(y_tot, avg_pred_tot, squared=False)
    pr_tot = sc.pearsonr(y_tot, avg_pred_tot)[0]

    return {
        "mae_dir": mae_dir, "mse_dir": mse_dir, "rmse_dir": rmse_dir, "pearson_r_dir": pr_dir,
        "mae_rev": mae_rev, "mse_rev": mse_rev, "rmse_rev": rmse_rev, "pearson_r_rev": pr_rev,
        "mae_tot": mae_tot, "mse_tot": mse_tot, "rmse_tot": rmse_tot, "pearson_r_tot": pr_tot
    }

def add_model_col(df, key):
    df['model'] = [key+"_"+f.split(".")[0].split('_')[-1] for f in df['model_name']]
    return df
def add_model_col_ens(df, key):
    df['model'] = key
    return df

def table_report(model_dir, w, df_train_ssym_dir, df_train_ssym_rev):
    
    model_type = "sing"
    key = f"{w}_{model_type}"
    df1 = evaluate_models(df_train_ssym_dir, df_train_ssym_rev, model_dir, model_type, key)
    df1 = add_model_col(df1, key)
    
    model_type = "ens"
    key = f"{w}_{model_type}"
    df2 = evaluate_models(df_train_ssym_dir, df_train_ssym_rev, model_dir, model_type, key)
    df2 = add_model_col_ens(df2, key)
    
    df = pd.concat([df1, df2])
    return df

if __name__ == "__main__":

    #example arguments
    #feature_type = "defdif"
    #evaluation_features_dir_dir = f"/home/nata/work/Projects/Protein_stability_prediction/Ssym/features/Ssym_ori/Ssym_{feature_type}_direct/"
    #evaluation_features_dir_rev = f"/home/nata/work/Projects/Protein_stability_prediction/Ssym/features/Ssym_ori/Ssym_{feature_type}_reverse/"
    #evaluation_dataset_path = "/home/nata/work/Programs/ThermoNet/data/datasets/Ssym.csv"
    
    #evalpathsave = "/home/nata/work/Projects/Protein_stability_prediction/Thermonet_var/14656_Unique_Mutations_Voxel_Features_PDBs/Tables_for_article/"
    #model_dir = f"/home/nata/work/Projects/Protein_stability_prediction/Thermonet_var/14656_Unique_Mutations_Voxel_Features_PDBs/Models_fin2/NA_5f_q1744/"
    #w = f"new_arch_nft"
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-evdirect", "--input_direct", help="Path to calculated direct features.", required=True, type=pathlib.Path)
    PARSER.add_argument("-evreverse", "--input_reverse", help="Path to calculated reverse features.", required=True, type=pathlib.Path)
    PARSER.add_argument("-evds", "--input_dataset", help="Path to dataset csv file.", required=True, type=pathlib.Path)
    PARSER.add_argument("-o", "--output", help="Path to save the evaluation dataframe.", required=True, type=pathlib.Path)
    PARSER.add_argument("-mod", "--model_dir", help="Path to models.", required=True, type=pathlib.Path)
    PARSER.add_argument("-flag", "--w", help="Sting to flag the models.", required=False, type=str)
    
    
    ARGS = PARSER.parse_args()    

    df_train_ssym_dir = load_data_ssym_dir(ARGS.input_dataset, ARGS.input_direct)
    df_train_ssym_rev = load_data_ssym_rev(ARGS.input_dataset, ARGS.input_reverse)
    
    if ARGS.flag == "":
        ARGS.flag = "flag"
    
    rf2 = table_report(ARGS.model_dir, ARGS.flag, df_train_ssym_dir, df_train_ssym_rev)
    rf2 = rf2[params].to_csv(f"{ARGS.output}/eval_results{ARGS.flag}.csv")


