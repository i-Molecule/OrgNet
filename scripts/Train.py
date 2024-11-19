"""
Small inference script that can be called via commandline using argparse.
Use the pre-trained model to predict the active and inactive probabilities for a single pdb file.

Required arguments:
-evdirect : Path to calculated direct features
-evreverse : Path to calculated reverse features
-evds : Path to dataset csv file
-trdirect : Path to calculated train direct features
-trreverse : Path to calculated train reverse features
-trds : Path to train dataset csv file
    
-mod : Path to save models
-log : Path to save logs
    
Example usage:
python Train.py -evdirect /home/nata/work/Projects/Protein_stability_prediction/Ssym/features/Ssym_ori/Ssym_{feature_type}_direct/ -evreverse  /home/nata/work/Projects/Protein_stability_prediction/Ssym/features/Ssym_ori/Ssym_{feature_type}_reverse/ -evds  /home/nata/work/Programs/ThermoNet/data/datasets/Ssym.csv -trdirect /home/nata/work/Projects/Protein_stability_prediction/S2648_VB/features/S2648_VB_oriented/S2648_V_defdif_direct/-trreverse /home/nata/work/Projects/Protein_stability_prediction/S2648_VB/features/S2648_VB_oriented/S2648_V_defdif_reverse/ -trds /home/nata/work/Projects/Protein_stability_prediction/S2648_VB/S2648_VB_dataset.csv -mod /home/nata/work/Projects/Protein_stability_prediction/Thermonet_var/14656_Unique_Mutations_Voxel_Features_PDBs/Models_fin2/NA_test_nft_264Vb/ -log /home/nata/work/Projects/Protein_stability_prediction/Thermonet_var/14656_Unique_Mutations_Voxel_Features_PDBs/Models_fin2/NA_test_nft_264Vb_pics/
"""
import argparse
import pathlib
import os,sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
import scipy.stats as sc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from tensorflow.keras import models, layers, optimizers
import itertools
import time

tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
tf.config.experimental.enable_op_determinism()

def prepare_dataset_dir(training_dataset_path,
                        path_to_features):
    print("1. Loading csv datasets")
    df = pd.read_csv(training_dataset_path)
    print(f'Total unique mutations: {len(df)}')
    
    # Load direct features
    df['features'] = df.apply(lambda row: load_feature_path(row, path_to_features), axis=1)
    df = df[df['features'].apply(check_path_exists)]
    print(f'Total mutations with features: {len(df)}')
    
    # For TensorFlow, we work directly with the paths, and load the features on-the-fly.
    paths_dir = df['features'].values
    labels_dir = df['ddg'].values  # Assuming 'ddg' is your label. Adjust if necessary.

    return paths_dir, labels_dir

def prepare_dataset_rev(training_dataset_path,
                        path_to_features):
    
    print('Augmenting reverse mutations')
    df_rev = pd.read_csv(training_dataset_path)
    df_rev['ddg'] = -df_rev['ddg']

    
    df_rev['features'] = df_rev.apply(lambda row: load_feature_path(row, path_to_features), axis=1)
    df_rev = df_rev[df_rev['features'].apply(os.path.exists)]
    print(f'Total reverse mutations with features: {len(df_rev)}')

    paths_rev = df_rev['features'].values
    labels_rev = df_rev['ddg'].values

    return paths_rev, labels_rev

def load_data_training_set_tf_full(training_dataset_path,
                                   training_features_dir_dir,
                                   training_features_dir_rev):
    
    paths_dir, labels_dir = prepare_dataset_dir(training_dataset_path,
                                                training_features_dir_dir)
        
    paths_rev, labels_rev = prepare_dataset_rev(training_dataset_path,
                                                training_features_dir_rev)
    
    paths = np.append(paths_dir, paths_rev)
    labels = np.append(labels_dir, labels_rev)
    
    
    #dataset = create_dataset(paths, labels)

    return paths, labels

def shuffle_df(df):
    
    data_copy = df_test_fold.values.copy()
    np.random.shuffle(data_copy)
    shuffled_df = pd.DataFrame(data_copy, columns=df_test_fold.columns)
    return shuffled_df

def load_feature_path(row, path_to_features):
    return f'{path_to_features}/{row.pdb_id}/{row.pdb_id}_{row.wild_type}{row.position}{row.mutant}.npy'

def check_path_exists(path):
    return os.path.exists(path)


def load_and_preprocess_from_path_label(path, label):
    # Decode path if it's a tensor.
    path = path.numpy().decode()
    feature = np.load(path)
    feature = np.transpose(feature, (1, 2, 3, 0))  # Adjust axis if necessary.
    return feature.astype(np.float32), np.array(label, dtype=np.float32)

def create_dataset(paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(paths.astype(str))
    label_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))
    
    dataset = tf.data.Dataset.zip((path_ds, label_ds))
    dataset = dataset.map(
        lambda path, label: tf.py_function(
            load_and_preprocess_from_path_label, [path, label], [tf.float32, tf.float32]
        ), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(8).prefetch(tf.data.AUTOTUNE)  # Un-comment batching and prefetching for performance.

    return dataset
    
def load_data_training_set():
    
    print("1. Loading csv datasets")
    df = pd.read_csv(training_dataset_path)
    print(f'Total unique mutations: {len(df)}')

    #load direct features
    df['features'] = df.apply(lambda r: f'{training_features_dir_dir}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
    df = df[df.features.apply(lambda v: os.path.exists(v))]
    print(f'Total mutations with features: {len(df)}')
    df.features = [np.load(f) for f in tqdm(df.features, desc="2. Loading features")]
    print(f'Total mutations after filtering: {len(df)}')
    df_train = df
    
    if AUGMENT_REVERSE_MUTATIONS:
        
        print('Augmenting reverse mutations')
        df_rev = pd.read_csv(training_dataset_path)
        df_rev.ddg = -df_rev.ddg

        
        df_rev['features'] = df_rev.apply(lambda r: f'{training_features_dir_rev}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
        df_rev = df_rev[df_rev.features.apply(lambda v: os.path.exists(v))]
        print(f'Total mutations with features: {len(df)}')
        
        
        df_rev.features = [np.load(f) for f in tqdm(df_rev.features, desc="3. Loading features")]
        print(f'Total mutations after filtering: {len(df_rev)}')
        df_train = pd.concat([df_train, df_rev], axis=0).sample(frac=1.).reset_index(drop=True)

    df_train.features = df_train.features.apply(lambda k: np.transpose(k, (1, 2, 3, 0)))
    
    return df_train

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

class EvaluateAndStoreMetrics(Callback):
    def __init__(self, X_val, y_val, key_prefix):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.key_prefix = key_prefix

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate the model on the validation set
        val_loss, val_mae, val_mse, val_rmse, val_pearson_r = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )

        # Store the evaluation metrics in history.history
        key_loss = self.key_prefix + 'loss'
        key_mae = self.key_prefix + 'mae'
        key_mse = self.key_prefix + 'mse'
        key_rmse = self.key_prefix + 'rmse'
        key_pearson_r = self.key_prefix + 'pearson_r'

        logs[key_loss] = val_loss
        logs[key_mae] = val_mae
        logs[key_mse] = val_mse
        logs[key_rmse] = val_rmse
        logs[key_pearson_r] = val_pearson_r

        # Print or log the metrics if needed
        print(f"\nValidation Metrics after Epoch {epoch + 1}:")
        print(f" - {key_loss}: {val_loss:.4f}")
        print(f" - {key_mae}: {val_mae:.4f}")
        print(f" - {key_mse}: {val_mse:.4f}")
        print(f" - {key_rmse}: {val_rmse:.4f}")
        print(f" - {key_pearson_r}: {val_pearson_r:.4f}")

def modify_df_train(training_dataset_path, AUGMENT_REVERSE_MUTATIONS, training_features_dir_dir, training_features_dir_rev):
    
    print("1. Loading csv datasets")
    df = pd.read_csv(training_dataset_path)
    print(f'Total unique mutations: {len(df)}')

    #load direct features
    df['features'] = df.apply(lambda r: f'{training_features_dir_dir}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
    df = df[df.features.apply(lambda v: os.path.exists(v))]
    print(f'Total mutations with features: {len(df)}')
    #df.features = [np.load(f) for f in tqdm(df.features, desc="2. Loading features")]
    print(f'Total mutations after filtering: {len(df)}')
    df_train = df
    
    
    if AUGMENT_REVERSE_MUTATIONS:
        
        print('Augmenting reverse mutations')
        df_rev = pd.read_csv(training_dataset_path)
        df_rev.ddg = -df_rev.ddg

        
        df_rev['features'] = df_rev.apply(lambda r: f'{training_features_dir_rev}/{r.pdb_id}/{r.pdb_id}_{r.wild_type}{r.position}{r.mutant}.npy', axis=1)
        df_rev = df_rev[df_rev.features.apply(lambda v: os.path.exists(v))]
        print(f'Total mutations with features: {len(df_rev)}')
        
        
        #df_rev.features = [np.load(f) for f in tqdm(df_rev.features, desc="3. Loading features")]
        print(f'Total mutations after filtering: {len(df_rev)}')
        df_train = pd.concat([df_train, df_rev], axis=0).sample(frac=1.).reset_index(drop=True)
    
    return df_train


class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name='pearson_r', **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.pearson_r = self.add_weight(name='pr', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        if tf.shape(y_true)[0] == 1:
            y_true = tf.concat([y_true, y_true], axis=0)
            y_pred = tf.concat([y_pred, y_pred], axis=0)

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        pr, _ = tf.py_function(sc.pearsonr, [y_true, y_pred], [tf.float64, tf.float64])
        pr = tf.cast(pr, tf.float32)  # Convert to float32
        self.pearson_r.assign(pr)

    def result(self):
        return self.pearson_r

    def reset_states(self):
        self.pearson_r.assign(0.0)


class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.mse = tf.keras.metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mse.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return tf.sqrt(self.mse.result())

    def reset_states(self):
        self.mse.reset_states()

# Define a function to compile and train the model
class EvaluateAndStoreMetrics(Callback):
    def __init__(self, X_val, y_val, key_prefix):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.key_prefix = key_prefix

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate the model on the validation set
        val_loss, val_mae, val_mse, val_rmse, val_pearson_r = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )

        # Store the evaluation metrics in history.history
        key_loss = self.key_prefix + 'loss'
        key_mae = self.key_prefix + 'mae'
        key_mse = self.key_prefix + 'mse'
        key_rmse = self.key_prefix + 'rmse'
        key_pearson_r = self.key_prefix + 'pearson_r'

        logs[key_loss] = val_loss
        logs[key_mae] = val_mae
        logs[key_mse] = val_mse
        logs[key_rmse] = val_rmse
        logs[key_pearson_r] = val_pearson_r

        # Print or log the metrics if needed
        print(f"\nValidation Metrics after Epoch {epoch + 1}:")
        print(f" - {key_loss}: {val_loss:.4f}")
        print(f" - {key_mae}: {val_mae:.4f}")
        print(f" - {key_mse}: {val_mse:.4f}")
        print(f" - {key_rmse}: {val_rmse:.4f}")
        print(f" - {key_pearson_r}: {val_pearson_r:.4f}")

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


def plot_metrics(history,picspathsave, m_nm, feature_type):
    epochs = range(1, len(history.history['loss']) + 1)

    # Subplot 1: Training and Validation Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title(f'{m_nm} Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Subplot 2: MSE, RMSE and Pearson_r
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['mse'], label='Training MSE')
    plt.plot(epochs, history.history['rmse'], label='Training RMSE')
    plt.plot(epochs, history.history['pearson_r'], label='Training Pearson_r')
    plt.plot(epochs, history.history['val_mse'], label='Validation MSE')
    plt.plot(epochs, history.history['val_rmse'], label='Validation RMSE')
    plt.plot(epochs, history.history['val_pearson_r'], label='Validation Pearson_r')
    plt.title(f'{m_nm} MSE, RMSE and Pearson_r on Training and Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{picspathsave}/{m_nm}_{feature_type}_losses_ori.png", dpi = 300)
    plt.show()

    # Plotting evaluation metrics for the second validation set
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, history.history['eval_direct_loss'], label='Validation Loss Ssym_dir')
    plt.plot(epochs, history.history['eval_direct_mae'], label='Validation MAE Ssym_dir')
    plt.plot(epochs, history.history['eval_direct_mse'], label='Validation MSE Ssym_dir')
    plt.plot(epochs, history.history['eval_direct_rmse'], label='Validation RMSE Ssym_dir')
    plt.plot(epochs, history.history['eval_direct_pearson_r'], label='Validation Pearson_r Ssym_dir')

    plt.title(f'{m_nm} Evaluation Metrics for Ssym_dir')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.savefig(f"{picspathsave}/{m_nm}_{feature_type}_dif_ssym_dir_ori.png", dpi = 300)
    plt.show()

    # Plotting evaluation metrics for the third validation set
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, history.history['eval_rev_loss'], label='Validation Loss Ssym_rev')
    plt.plot(epochs, history.history['eval_rev_mae'], label='Validation MAE Ssym_rev')
    plt.plot(epochs, history.history['eval_rev_mse'], label='Validation MSE Ssym_rev')
    plt.plot(epochs, history.history['eval_rev_rmse'], label='Validation RMSE Ssym_rev')
    plt.plot(epochs, history.history['eval_rev_pearson_r'], label='Validation Pearson_r Ssym_rev')

    plt.title(f'{m_nm} Evaluation Metrics for Ssym_rev')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.savefig(f"{picspathsave}/{m_nm}_{feature_type}_ssym_rev_ori.png", dpi = 300)
    plt.show()

#old architecture
def callbacks_and_train_grsrch(model, train_dataset, val_dataset, epochs, batch_size, model_path):

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10
    )
    
    
    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        verbose=1,
        mode='min',
        patience=5,
        cooldown=0,
        min_lr=1e-8
    )

    evaluate_callback_Ssym_dir = EvaluateAndStoreMetrics(X_direct_ssym_dir, y_direct_ssym_dir, key_prefix='eval_direct_')
    evaluate_callback_Ssym_rev = EvaluateAndStoreMetrics(X_direct_ssym_rev, y_direct_ssym_rev, key_prefix='eval_rev_')
    
    callbacks_list = [
        checkpoint,
        early_stopping,
        reduce_lr,
        evaluate_callback_Ssym_dir,
        evaluate_callback_Ssym_rev
    ]
    
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks_list,
                        verbose=1)
    model.save(model_path)
    return history



def build_and_compile_model_new_arch(learning_rate):
    
    
    ###
    #Use default architecture than modify it
    ###
    
    model = models.Sequential()

    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(16, 16, 16, 14)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((2, 2, 2)))

    # Block 2
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((2, 2, 2)))

    # Block 3
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((2, 2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))  # Dropout layer after the first dense layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))  # Dropout layer after the second dense layer
    model.add(layers.Dense(units=1))

    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae', 'mse', rmse, pearson_r])

    return model








if __name__ == "__main__":


    # parse arguments
    PARSER = argparse.ArgumentParser(description="Training script for OrgNet.")
    
    PARSER.add_argument("-evdirect", "--input_direct", help="Path to calculated direct features.", required=True, type=pathlib.Path)
    PARSER.add_argument("-evreverse", "--input_reverse", help="Path to calculated reverse features.", required=True, type=pathlib.Path)
    PARSER.add_argument("-evds", "--input_dataset", help="Path to dataset csv file.", required=True, type=pathlib.Path)
    
    PARSER.add_argument("-trdirect", "--train_direct", help="Path to calculated train direct features.", required=True, type=pathlib.Path)
    PARSER.add_argument("-trreverse", "--train_reverse", help="Path to calculated train reverse features.", required=True, type=pathlib.Path)
    PARSER.add_argument("-trds", "--train_dataset", help="Path to train dataset csv file.", required=True, type=pathlib.Path)
    
    PARSER.add_argument("-mod", "--model_path", help="Path to save models.", required=True, type=pathlib.Path)
    PARSER.add_argument("-log", "--path_to_save_logging", help="Path to save logs.", required=True, type=pathlib.Path)

    
    ARGS = PARSER.parse_args()
    
    #evaluation data
    df_train_ssym_dir = load_data_ssym_dir(ARGS.input_dataset, ARGS.input_direct)
    df_train_ssym_rev = load_data_ssym_rev(ARGS.input_dataset, ARGS.input_reverse)

    #prepare validation sets on Ssym direct dataset
    X_direct_ssym_dir = np.array(df_train_ssym_dir.features.to_list())
    y_direct_ssym_dir = df_train_ssym_dir.ddg.to_numpy()


    #prepare validation sets on Ssym reverse dataset
    X_direct_ssym_rev = np.array(df_train_ssym_rev.features.to_list())
    y_direct_ssym_rev = df_train_ssym_rev.ddg.to_numpy()


    #10f CV
    training_features_dir_dir = ARGS.train_direct
    training_features_dir_rev = ARGS.train_reverse
    training_dataset_path = ARGS.train_dataset
    
    mp = ARGS.model_path
    picspathsave = ARGS.path_to_save_logging


    epochs = 100
    batch_size = 8


    
    learning_rate = 0.001

    df_train = modify_df_train(training_dataset_path, True, training_features_dir_dir, training_features_dir_rev)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_num = 1
    histories = []

    log_loss_fold = []
    ev=[]
    results = []
    fold_num = 0

    for train_index, val_index in kf.split(df_train):
        train_fold = df_train.iloc[train_index]
        val_fold = df_train.iloc[val_index]

        train_paths = train_fold['features'].values
        train_labels = train_fold['ddg'].values
        val_paths = val_fold['features'].values
        val_labels = val_fold['ddg'].values

        train_dataset = create_dataset(train_paths, train_labels)
        val_dataset = create_dataset(val_paths, val_labels)

        model = build_and_compile_model_new_arch(learning_rate=learning_rate)
        model_path = f"{mp}kerv1_member_{fold_num}.h5"
        m_nm = f"kerv1_member_{fold_num}.h5"
        history = callbacks_and_train_grsrch(model, train_dataset, val_dataset, epochs, batch_size, model_path)
    
        #logging of mean loss and other metrics across the folds
        result = {
            'fold': fold_num,
            'val_loss': min(history.history['val_loss']),
            'val_mae': min(history.history['val_mae']),
            'val_mse': min(history.history['val_mse']),
            'val_rmse': min(history.history['val_rmse']),
            'val_pearson_r': max(history.history['val_pearson_r'])
        }
    
        results.append(result)
        log_loss_fold.append(min(history.history['val_loss']))
        ev.append({'min_loss_CV': min(log_loss_fold), 'mean_loss_CV': np.mean(log_loss_fold)})
        df_results_inter = pd.DataFrame(results)
        ev_results_inter =pd.DataFrame(ev)
        ev_results_inter.to_csv(f'{ARGS.path_to_save_logging}/Res_inter_ev.csv', index=False)
        df_results_inter.to_csv(f'{ARGS.path_to_save_logging}/Results_inter.csv', index=False)
    
    
        #logging of metrics across the epochs
        plot_metrics(history,picspathsave, m_nm, feature_type)
        fold_num += 1

    





