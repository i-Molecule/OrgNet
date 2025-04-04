import argparse
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from data.vox_dataset import VoxDataset, load_voxels, reshape_voxels_array
from models.orgnet import OrgNet
from models.thermonet import ThermoNet
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils.helpers import (
    loss_by_name_dict,
    metric_by_name,
    train_one_epoch,
    validate_one_epoch,
)


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def reset_weights(net):
    for layer in net.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def call_train(
    path_to_data_json,
    device: Literal["cuda", "cpu"] = "cuda",
    num_workers: int = 0,
    loss_name: Literal["MSE"] = "MSE",
    metric_name: Literal[
        "RootMeanSquaredError", "PearsonCorrCoef", "MeanAbsoluteError"
    ] = "PearsonCorrCoef",
):
    with open(path_to_data_json) as file:
        data_dic = json.load(file)

    model_name = data_dic.get("model_name", "OrgNet")

    SEED = data_dic.get("SEED", 126)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(SEED)

    if model_name == "OrgNet":
        net = OrgNet()
    elif model_name == "ThermoNet":
        net = ThermoNet(se3conv=False)
    elif model_name == "ThermoNet_steerable":
        net = ThermoNet(se3conv=True, device=device)

    device = torch.device(device)
    net.to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-07,
        weight_decay=0,
        amsgrad=False,
    )

    scheduler_args = data_dic.get(
        "scheduler_args",
        {
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "cooldown": 5,
            "min_lr": 0,
            "eps": 1e-11,
        },
    )

    batch_size = data_dic.get("batch_size", 8)
    epochs = data_dic.get("epochs", 100)
    patience = data_dic.get("patience", 15)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    dir_pt_save = data_dic.get("dir_pt_save", None)

    train_val_tuples = []

    metric_values = {}
    loss_values = {}
    training_loss_values = {}

    best_val_loss_values = {}
    best_metric_values = {}
    best_training_loss_values = {}

    if ("k_folds" in data_dic.keys()) and (
        set(["X_test", "X_train", "y_test", "y_train"])
        == set(data_dic["k_folds"].keys())
    ):
        _, _, n_channels, grid_size = reshape_voxels_array(
            np.load(data_dic["k_folds"]["X_train"][0])
        )

        for fold in range(len(data_dic["k_folds"]["X_train"])):
            X_train, _, _, _ = reshape_voxels_array(
                np.load(data_dic["k_folds"]["X_train"][fold])
            )
            X_test, _, _, _ = reshape_voxels_array(
                np.load(data_dic["k_folds"]["X_test"][fold])
            )

            train_val_tuples.append(
                (
                    VoxDataset(
                        voxels=X_train,
                        values=np.load(data_dic["k_folds"]["y_train"][fold]),
                        n_channels=n_channels,
                        grid_size=grid_size,
                        device=device,
                    ),
                    VoxDataset(
                        voxels=X_test,
                        values=np.load(data_dic["k_folds"]["y_test"][fold]),
                        n_channels=n_channels,
                        grid_size=grid_size,
                        device=device,
                    ),
                )
            )

            del X_train
            del X_test
    elif (
        ("k_folds" not in data_dic.keys())
        and ("path_to_voxels" in data_dic.keys())
        and ("path_to_values" in data_dic.keys())
    ):
        full_voxels, _, n_channels, grid_size, full_values = load_voxels(
            data_dic["path_to_voxels"], data_dic["path_to_values"]
        )
        if ("path_to_ind_dir" in data_dic.keys()) and (
            "path_to_ind_rev" in data_dic.keys()
        ):
            k = 5
            ind_dir = np.load(data_dic["path_to_ind_dir"])
            ind_rev = np.load(data_dic["path_to_ind_rev"])
            assert len(ind_dir) == len(ind_rev)
            sample_size = ind_dir.shape[0]
            num_validation_samples = sample_size // k

            X_direct = full_voxels[ind_dir].copy()
            y_direct = full_values[ind_dir].copy()

            X_inverse = full_voxels[ind_rev].copy()
            y_inverse = full_values[ind_rev].copy()

            del full_voxels
            del full_values

            for i in range(k):
                X_val_direct = X_direct[
                    i * num_validation_samples : (i + 1) * num_validation_samples
                ]
                y_val_direct = y_direct[
                    i * num_validation_samples : (i + 1) * num_validation_samples
                ]
                X_val_inverse = X_inverse[
                    i * num_validation_samples : (i + 1) * num_validation_samples
                ]
                y_val_inverse = y_inverse[
                    i * num_validation_samples : (i + 1) * num_validation_samples
                ]
                X_val = np.concatenate((X_val_direct, X_val_inverse))
                y_val = np.concatenate((y_val_direct, y_val_inverse))

                X_train_direct = np.concatenate(
                    [
                        X_direct[: i * num_validation_samples],
                        X_direct[(i + 1) * num_validation_samples :],
                    ],
                    axis=0,
                )
                X_train_inverse = np.concatenate(
                    [
                        X_inverse[: i * num_validation_samples],
                        X_inverse[(i + 1) * num_validation_samples :],
                    ],
                    axis=0,
                )
                y_train_direct = np.concatenate(
                    [
                        y_direct[: i * num_validation_samples],
                        y_direct[(i + 1) * num_validation_samples :],
                    ],
                    axis=0,
                )
                y_train_inverse = np.concatenate(
                    [
                        y_inverse[: i * num_validation_samples],
                        y_inverse[(i + 1) * num_validation_samples :],
                    ],
                    axis=0,
                )
                X_train = np.concatenate((X_train_direct, X_train_inverse))
                y_train = np.concatenate((y_train_direct, y_train_inverse))

                # shuffle the training set
                # indices = np.arange(0, X_train.shape[0])
                # np.random.shuffle(indices)
                # X_train = X_train[indices]
                # y_train = y_train[indices]

                train_val_tuples.append(
                    (
                        VoxDataset(
                            voxels=X_train,
                            values=y_train,
                            n_channels=n_channels,
                            grid_size=grid_size,
                            device=device,
                        ),
                        VoxDataset(
                            voxels=X_val,
                            values=y_val,
                            n_channels=n_channels,
                            grid_size=grid_size,
                            device=device,
                        ),
                    )
                )
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            for train_indices, valid_indices in kf.split(full_voxels):
                train_val_tuples.append(
                    (
                        VoxDataset(
                            voxels=full_voxels[train_indices, :, :, :, :],
                            values=full_values[train_indices],
                            n_channels=n_channels,
                            grid_size=grid_size,
                            device=device,
                        ),
                        VoxDataset(
                            voxels=full_voxels[valid_indices, :, :, :, :],
                            values=full_values[valid_indices],
                            n_channels=n_channels,
                            grid_size=grid_size,
                            device=device,
                        ),
                    )
                )
    elif (
        ("k_folds" not in data_dic.keys())
        and ("path_to_voxels_dir" in data_dic.keys())
        and ("path_to_values_dir" in data_dic.keys())
        and ("path_to_voxels_rev" in data_dic.keys())
        and ("path_to_values_rev" in data_dic.keys())
    ):
        k = 10
        X_direct, _, n_channels, grid_size, y_direct = load_voxels(
            data_dic["path_to_voxels_dir"], data_dic["path_to_values_dir"]
        )
        X_inverse, _, n_channels_, grid_size_, y_inverse = load_voxels(
            data_dic["path_to_voxels_rev"], data_dic["path_to_values_rev"]
        )
        sample_size = X_direct.shape[0]
        num_validation_samples = sample_size // k
        assert n_channels == n_channels_
        assert grid_size == grid_size_
        for i in range(k):
            X_val_direct = X_direct[
                i * num_validation_samples : (i + 1) * num_validation_samples
            ]
            y_val_direct = y_direct[
                i * num_validation_samples : (i + 1) * num_validation_samples
            ]
            X_val_inverse = X_inverse[
                i * num_validation_samples : (i + 1) * num_validation_samples
            ]
            y_val_inverse = y_inverse[
                i * num_validation_samples : (i + 1) * num_validation_samples
            ]
            X_val = np.concatenate((X_val_direct, X_val_inverse))
            y_val = np.concatenate((y_val_direct, y_val_inverse))

            X_train_direct = np.concatenate(
                [
                    X_direct[: i * num_validation_samples],
                    X_direct[(i + 1) * num_validation_samples :],
                ],
                axis=0,
            )
            X_train_inverse = np.concatenate(
                [
                    X_inverse[: i * num_validation_samples],
                    X_inverse[(i + 1) * num_validation_samples :],
                ],
                axis=0,
            )
            y_train_direct = np.concatenate(
                [
                    y_direct[: i * num_validation_samples],
                    y_direct[(i + 1) * num_validation_samples :],
                ],
                axis=0,
            )
            y_train_inverse = np.concatenate(
                [
                    y_inverse[: i * num_validation_samples],
                    y_inverse[(i + 1) * num_validation_samples :],
                ],
                axis=0,
            )
            X_train = np.concatenate((X_train_direct, X_train_inverse))
            y_train = np.concatenate((y_train_direct, y_train_inverse))

            indices = np.arange(0, X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            train, val = (
                VoxDataset(
                    voxels=X_train,
                    values=y_train,
                    n_channels=n_channels,
                    grid_size=grid_size,
                    cubic_rotations=False,
                    device=device,
                ),
                VoxDataset(
                    voxels=X_val,
                    values=y_val,
                    n_channels=n_channels,
                    grid_size=grid_size,
                    cubic_rotations=False,
                    device=device,
                ),
            )

            fold = i
            print(f"Fold: {fold + 1}")
            trainloader = torch.utils.data.DataLoader(
                train,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            valloader = torch.utils.data.DataLoader(
                val,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

            metric = metric_by_name(metric_name="PearsonCorrCoef", device=device)
            loss_function = loss_by_name_dict[loss_name]

            reset_weights(net)
            best_val_loss = float("inf")
            counter = 0
            for epoch in tqdm(range(epochs)):
                training_loss_values[fold] = train_one_epoch(
                    net=net,
                    trainloader=trainloader,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    device=device,
                )
                metric_values[fold], loss_values[fold] = validate_one_epoch(
                    net,
                    valloader=valloader,
                    loss_function=loss_function,
                    metric=metric,
                    device=device,
                )

                scheduler.step(loss_values[fold])

                val_loss = loss_values[fold]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    if dir_pt_save is not None:
                        torch.save(
                            net.state_dict(),
                            Path(Path(dir_pt_save), str(fold) + ".pt"),
                        )
                    best_val_loss_values[fold] = loss_values[fold]
                    best_metric_values[fold] = metric_values[fold].cpu().item()
                    best_training_loss_values[fold] = training_loss_values[fold]
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
    else:
        raise ValueError(
            "JSON should either contain `k_folds` field with keys `X_test`, `X_train`, `y_test`, `y_train`, each specifying paths to voxels and values OR contain `path_to_voxels` and `path_to_values`"
        )

    if (
        ("k_folds" in data_dic.keys())
        and (
            set(["X_test", "X_train", "y_test", "y_train"])
            == set(data_dic["k_folds"].keys())
        )
    ) or (
        ("k_folds" not in data_dic.keys())
        and ("path_to_voxels" in data_dic.keys())
        and ("path_to_values" in data_dic.keys())
    ):
        dataset = train_val_tuples

        for fold, (train, val) in enumerate(dataset):
            print(f"Fold: {fold + 1}")
            trainloader = torch.utils.data.DataLoader(
                train,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            valloader = torch.utils.data.DataLoader(
                val,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

            metric = metric_by_name(metric_name="PearsonCorrCoef", device=device)
            loss_function = loss_by_name_dict[loss_name]

            reset_weights(net)
            best_val_loss = float("inf")
            counter = 0
            for epoch in tqdm(range(epochs)):
                training_loss_values[fold] = train_one_epoch(
                    net=net,
                    trainloader=trainloader,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    device=device,
                )
                metric_values[fold], loss_values[fold] = validate_one_epoch(
                    net,
                    valloader=valloader,
                    loss_function=loss_function,
                    metric=metric,
                    device=device,
                )

                scheduler.step(loss_values[fold])

                val_loss = loss_values[fold]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    if dir_pt_save is not None:
                        torch.save(
                            net.state_dict(),
                            Path(Path(dir_pt_save), str(fold) + ".pt"),
                        )
                    best_val_loss_values[fold] = loss_values[fold]
                    best_metric_values[fold] = metric_values[fold].cpu().item()
                    best_training_loss_values[fold] = training_loss_values[fold]
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    mean_train_loss = np.mean(list(best_training_loss_values.values()))
    std_train_loss = np.std(list(best_training_loss_values.values()))

    mean_val_loss = np.mean(list(best_val_loss_values.values()))
    std_val_loss = np.std(list(best_val_loss_values.values()))

    mean_val_metric = np.mean(list(best_metric_values.values()))
    std_val_metric = np.std(list(best_metric_values.values()))

    print(f"\n*****MEAN TRAIN LOSS ({loss_name}): {mean_train_loss}")
    print(f"*****STD TRAIN LOSS ({loss_name}): {std_train_loss}")

    print(f"\n*****MEAN VALIDATION LOSS ({loss_name}): {mean_val_loss}")
    print(f"*****STD VALIDATION LOSS ({loss_name}): {std_val_loss}")

    print(f"\n*****MEAN VALIDATION METRIC ({metric_name}): {mean_val_metric}")
    print(f"*****STD VALIDATION METRIC ({metric_name}): {std_val_metric}")


def _parse_args(args: Optional[str] = None):
    parser = argparse.ArgumentParser(description="OrgNet training")
    parser.add_argument(
        "-data_json",
        "--path_to_data_json",
        required=True,
        help="Path to .json file specifying data and settings",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run the inference on (`cpu` or `cuda`)",
        default="cuda",
    )
    return parser.parse_args(args=args)


def main(args: Optional[str] = None):
    args = _parse_args(args)

    call_train(args.path_to_data_json, device=args.device)


if __name__ == "__main__":
    main()
