import argparse
import os
import os.path as osp
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from data.vox_dataset import VoxDataset, load_voxels
from models.orgnet import OrgNet
from models.thermonet import ThermoNet
from utils.helpers import get_predictions, get_pt_files, metric_by_name, seeds


def call_predict(
    path_to_X: Union[str, os.PathLike],
    path_to_y: Union[str, os.PathLike],
    save_to: Optional[Union[str, os.PathLike]] = None,
    device: Literal["cuda", "cpu"] = "cpu",
    training_data: Literal["Q3214", "S2648_V"] = "S2648_V",
    paths_to_kth_model: Optional[list] = None,
    model_name: Literal["OrgNet", "ThermoNet", "ThermoNet_steerable"] = "OrgNet",
    random_rotations: Optional[bool] = None,
    fully_rotated: bool = False,
) -> list:
    """
    Predicts using the OrgNet model with optional rotation augmentation.

    Args:
        path_to_X (Union[str, os.PathLike]): Path to the input voxel data.
        path_to_y (Union[str, os.PathLike]): Path to the target values.
        save_to (Optional[Union[str, os.PathLike]]): Path to save the predictions.
        device (Literal["cuda", "cpu"]): Device to run the model on.
        training_data (Literal["Q3214", "S2648_V"]): Training data identifier.
        paths_to_kth_model (Optional[list]): List of paths to model weights.
        model_name (Literal["OrgNet", "ThermoNet", "ThermoNet_steerable"]): Model identifier.
        random_rotations (Optional[bool]): Whether to apply random rotations to voxels during inference.
        fully_rotated (bool): Whether to use all 24 rotations for prediction.

    Returns:
        list: List containing metric values (RootMeanSquaredError, PearsonCorrCoef, MeanAbsoluteError).
    """
    SEED = seeds[training_data]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    full_voxels, n_samples, n_channels, grid_size, full_values = load_voxels(
        path_to_X, path_to_y
    )
    device = torch.device(device)

    if paths_to_kth_model is None:
        paths_to_kth_model = [
            osp.join("models", "weights", "orgnet", training_data, f"{k}.pt")
            for k in range(5)
        ]

    if fully_rotated:
        all_predictions = np.zeros((n_samples, 24, 5))
        fold_mean_predictions = np.zeros((n_samples, 5))
        fold_std_rotations = np.zeros((n_samples, 5))

        for rotation_index in range(24):
            test_dataset = VoxDataset(
                voxels=full_voxels,
                values=np.array(range(n_samples)),
                n_channels=n_channels,
                grid_size=grid_size,
                rotation_index=rotation_index,
                device=device,
                v_dtype=torch.int64
            )

            testloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=128, shuffle=False, num_workers=0
            )

            for k, path_to_kth_model in enumerate(paths_to_kth_model):
                if paths_to_kth_model is None:
                    net = OrgNet()
                else:
                    if model_name == "OrgNet":
                        net = OrgNet()
                    elif model_name == "ThermoNet":
                        net = ThermoNet(se3conv=False)
                    elif model_name == "ThermoNet_steerable":
                        net = ThermoNet(se3conv=True, device=device)

                net.to(device)
                net.load_state_dict(torch.load(path_to_kth_model, map_location=device))
                net.eval()

                preds_df = get_predictions(net, testloader=testloader, device=device)
                preds_df.rename(columns={"preds": str(k)}, inplace=True)

                all_predictions[:, rotation_index, k] = preds_df[str(k)].values

        for k in range(5):
            fold_mean_predictions[:, k] = np.mean(all_predictions[:, :, k], axis=1)
            fold_std_rotations[:, k] = np.std(all_predictions[:, :, k], axis=1)

        mean_predictions = np.mean(fold_mean_predictions, axis=1)
        std_predictions_folds = np.std(fold_mean_predictions, axis=1)

        gt = pd.DataFrame(
            {
                "id": np.arange(n_samples),
                "target": full_values,
                "mean_predictions": mean_predictions,
                "std_predictions_folds": std_predictions_folds,
            }
        )

        # for k in range(5):
        #     gt[f"fold_{k}_mean_predictions"] = fold_mean_predictions[:, k]
        #     gt[f"fold_{k}_std_rotations"] = fold_std_rotations[:, k]

    else:
        if random_rotations is None: # default
            if model_name == "OrgNet":
                random_rotations = True
            else:
                random_rotations = False
        
        test_dataset = VoxDataset(
            voxels=full_voxels,
            values=np.arange(n_samples),
            n_channels=n_channels,
            grid_size=grid_size,
            device=device,
            cubic_rotations=random_rotations,
            v_dtype=torch.int64,
        )

        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=0
        )

        fold_predicts = []

        for k, path_to_kth_model in enumerate(paths_to_kth_model):
            if paths_to_kth_model is None:
                net = OrgNet()
            else:
                if model_name == "OrgNet":
                    net = OrgNet()
                elif model_name == "ThermoNet":
                    net = ThermoNet(se3conv=False)
                elif model_name == "ThermoNet_steerable":
                    net = ThermoNet(se3conv=True, device=device)

            net.to(device)
            net.load_state_dict(torch.load(path_to_kth_model, map_location=device))
            net.eval()

            preds_df = get_predictions(net, testloader=testloader, device=device)
            preds_df.rename(columns={"preds": str(k)}, inplace=True)

            fold_predicts.append(preds_df[str(k)].values)

        mean_predictions = np.mean(fold_predicts, axis=0)
        std_predictions_folds = np.std(fold_predicts, axis=0)

        gt = pd.DataFrame(
            {
                "id": np.arange(n_samples),
                "target": full_values,
                "mean_predictions": mean_predictions,
                "std_predictions_folds": std_predictions_folds,
            }
        )

    metric_values = []

    for metric_name in ["RootMeanSquaredError", "PearsonCorrCoef", "MeanAbsoluteError"]:
        metric = metric_by_name(metric_name=metric_name, device="cpu")
        metric.reset()
        metric.update(
            torch.tensor(gt.mean_predictions.values),
            torch.tensor(gt["target"].values),
        )
        metric_value = metric.compute()
        metric_values.append(metric_value.cpu().numpy())

    if save_to:
        gt.to_csv(save_to, index=False)

    return metric_values


def _parse_args(args: Optional[str] = None):
    parser = argparse.ArgumentParser(description="OrgNet inference")
    parser.add_argument(
        "-X", "--path_to_X", required=True, help="Path to .npy file with voxels"
    )
    parser.add_argument(
        "-y", "--path_to_y", required=True, help="Path to .npy file with values"
    )
    parser.add_argument(
        "--model_name",
        choices=["OrgNet", "ThermoNet", "ThermoNet_steerable"],
        help="Model architecture (`ThermoNet`, `ThermoNet_steerable` or `OrgNet`)",
        default="OrgNet",
    )
    parser.add_argument(
        "--model_weights_dir",
        help="Directory with models' weights",
        default=None,
    )
    parser.add_argument(
        "--save_to", help=".csv path where predictions will be saved", default=None
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to run the inference on (`cpu` or `cuda`)",
        default="cpu",
    )
    parser.add_argument(
        "--trained_on",
        choices=["Q3214", "S2648_V"],
        help="Weights for OrgNet trained on (`Q3214` or `S2648_V`)",
        default="S2648_V",
    )
    parser.add_argument(
        "--random_rotations",
        type=lambda x: x.lower() == "true",
        help="Enable or disable random rotations (true/false). Default is None.",
        default=None,
    )
    parser.add_argument(
        "--fully_rotated",
        action="store_true",
        help="Enable fully rotated mode (default: False)",
    )
    return parser.parse_args(args=args)


def main(args: Optional[str] = None):
    args = _parse_args(args)

    assert args.path_to_X.endswith(".npy"), "path_to_X should be a .npy file"
    assert args.path_to_y.endswith(".npy"), "path_to_y should be a .npy file"
    if args.save_to:
        assert args.save_to.endswith(".csv"), "save_to should be a .csv file"

    if args.model_weights_dir:
        pt_files = get_pt_files(args.model_weights_dir)
    else:
        pt_files = None

    RMSE_, pearsonr_, mae_ = call_predict(
        args.path_to_X,
        args.path_to_y,
        save_to=args.save_to,
        device=args.device,
        training_data=args.trained_on,
        paths_to_kth_model=pt_files,
        model_name=args.model_name,
        random_rotations=args.random_rotations,
        fully_rotated=args.fully_rotated
    )
    print("  r  | RMSE | MAE")
    print(
        "%.2f" % round(pearsonr_.item(), 2),
        "|",
        "%.2f" % round(RMSE_.item(), 2),
        "|",
        "%.2f" % round(mae_.item(), 2),
    )


if __name__ == "__main__":
    main()
