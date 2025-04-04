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

    if model_name == "OrgNet":
        cubic_rotations = True
    else:
        cubic_rotations = False

    test_dataset = VoxDataset(
        voxels=full_voxels,
        values=np.arange(n_samples),
        n_channels=n_channels,
        grid_size=grid_size,
        device=device,
        cubic_rotations=cubic_rotations,
        v_dtype=torch.int64,
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    fold_predicts = []

    if paths_to_kth_model is None:
        paths_to_kth_model = [
            osp.join("models", "weights", "orgnet", training_data, f"{k}.pt")
            for k in range(5)
        ]

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
