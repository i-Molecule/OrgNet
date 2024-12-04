import os
import os.path as osp
from typing import Literal, Union

import numpy as np
import pandas as pd
import torch
from data.vox_dataset import VoxDataset, load_voxels
from models.orgnet import OrgNet
from utils.helpers import get_predictions, metric_by_name


def call_predict(
    path_to_X: Union[str, os.PathLike],
    path_to_y: Union[str, os.PathLike],
    save_to: Union[str, os.PathLike] | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
    SEED=1213,
):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    full_voxels, n_samples, n_channels, grid_size, full_values = load_voxels(
        path_to_X, path_to_y
    )
    device = torch.device(device)

    test_dataset = VoxDataset(
        voxels=full_voxels,
        values=np.array(range(n_samples)),
        n_channels=n_channels,
        grid_size=grid_size,
        device=device,
    )

    shuffle = False

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=shuffle, num_workers=0
    )

    target_col = "target"
    gt = pd.DataFrame(
        {
            "id": np.array(range(n_samples)),
            target_col: full_values,
        }
    )

    fold_predicts = []

    for k, path_to_kth_model in enumerate(
        [
            osp.join("models", "weights", "orgnet", "d6c5b74e_0.pt"),
            osp.join("models", "weights", "orgnet", "d6c5b74e_1.pt"),
            osp.join("models", "weights", "orgnet", "d6c5b74e_2.pt"),
            osp.join("models", "weights", "orgnet", "d6c5b74e_3.pt"),
            osp.join("models", "weights", "orgnet", "d6c5b74e_4.pt"),
        ]
    ):
        net = OrgNet()
        net.to(device)
        net.load_state_dict(torch.load(path_to_kth_model, map_location=device))
        net.eval()

        preds_df = get_predictions(net, testloader=testloader, device=device)
        preds_df.rename(columns={"preds": str(k)}, inplace=True)

        gt = pd.DataFrame.merge(
            gt,
            preds_df[["id", str(k)]],
            on="id",
        )

        fold_predicts.append(str(k))

    gt["predictions"] = gt[fold_predicts].mean(axis=1)

    metric_values = []

    for metric_name in ["RootMeanSquaredError", "PearsonCorrCoef", "MeanAbsoluteError"]:
        metric = metric_by_name(metric_name=metric_name, device="cpu")
        metric.reset()
        metric_value = metric.update(
            torch.tensor(gt.predictions.values),
            torch.tensor(gt[target_col].values),
        )
        metric_value = metric.compute()
        metric_values.append(metric_value.cpu().numpy())

    if save_to:
        gt[["id", "predictions"]].to_csv(save_to, index=False)

    return metric_values
