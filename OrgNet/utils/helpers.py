from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics

seeds = {"S2648_V": 1213, "Q3214": 1527}


def get_pt_files(folder_path: str):
    folder = Path(folder_path)
    pt_files = list(folder.glob("*.pt"))
    return pt_files


def make_norm(channels, norm="bn"):
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    elif norm == "in":
        return nn.InstanceNorm3d(channels)
    elif norm == "None" or norm is None:
        return nn.Identity()
    else:
        raise ValueError("Implemented normalization layers: bn, in, None")


def make_active(active="relu"):
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leakyrelu": nn.LeakyReLU(inplace=True),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "logsigmoid": nn.LogSigmoid(),
        "sigmoid": nn.Sigmoid(),
        "None": nn.Identity(),
    }
    return activations.get(active, nn.Identity())


def train_one_epoch(net, trainloader, loss_function, optimizer, device):
    running_loss = 0.0
    epoch_steps = 0
    for _, data in enumerate(trainloader, 0):
        input, target = data
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(input)

        target = target.unsqueeze(1)
        if target.shape == output.shape:
            loss = loss_function(output, target)
        else:
            raise ValueError(
                f"Shapes of target and output do not match: {target.shape} vs {output.shape}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        epoch_steps += 1
    return loss.item()


def validate_one_epoch(net, valloader, loss_function, metric, device):
    val_loss = 0.0
    val_steps = 0

    metric.reset()
    for _, data in enumerate(valloader, 0):
        with torch.no_grad():
            input, target = data
            input, target = input.to(device), target.to(device)
            output = net(input)

            target = target.unsqueeze(1)
            if target.shape == output.shape:
                loss = loss_function(output, target)
            else:
                raise ValueError(
                    f"Shapes of target and output do not match: {target.shape} vs {output.shape}"
                )
            metric_value = metric.update(output, target)

            val_loss += loss.cpu().numpy()
            val_steps += 1
    loss = val_loss / val_steps
    metric_value = metric.compute()
    return metric_value, loss


def predict_step(data, device, net):
    input, mol_id = data
    input, mol_id = input.to(device), mol_id.to(device)
    output = net(input)
    mol_id = mol_id.unsqueeze(1)
    if mol_id.shape == output.shape:
        return np.squeeze(output.cpu().numpy()).tolist(), np.squeeze(
            mol_id.cpu().numpy()
        ).tolist()
    else:
        raise ValueError(
            f"Shapes of mol_id and output do not match: {mol_id.shape} vs {output.shape}"
        )


def get_predictions(net, testloader, device):
    preds = []
    ids = []

    for i, batch in enumerate(testloader):
        with torch.no_grad():
            output, mol_id = predict_step(batch, device, net)
            preds.append(output)
            ids.append(mol_id)

    preds_df = pd.DataFrame()
    preds_df["id"] = np.concatenate(ids)
    preds_df["preds"] = np.concatenate(preds)
    preds_df = preds_df.sort_values(by="id")

    return preds_df


def metric_by_name(metric_name: str, device="cpu"):
    metric_by_name_dict = {
        "MeanAbsoluteError": torchmetrics.MeanAbsoluteError().to(device),
        "RootMeanSquaredError": torchmetrics.MeanSquaredError(squared=False).to(device),
        "PearsonCorrCoef": torchmetrics.regression.PearsonCorrCoef().to(device),
    }
    return metric_by_name_dict[metric_name]


loss_by_name_dict = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCELoss(),
    "SmoothL1Loss": nn.SmoothL1Loss(),
}
