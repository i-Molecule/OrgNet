import numpy as np
import pandas as pd
import torch
import torchmetrics

seeds = {"S2648_V": 1213, "Q3214": 1527}


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
