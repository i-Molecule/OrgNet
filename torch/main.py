import argparse

from predict import call_predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrgNet inference")
    parser.add_argument(
        "-X", "--path_to_X", required=True, help="Path to .npy file with voxels"
    )
    parser.add_argument(
        "-y", "--path_to_y", required=True, help="Path to .npy file with values"
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

    args = parser.parse_args()

    assert args.path_to_X.endswith(".npy"), "path_to_X should be a .npy file"
    assert args.path_to_y.endswith(".npy"), "path_to_y should be a .npy file"
    if args.save_to:
        assert args.save_to.endswith(".csv"), "save_to should be a .csv file"

    RMSE_, pearsonr_, mae_ = call_predict(
        args.path_to_X, args.path_to_y, save_to=args.save_to, device=args.device, training_data=args.trained_on
    )
    print("  r  | RMSE | MAE")
    print(
        "%.2f" % round(pearsonr_.item(), 2),
        "|",
        "%.2f" % round(RMSE_.item(), 2),
        "|",
        "%.2f" % round(mae_.item(), 2),
    )
