import torch
import sys

import argparse
import glob

from model import ThinCloudModel


def main():
    parser = argparse.ArgumentParser(
        prog="export_mlp.py",
        description="Save MLP file from a checkpoint",
        epilog="This script is part of the ThinClouds project",
    )

    parser.add_argument("checkpoint", help="A *.pt file with a trained network")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]
    model = ThinCloudModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.save(args.checkpoint.replace(".pt", ".mlp"))

if __name__ == "__main__":
    main()
