import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="plot_loss.py",
        description="Plot loss from a saved checkpoint",
        epilog="This script is part of the ThinClouds project",
    )

    parser.add_argument("checkpoint", help="A *.pt file with a trained network")
    parser.add_argument(
        "-ss",
        "--style_sheet",
        help="If default matplotlib style sheet should be overridden",
    )
    parser.add_argument("--tex", help="Use LaTeX for titles", action=argparse.BooleanOptionalAction)
    parser.add_argument("-t", "--title", help="Set a custom figure title")

    args = parser.parse_args()

    font = {"family": "cursive", "weight": "normal", "size": 24}
    if args.style_sheet:
        font = {"family": "Times", "weight": "normal", "size": 24}
    plt.rc("font", **font)
    plt.rc("text", usetex=True if args.tex is None else args.tex)
    plt.rc("pdf", compression=9)

    if args.style_sheet:
        plt.style.use(args.style_sheet)

    title = (
        args.title
        if args.title
        else f"Training losses\nCheckpoint: {args.checkpoint}"
    )

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    losses = checkpoint["losses"]

    train_losses = [l[0] for l in losses]
    valid_losses = [l[1] for l in losses]

    train_psnrs = 20.0 * np.log10(1.0) - 10 * np.log10(np.array(train_losses))
    valid_psnrs = 20.0 * np.log10(1.0) - 10 * np.log10(np.array(valid_losses))

    # Plot metrics
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle(title)
    fig.set_figwidth(3.5)
    fig.set_figheight(3.5)
    fig.set_dpi(300)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.plot(range(len(train_losses)), train_losses, marker="s", label="Training loss (MSE)", color="tab:green")
    ax.plot(range(len(valid_losses)), valid_losses, marker="o", label="Validation loss (MSE)", color="tab:orange")
    ax.tick_params("y", labelcolor="tab:green")
    ax.set_xlim(0, 250)
    ax2 = ax.twinx()
    ax2.set_ylabel("PSNR")
    ax.plot(range(len(train_psnrs)), train_psnrs, marker="s", label="Training loss (PSNR)", color="tab:purple")
    ax.plot(range(len(valid_psnrs)), valid_psnrs, marker="o", label="Validation loss (PSNR)", color="tab:blue")
    ax2.tick_params("y", labelcolor="tab:purple")
    ax.legend(loc="upper left")
    # ax2.legend(loc="upper right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
