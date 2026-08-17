import torch
import torch.utils.data

import OpenEXR as exr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

from scipy.signal import convolve2d

import flip_evaluator as flip

import argparse
import glob

from model import ThinCloudModel


torch.set_float32_matmul_precision('high')


def mipmap_level(img):
    # assume H and W are even
    return 0.25 * (
        img[0::2, 0::2] +
        img[1::2, 0::2] +
        img[0::2, 1::2] +
        img[1::2, 1::2]
    )


def box_blur(img, k):
    size = 2 ** k
    kernel = np.ones((size, size)) / (size * size)
    return convolve2d(img, kernel, mode="same", boundary="symm")


def build_mipmaps(img):
    levels = [img]
    while min(img.shape[:2]) > 1:
        img = mipmap_level(img)
        levels.append(img)
    return levels


def main():
    parser = argparse.ArgumentParser(
        prog="triplane_plot.py",
        description="Inspect triplanes of a model",
        epilog="This script is part of the ThinClouds project",
    )

    parser.add_argument("checkpoint", help="A *.pt file with a trained network")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]

    model = ThinCloudModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    plane_xy = model.triplanes.plane_xy.data.cpu().numpy()
    plane_xz = model.triplanes.plane_xz.data.cpu().numpy()
    plane_yz = model.triplanes.plane_yz.data.cpu().numpy()

    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle("Triplanes")
    fig.set_figwidth(15)
    fig.set_figheight(5)

    ax = fig.add_axes((0.95, 0.1, 0.03, 0.8))
    slider = Slider(
        ax=ax,
        label="Feature",
        valmin=0,
        valmax=plane_xy.shape[1] - 1,
        valinit=0,
        valstep=1,
        orientation="vertical",
    )

    def update(val):
        for ax in axs:
            ax.clear()
            ax.axis("off")

        p0 = plane_xy[0, val]
        p1 = plane_xz[0, val]
        p2 = plane_yz[0, val]

        # p1 and p2 need to flipped along the vertical axis
        p1 = np.flip(p1, axis=0)
        p2 = np.flip(p2, axis=0)

        cmap = plt.colormaps['berlin']
        # normalizer = Normalize(np.min((p0, p1, p2)), np.max((p0, p1, p2)))
        normalizer = Normalize(-0.5, 0.5)
        im = cm.ScalarMappable(norm=normalizer)

        im_xy = axs[0].imshow(p0, cmap=cmap, norm=normalizer)
        axs[0].set_title("Plane XY")

        im_xz = axs[1].imshow(p1, cmap=cmap, norm=normalizer)
        axs[1].set_title("Plane XZ")

        im_yz = axs[2].imshow(p2, cmap=cmap, norm=normalizer)
        axs[2].set_title("Plane YZ")

    slider.on_changed(update)
    slider.set_val(0)

    def save(event):
        print("Saving images")
        cmap = plt.colormaps['berlin']

        im_xy = np.clip(plane_xy[0, slider.val], -0.5, 0.5)
        im_xz = np.flip(np.clip(plane_xz[0, slider.val], -0.5, 0.5), axis=0)
        im_yz = np.flip(np.clip(plane_yz[0, slider.val], -0.5, 0.5), axis=0)

        plt.imsave(f"{slider.val}_plane_xy.png", im_xy, cmap=cmap)
        plt.imsave(f"{slider.val}_plane_xz.png", im_xz, cmap=cmap)
        plt.imsave(f"{slider.val}_plane_yz.png", im_yz, cmap=cmap)

        # Save mipmaps
        if True:
            mipmap_level = 4
            ims_xy = box_blur(np.clip(plane_xy[0, slider.val], -0.5, 0.5), 4)
            ims_xz = box_blur(np.flip(np.clip(plane_xz[0, slider.val], -0.5, 0.5), axis=0), 4)
            ims_yz = box_blur(np.flip(np.clip(plane_yz[0, slider.val], -0.5, 0.5), axis=0), 4)

            plt.imsave(f"{slider.val}_plane_xy_mip{mipmap_level}.png", ims_xy, cmap=cmap)
            plt.imsave(f"{slider.val}_plane_xz_mip{mipmap_level}.png", ims_xz, cmap=cmap)
            plt.imsave(f"{slider.val}_plane_yz_mip{mipmap_level}.png", ims_yz, cmap=cmap)

    ax = fig.add_axes((0.90, 0.02, 0.03, 0.02))
    save_button = Button(ax, "Save")
    save_button.on_clicked(save)

    plt.show()


if __name__ == "__main__":
    main()
