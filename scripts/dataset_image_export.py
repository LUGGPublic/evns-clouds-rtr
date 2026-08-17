import numpy as np
import OpenEXR as exr
import glob
import torch

import matplotlib.pyplot as plt

from torch.utils.data import Dataset

def read_exr(file, channel):
    with exr.File(file) as infile:
        return infile.channels()[channel].pixels

    
def read_dataset_files(glob_pattern, channels, limit=None):
    files = sorted(glob.glob(glob_pattern))
    if limit is not None:
        files = files[:limit]

    data = []
    for file in files:
        print(f"\r\t{file:128}", end="")
        if channels in ("RGB", "RGBA"):
            data.append(read_exr(file, channels))
        else: # XYZ
            pixels = []
            for c in channels:
                pixels.append(read_exr(file, c))
            data.append(np.array(np.transpose(pixels, (1, 2, 0))))

    return data


def linear_to_srgb(img):
    rgb = img[..., :3]
    alpha = img[..., 3:]

    srgb = np.where(rgb > 0.0031308,
                    1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
                    12.92 * rgb)

    return np.concatenate([srgb, alpha], axis=-1)


def exposure(img, exposure):
    """Apply exposure to an image."""
    rgb = img[..., :3]
    alpha = img[..., 3:]
    exposed_rgb = 1.0 - np.exp(-rgb * exposure)
    return np.concatenate([exposed_rgb, alpha], axis=-1)


def alpha_blend_to_black(rgba, alpha=None):
    """Blend an RGBA image with a black background."""
    rgb = rgba[..., :3]
    if alpha is None:
        alpha = rgba[..., 3:]
    # Composite over black: C = alpha * foreground + (1 - alpha) * background
    black_bg = np.zeros_like(rgb)
    blended_rgb = alpha * rgb + (1 - alpha) * black_bg
    return blended_rgb


def main():
    folder = "D:\\scenes\\meshed_volumes\\wdas_mesh\\eval_20260112_162339_100"
    color = read_dataset_files(folder + "/color0000.exr", "RGBA")
    pos = read_dataset_files(folder + "/position0000.exr", "XYZ")
    vis = read_dataset_files(folder + "/visibility0000.exr", "RGB")
    back = read_dataset_files(folder + "/back0000.exr", "XYZ")
    normal = read_dataset_files(folder + "/normal0000.exr", "XYZ")
    labels = np.genfromtxt(folder + "/labels.csv", delimiter=",", skip_header=1)
    bounds = np.genfromtxt(folder + "/bounds.csv", delimiter=",", skip_header=1)

    resolution = color[0].shape[0:2]
    alpha = color[0][:, :, 3]

    # Extract bounds
    min_bound = np.array(bounds[0:3])
    max_bound = np.array(bounds[3:6])

    # Extract view directions from labels
    view = labels[:, 1:4]

    # Sun directions from labels
    sun = labels[:, 4:6]

    print(f"\rGenerating dataset{'':128}")

    # Back images need to be flipped horizontally
    for i in range(len(back)):
        back[i] = np.flip(back[i], axis=1)

    from scipy.special import comb
    def smoothstep(x, x_min=0.0, x_max=1.0, N=1):
        x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

        result = 0
        for n in range(0, N + 1):
                result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

        result *= x ** (N + 1)

        return result

    # Create gray scale image (luminosity method)
    vis = [0.299 * v[:, :, 0] + 0.587 * v[:, :, 1] + 0.114 * v[:, :, 2] for v in vis]

    # Apply smoothstep to visibility
    vis = [smoothstep(v, x_min=0.0, x_max=0.1, N=3) for v in vis]

    # Normalize positions to [0,1]
    for i in range(len(pos)):
        pos[i] = (pos[i] - min_bound) / (max_bound - min_bound)
        back[i] = (back[i] - min_bound) / (max_bound - min_bound)

    # Calculate thickness
    thickness = []
    for i in range(len(pos)):
        thickness.append(np.linalg.norm(back[i] - pos[i], axis=2, keepdims=True))

    # Mask out pixels with zero alpha
    for i in range(len(pos)):
        pos[i][alpha == 0] = 0
        back[i][alpha == 0] = 0
        vis[i][alpha == 0] = 0


    # Save all images
    plt.imsave(f"color0000.png", alpha_blend_to_black(np.clip(linear_to_srgb(exposure(color[0], 1.0)), 0.0, 1.0)))
    plt.imsave(f"position0000.png", pos[0])
    plt.imsave(f"visibility0000.png", vis[0], cmap='gray')
    plt.imsave(f"back0000.png", back[0])



if __name__ == "__main__":
    main()