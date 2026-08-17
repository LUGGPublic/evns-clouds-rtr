import torch
import torch.utils.data

import OpenEXR as exr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

import flip_evaluator as flip

import argparse
import glob

from model import ThinCloudModel
from dataset import ThinCloudDataset


torch.set_float32_matmul_precision('high')


def linear_to_srgb(img):
    rgb = img[..., :3]
    alpha = img[..., 3:]

    srgb = np.where(rgb > 0.0031308,
                    1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
                    12.92 * rgb)

    return np.concatenate([srgb, alpha], axis=-1)


def compute_exposures(img, tm: str):
    # Tone mapper selection
    tone_mapper = {"reinhard": 0, "aces": 1, "hable": 2}.get(tm, 1)

    # Tone mapping coefficients
    coefficients = [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.6 * 0.6 * 2.51, 0.6 * 0.03, 0.0, 0.6 * 0.6 * 2.43, 0.6 * 0.59, 0.14],
        [0.231683, 0.013791, 0.0, 0.18, 0.3, 0.018]
    ]
    tc = coefficients[tone_mapper]

    t = 0.85
    a = tc[0] - t * tc[3]
    b = tc[1] - t * tc[4]
    c = tc[2] - t * tc[5]

    # Solve a*x^2 + b*x + c = 0
    disc = b * b - 4 * a * c
    if disc < 0:
        x_min = x_max = -b / (2 * a)
    else:
        sqrt_disc = np.sqrt(disc)
        x1 = (-b - sqrt_disc) / (2 * a)
        x2 = (-b + sqrt_disc) / (2 * a)
        x_min, x_max = min(x1, x2), max(x1, x2)

    # Compute luminances
    # Assuming image is an array of shape (H, W, 3) in linear RGB
    luminances = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    nonzero_lum = luminances[luminances > 0]
    if nonzero_lum.size > 0:
        Ymin = np.min(nonzero_lum)
    else:
        Ymin = 0.0
    Ymax = np.max(luminances)

    # Median luminance (ignore zeros)
    Ymedian = np.median(luminances)
    Ymedian = max(Ymedian, np.finfo(np.float32).eps)

    start_exposure = np.log2(x_max / Ymax)
    stop_exposure = np.log2(x_max / Ymedian)

    return start_exposure, stop_exposure


def exposure(img, exposure):
    """Apply exposure to an image."""
    rgb = img[..., :3]
    alpha = img[..., 3:]
    exposed_rgb = 1.0 - np.exp(-rgb * exposure)
    return np.concatenate([exposed_rgb, alpha], axis=-1)


def alpha_blend_to_white(rgba, alpha=None):
    """Blend an RGBA image with a white background."""
    rgb = rgba[..., :3]
    if alpha is None:
        alpha = rgba[..., 3:]
    # Composite over white: C = alpha * foreground + (1 - alpha) * background
    white_bg = np.ones_like(rgb)
    blended_rgb = alpha * rgb + (1 - alpha) * white_bg
    return blended_rgb


def alpha_blend_to_black(rgba, alpha=None):
    """Blend an RGBA image with a black background."""
    rgb = rgba[..., :3]
    if alpha is None:
        alpha = rgba[..., 3:]
    # Composite over black: C = alpha * foreground + (1 - alpha) * background
    black_bg = np.zeros_like(rgb)
    blended_rgb = alpha * rgb + (1 - alpha) * black_bg
    return blended_rgb


from scipy.special import comb
def smoothstep(x, x_min=0.0, x_max=1.0, N=3):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
            result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate a trained model",
        epilog="This script is part of the ThinClouds project",
    )

    parser.add_argument("dataset", help="Dataset folder")
    parser.add_argument("checkpoint", help="A *.pt file with a trained network")
    parser.add_argument("-n", "--dataset_limit", default=10, type=int, help="Set how many samples from evaluation dataset that should be loaded")
    parser.add_argument("-x", "--xres", default=1024, type=int, help="Resolution width")
    parser.add_argument(
        "-y", "--yres", default=1024, type=int, help="Resolution height"
    )
    parser.add_argument(
        "-dt", "--dtype", default="float32", help="Datatype for dataset"
    )
    parser.add_argument(
        "-ss",
        "--style_sheet",
        help="If default matplotlib style sheet should be overridden",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Save plots to folder instead of displaying",
    )
    parser.add_argument("-t", "--title", help="Set a custom figure title")
    parser.add_argument("--tex", help="Use LaTeX for titles", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-fp16",
        "--half",
        action="store_true",
        help="Use 16-bit floating point precision",
    )
    parser.add_argument("-e", "--exposure", default=1.0, type=float, help="Set exposure value for displaying images")
    parser.add_argument("-m", "--mipmap", default=0, type=int, help="Set triplane mipmap level")

    args = parser.parse_args()

    config = {"training": {"dataset": args.dataset}}

    def convert_dtype(dtype_str):
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "float32":
            return torch.float32
        if dtype_str == "float64":
            return torch.float64
        print("Config error: unsupported dtype")

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]

    config["training"]["dataset"] = args.dataset
    config["training"]["batch_size"] = str(int(args.xres) * int(args.yres))
    config["training"]["dataset_limit"] = str(args.dataset_limit)
    config["training"]["validation"] = str(0.0)

    device = "cuda"

    if args.dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    # Old cloud models are missing parameter
    if "model" not in config["general"]:
        config["general"]["model"] = "cloud"

    model = ThinCloudModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.half:
        model.half()

    dataset = ThinCloudDataset(
        config, dtype=convert_dtype(args.dtype), evaluate=True
    )

    def psnr(img1, img2):
        mask = img1[..., 3] > 0.0
        mse = np.mean((img1[mask] - img2[mask]) ** 2.0)
        if mse == 0:
            return np.Inf
        return 20.0 * np.log10(1.0) - 10 * np.log10(mse)

    # def psnr(img1, img2, max_value=1.0):
    #     mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    #     if mse == 0:
    #         return 100
    #     return 20 * np.log10(max_value / (np.sqrt(mse)))

    def mpsnr(img1, img2, params):
        def tone_map(img, exp, gamma=2.2):
            return np.clip(np.power(np.power(2.0, exp) * img, 1.0 / gamma), 0.0, 1.0);

        # Create series of exposures in first dimension
        exps = np.linspace(params["startExposure"], params["stopExposure"], params["numExposures"])

        img1s = [tone_map(img1[..., 0:3], exp) for exp in exps]
        img2s = [tone_map(img2[..., 0:3], exp) for exp in exps]

        img1s = np.stack(img1s, axis=0)
        img2s = np.stack(img2s, axis=0)

        mse = np.mean((img1 - img2) ** 2.0)
        if mse == 0:
            return np.Inf

        return 10.0 * np.log10(3.0 / mse)

    print("Inferring MLP images")
    images_mlp = []
    images_gt = []
    images_mlp_alpha = []
    images_gt_alpha = []
    images_visibility = []
    images_position = []
    images_thickness = []
    images_thickness_smoothstepped = []

    n = 0
    for data in dataset:
        n += 1
        if n > args.dataset_limit:
            break

        data = [i.to(device, non_blocking=True) for i in data]

        if args.half:
            data = [i.half() for i in data]

        color, pos, vis, normal, thickness, view, sun = data
        
        #t2 = torch.tensor(smoothstep(thickness.cpu().numpy(), x_min=0.0, x_max=0.4, N=3)).to("cuda")

        output = model((pos, view, sun, thickness), args.mipmap)
        #output = model((pos, view, sun, t2), args.mipmap)
        rgb_vis = output[:, :3]
        rgb_hid = output[:, 3:6]
        rgb = rgb_vis * vis + rgb_hid * (1.0 - vis)
        # rgb = rgb_vis
        alpha = output[:, 6].unsqueeze(-1)
        pred_color = torch.cat((rgb, alpha), dim=-1)

        pred_color = pred_color.reshape((args.xres, args.yres, -1))
        pred_color = pred_color.cpu().detach().numpy()
        color = color.reshape((args.xres, args.yres, -1))
        color = color.cpu().detach().numpy()

        images_mlp.append(pred_color.astype(np.float32))
        images_mlp_alpha.append(np.clip(pred_color[..., 3], 0.0, 1.0))
        images_gt.append(color)
        images_gt_alpha.append(color[..., 3])
        images_visibility.append(vis.cpu().numpy().reshape((args.xres, args.yres)))
        images_position.append(pos.cpu().numpy().reshape((args.xres, args.yres, 3)))
        images_thickness.append(np.clip(thickness.cpu().numpy().reshape((args.xres, args.yres)), 0.0, 1.0))
        images_thickness_smoothstepped.append(smoothstep(thickness.cpu().numpy().reshape((args.xres, args.yres)), x_min=0.0, x_max=0.2, N=3))

    print("Calculating image metrics")
    psnrs = []
    flips = []
    for img_ref, img_test in zip(images_gt, images_mlp):
        ref = alpha_blend_to_black(img_ref)
        test = alpha_blend_to_black(img_test, alpha=img_ref[..., 3:])
        start_exp, stop_exp = compute_exposures(ref, "aces")
        num_exps = int(max(2.0, stop_exp - start_exp))
        params = {"startExposure": start_exp, "stopExposure": stop_exp, "numExposures": num_exps}
        #psnrs.append(mpsnr(ref, test, params))
        psnrs.append(psnr(exposure(img_ref, args.exposure), exposure(img_test, args.exposure)))
        # flips.append(flip.evaluate(ref, test, "HDR", parameters=params))
        flips.append(flip.evaluate(alpha_blend_to_black(exposure(img_ref, args.exposure)),
                                   alpha_blend_to_black(exposure(img_test, args.exposure), alpha=img_ref[..., 3:]),
                                   "LDR"))

    psnrs = np.array(psnrs)
    avg_psnr = psnrs.mean()
    avg_flip = np.mean([f[1] for f in flips])

    print("Plotting")
    
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
        else f"Dataset: {args.dataset}\nCheckpoint: {args.checkpoint}"
    )

    # Plot metrics
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle(title)
    fig.set_figwidth(3.5)
    fig.set_figheight(3.5)
    fig.set_dpi(300)
    ax.set_xlabel("Evaluation Dataset Sample")
    ax.set_ylabel("dB")
    ax.plot(range(args.dataset_limit), psnrs, marker="s", label="PSNR", color="tab:green")
    ax.tick_params("y", labelcolor="tab:green")
    #ax.set_ylim(11, 23)
    ax2 = ax.twinx()
    ax2.set_ylabel("Error")
    ax2.plot(range(args.dataset_limit), [f[1] for f in flips], marker="^", label="FLIP", color="tab:purple")
    ax2.tick_params("y", labelcolor="tab:purple")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Plot dataset
    fig, axs = plt.subplots(3, 3)

    fig.set_figwidth(19.20)
    fig.set_figheight(10.80)

    fig.suptitle(title)

    fig.text(0.35, 0.04, f"Average PSNR: {avg_psnr:6.3f} dB")
    fig.text(0.35, 0.06, f"Average FLIP: {avg_flip:6.3f}")

    ax = fig.add_axes((0.95, 0.1, 0.03, 0.8))
    slider = Slider(
        ax=ax,
        label="Dataset",
        valmin=0,
        valmax=args.dataset_limit - 1,
        valinit=0,
        valstep=1,
        orientation="vertical",
    )

    def update(val):
        for ax in axs:
            for a in ax:
                a.clear()
                a.axis("off")

        axs[0][0].imshow(alpha_blend_to_black(linear_to_srgb(exposure(images_gt[int(val)], args.exposure))))
        axs[0][1].imshow(alpha_blend_to_black(linear_to_srgb(exposure(images_mlp[int(val)], args.exposure))))
        axs[0][2].imshow(flips[int(val)][0])

        axs[0][0].set_title("Path-Traced Ground Truth")
        axs[0][1].set_title("Thin Cloud")
        axs[0][2].set_title("FLIP")

        axs[0][1].annotate(
            f"{psnrs[int(val)]:6.3f} dB",
            xy=(0.95, 0.01),
            xycoords="axes fraction",
            fontsize=24,
            color="w",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        axs[0][2].annotate(
            f"{flips[int(val)][1]:6.3f}",
            xy=(0.95, 0.01),
            xycoords="axes fraction",
            fontsize=24,
            color="w",
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        cmap = plt.colormaps['viridis']
        normalizer = Normalize(0, 1)
        im = cm.ScalarMappable(norm=normalizer)

        axs[1][0].imshow(images_gt_alpha[int(val)], cmap=cmap, norm=normalizer)  # Alpha GT
        axs[1][1].imshow(images_mlp_alpha[int(val)], cmap=cmap, norm=normalizer)  # Alpha MLP
        axs[1][0].set_title("Alpha Channel")
        axs[1][1].set_title("Alpha Channel")

        axs[1][2].imshow(images_visibility[int(val)], cmap=cmap, norm=normalizer)  # Visibility
        axs[1][2].set_title("Visibility Input")

        axs[2][2].imshow(images_position[int(val)], cmap=cmap, norm=normalizer)  # Position
        axs[2][2].set_title("Position Input")

        axs[2][0].imshow(images_thickness[int(val)], cmap=cmap, norm=normalizer)  # Thickness
        axs[2][1].imshow(images_thickness_smoothstepped[int(val)], cmap=cmap, norm=normalizer)  # Thickness Smoothstepped
        axs[2][0].set_title("Thickness Input")
        axs[2][1].set_title("Thickness Input Smoothstep")

        #fig.colorbar(im, ax=axs.ravel().tolist())

    slider.on_changed(update)
    slider.set_val(0)

    def save(event):
        print("Saving images")
        cmap = plt.colormaps['viridis']
        normalizer = Normalize(0, 1)
        plt.imsave(f"{slider.val}_reference.png", np.clip(alpha_blend_to_black(linear_to_srgb(exposure(images_gt[int(slider.val)], args.exposure))), 0.0, 1.0))
        plt.imsave(f"{slider.val}_thincloud.png", np.clip(alpha_blend_to_black(linear_to_srgb(exposure(images_mlp[int(slider.val)], args.exposure))), 0.0, 1.0))
        plt.imsave(f"{slider.val}_flip.png", flips[int(slider.val)][0])
        plt.imsave(f"{slider.val}_alpha_reference.png", np.clip(images_gt_alpha[int(slider.val)], 0.0, 1.0), cmap=cmap)
        plt.imsave(f"{slider.val}_alpha_thincloud.png", np.clip(images_mlp_alpha[int(slider.val)], 0.0, 1.0), cmap=cmap)

    ax = fig.add_axes((0.90, 0.02, 0.03, 0.02))
    save_button = Button(ax, "Save")
    save_button.on_clicked(save)

    for ax in axs:
        for a in ax:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)

    if args.save:
        for v in range(args.dataset_limit):
            slider.set_val(v)
            plt.savefig(f"{args.save}/sample{v}.png", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
