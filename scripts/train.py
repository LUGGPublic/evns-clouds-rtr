import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.profiler import profile, ProfilerActivity, record_function
from tqdm import tqdm

import argparse
import configparser
import datetime
import os
import shutil
import time

from model import ThinCloudModel
from dataset import ThinCloudDataset


PROFILE = False


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save(model, optimizer, scheduler, losses, epoch, config, checkpoint=False):
    # Model
    model.save(f"{args.output}/{get_timestamp()}_{epoch + 1}.mlp")

    # Checkpoint
    if checkpoint:
        checkpoint_path = f"{args.output}/checkpoints"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(
            {
                "config": config,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dic": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": losses,
            },
            f"{checkpoint_path}/{get_timestamp()}_{epoch + 1}.pt",
        )


def train(
    config,
    args,
    dataset,
    checkpoint=None,
    device="cuda",
):
    # Create a model
    with torch.device(config["general"]["device"]):
        model = ThinCloudModel(config)
        if checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Scheduled learning rate decay
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"]["lr_step_size"]),
        gamma=float(config["training"]["lr_step_gamma"]),
    )

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dic"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if args.save_initial and not args.dry_run:
        model.save(f"{args.output}/{get_timestamp()}_{0}.mlp")

    model = model.to(device=device)

    start = time.time()
    epoch_start = 0
    epochs_total = int(config["training"]["epochs"])
    losses = []

    if checkpoint:
        epoch_start = checkpoint["epoch"] + 1
        losses = checkpoint["losses"]

    # Loop over epochs
    for e in range(epoch_start, epochs_total):
        batch_progress = tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"Epoch {e+1}/{epochs_total}",
            leave=True,
        )

        dataset.shuffle()

        model.train()

        if PROFILE:
            def print_profiler_summary(prof_export):
                """
                This function will be called by the profiler when a trace is ready.
                'prof_export' is a profiler object for the completed trace.
                """
                print(f"\n--- Profiler Summary for trace (step {prof_export.step_num}) ---")
                print(prof_export.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
                print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
                print(prof_export.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                on_trace_ready=print_profiler_summary,
                ) as prof:
                for i, data in batch_progress:
                    if i > 1:
                        break

                    with record_function("data_transfer"):
                        data = [i.to(device, non_blocking=True) for i in data]
                        color, pos, normal, thickness, view, sun = data
                        # Zero your gradients for every batch!
                        optimizer.zero_grad()

                    with record_function("foward_pass"):
                        # Make predictions for this batch
                        pred_color = model((pos, view, sun, thickness))
                        loss = loss_fn(pred_color, color)

                    with record_function("backward_pass"):
                        loss.backward()

                    with record_function("optimizer_step"):
                        # Adjust learning weights
                        optimizer.step()

                    prof.step()

            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            prof.export_chrome_trace("trace.json")
            exit(0)

        loss_avg = 0.0
        for i, data in batch_progress:
            # Transfer data if it is not on the device already
            if config["general"]["dataset_device"] != config["general"]["device"]:
                data = [i.to(device, non_blocking=True) for i in data]
            color, pos, vis, normal, thickness, view, sun = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            output = model((pos, view, sun, thickness))
            rgb_vis = output[:, :3]
            rgb_hid = output[:, 3:6]
            rgb = rgb_vis * vis + rgb_hid * (1.0 - vis)
            alpha = output[:, 6].unsqueeze(-1)
            pred_color = torch.cat((rgb, alpha), dim=-1)

            loss = loss_fn(torch.log1p(pred_color), torch.log1p(color))
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and 
            mse = loss.detach().item()
            loss_avg += mse
            psnr = 10.0 * np.log10(1.0 / mse)
            batch_progress.set_postfix({"PSNR": psnr})

        # Step the scheduler
        scheduler.step()

        # Validation
        vloss_avg = 0.0
        model.eval()

        validation_batches = dataset.get_number_of_validation_batches()
        with torch.no_grad():
            for i in range(validation_batches):
                data = dataset.get_validation_batch(i)
                # Transfer data if it is not on the device already
                if config["general"]["dataset_device"] != config["general"]["device"]:
                    data = [i.to(device, non_blocking=True) for i in data]
                color, pos, vis, normal, thickness, view, sun = data

                output = model((pos, view, sun, thickness))
                rgb_vis = output[:, :3]
                rgb_hid = output[:, 3:6]
                rgb = rgb_vis * vis + rgb_hid * (1.0 - vis)
                alpha = output[:, 6].unsqueeze(-1)
                pred_color = torch.cat((rgb, alpha), dim=-1)

                vloss = loss_fn(torch.log1p(pred_color), torch.log1p(color))
                vloss_avg += vloss.item()

        # Save losses
        loss_avg /= len(dataset)
        vloss_avg /= validation_batches if validation_batches > 0 else 1
        losses.append((loss_avg, vloss_avg))

        # Save
        if not args.dry_run and args.save_all:
            save(model, optimizer, scheduler, losses, e, config, args.checkpoint)

    if not args.dry_run:
        save(model, optimizer, scheduler, losses, e, config, args.checkpoint)

    print(f"Total training time: {(time.time() - start) / 60:8.2f} m")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a ThinCloud model on a dataset",
        epilog="This script is part of the ThinClouds project",
    )

    parser.add_argument("config")
    parser.add_argument(
        "-o", "--output", default=".", type=str, help="Output directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run without saving")
    parser.add_argument(
        "-si", "--save-initial", action="store_true", help="Save initial state"
    )
    parser.add_argument(
        "-sa", "--save-all", action="store_true", help="Save every epoch"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        action="store_true",
        help="Save each epoch in a checkpoint that can be used to continue training later on",
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        help="Load a checkpoint and continue training from there",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    checkpoint = None
    if args.load:
        checkpoint = torch.load(args.load, weights_only=False)
        config = checkpoint["config"]
    else:
        config = configparser.ConfigParser()
        config.read(args.config)

    # Save config file
    if not args.dry_run:
        shutil.copy(args.config, args.output)

    def convert_dtype(dtype_str) -> torch.dtype:
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "float32":
            return torch.float32
        if dtype_str == "float64":
            return torch.float64
        print("Config error: unsupported dtype")
        return torch.float

    # Dataset device might differ from training device
    config["general"]["dataset_device"] = (
        config["general"]["device"]
        if config["training"]["keep_on_device"].lower() in ("true", "yes", "1")
        else "cpu"
    )

    torch.set_default_device(config["general"]["dataset_device"])
    torch.set_default_dtype(convert_dtype(config["general"]["dtype"]))

    if config["general"]["dtype"] == "float32":
        torch.set_float32_matmul_precision("high")

    dataset = ThinCloudDataset(config, dtype=convert_dtype(config["general"]["dtype"]))

    train(config, args, dataset, checkpoint)
