import torch
import torch.utils.data

import numpy as np
import OpenEXR as exr

import glob

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
        else:
            pixels = []
            for c in channels:
                pixels.append(read_exr(file, c))
            data.append(np.array(np.transpose(pixels, (1, 2, 0))))  # XYZ

    return data


class ThinCloudDataset(Dataset):
    def __init__(self, config, dtype, evaluate=False):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.evaluate = evaluate

        dataset = torch.load(config["training"]["dataset"])
        self.color = dataset["color"]
        self.pos = dataset["pos"]
        self.vis = dataset["vis"]
        self.normal = dataset["normal"]
        self.thickness = dataset["thickness"]
        self.view = dataset["view"]
        self.sun = dataset["sun"]
        self.resolution = dataset["resolution"]
        self.min_bound = dataset["min_bound"]
        self.max_bound = dataset["max_bound"]

        # Filter based on transparent pixels
        if not evaluate:
            transparent = self.color[:, 3] < 1e-4
            keep = ~transparent
            self.color = self.color[keep]
            self.pos = self.pos[keep]
            self.vis = self.vis[keep]
            self.normal = self.normal[keep]
            self.thickness = self.thickness[keep]
            self.view = self.view[keep]
            self.sun = self.sun[keep]

        # Reduce to fraction if requested
        if "dataset_fraction" in config["training"] and not evaluate:
            device = config["general"]["dataset_device"]
            fraction = float(config["training"]["dataset_fraction"])
            total = self.color.shape[0]
            subset_size = int(total * fraction)

            subset_indices = torch.randperm(total, device=self.color.device)[
                :subset_size
            ]

            self.color = self.color[subset_indices].to(device)
            self.pos = self.pos[subset_indices].to(device)
            self.vis = self.vis[subset_indices].to(device)
            self.normal = self.normal[subset_indices].to(device)
            self.thickness = self.thickness[subset_indices].to(device)
            self.view = self.view[subset_indices].to(device)
            self.sun = self.sun[subset_indices].to(device)

        torch.cuda.empty_cache()

        # Validation subset
        if "validation" in config["training"]:
            validation_fraction = float(config["training"]["validation"])
        else:
            validation_fraction = 0.0

        total = self.color.shape[0]
        validation_size = int(total * validation_fraction)
        train_size = total - validation_size

        if self.evaluate:
            all_indices = torch.arange(total, device=self.color.device)
        else:
            all_indices = torch.randperm(total, device=self.color.device)

        # Taining and validation indices
        self.indices = all_indices[:train_size]
        self.validation_indices = all_indices[train_size:]

        # Batches
        if evaluate:
            self.batch_size = self.resolution[0] * self.resolution[1]
        else:
            self.batch_size = int(config["training"]["batch_size"])
        self.number_of_batches = self.indices.shape[0] // self.batch_size
        self.validation_batches = self.validation_indices.shape[0] // self.batch_size

        # Write size of dataset
        size_in_bytes = (
            self.color.nelement() * self.color.element_size()
            + self.pos.nelement() * self.pos.element_size()
            + self.vis.nelement() * self.vis.element_size()
            + self.normal.nelement() * self.normal.element_size()
            + self.thickness.nelement() * self.thickness.element_size()
            + self.view.nelement() * self.view.element_size()
            + self.sun.nelement() * self.sun.element_size()
        )

        print("Dataset:")
        print(f"\tDevice: {self.color.device}")
        print(f"\t{self.color.shape[0]:,} samples".replace(",", " "))
        print(f"\t{size_in_bytes / 10**9} GB")


    def __len__(self):
        return self.number_of_batches

    def __iter__(self):
        for i in range(self.number_of_batches):
            yield self[i]

    def __getitem__(self, idx):
        if idx >= self.number_of_batches:
            raise IndexError(f"Batch index {idx} out of range.")

        start = idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]

        color_batch = self.color[batch_indices]
        pos_batch = self.pos[batch_indices]
        vis_batch = self.vis[batch_indices]
        normal_batch = self.normal[batch_indices]
        thickness_batch = self.thickness[batch_indices]
        view_batch = self.view[batch_indices]
        sun_batch = self.sun[batch_indices]

        return (color_batch, pos_batch, vis_batch, normal_batch, thickness_batch, view_batch, sun_batch)

    def get_number_of_validation_batches(self):
        return self.validation_batches

    def get_validation_batch(self, idx):
        if idx >= self.validation_batches:
            raise IndexError(f"Validation batch index {idx} out of range.")

        start = idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.validation_indices[start:end]

        color_batch = self.color[batch_indices]
        pos_batch = self.pos[batch_indices]
        vis_batch = self.vis[batch_indices]
        normal_batch = self.normal[batch_indices]
        thickness_batch = self.thickness[batch_indices]
        view_batch = self.view[batch_indices]
        sun_batch = self.sun[batch_indices]

        return (color_batch, pos_batch, vis_batch, normal_batch, thickness_batch, view_batch, sun_batch)
            
    def shuffle(self):
        """Shuffle dataset order for the next training epoch."""
        self.indices = self.indices[torch.randperm(self.indices.shape[0], device=self.indices.device)]
