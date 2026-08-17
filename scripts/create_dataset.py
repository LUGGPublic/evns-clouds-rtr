import numpy as np
import OpenEXR as exr
import glob
import torch

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


class ThinCloudDataset(Dataset):
    def __init__(self, folder):
        super().__init__()

        # Read files
        print("Reading files of dataset")

        color = read_dataset_files(folder + "/color*.exr", "RGBA")
        pos = read_dataset_files(folder + "/position*.exr", "XYZ")
        vis = read_dataset_files(folder + "/visibility*.exr", "RGB")
        back = read_dataset_files(folder + "/back*.exr", "XYZ")
        normal = read_dataset_files(folder + "/normal*.exr", "XYZ")
        labels = np.genfromtxt(folder + "/labels.csv", delimiter=",", skip_header=1)
        bounds = np.genfromtxt(folder + "/bounds.csv", delimiter=",", skip_header=1)

        self.resolution = color[0].shape[0:2]

        # Extract bounds
        min_bound = np.array(bounds[0:3])
        max_bound = np.array(bounds[3:6])
        self.min_bound = min_bound
        self.max_bound = max_bound

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

        # Convert to tensors
        self.color = torch.tensor(np.array(color), dtype=torch.float32)
        self.pos = torch.tensor(np.array(pos), dtype=torch.float32)
        self.vis = torch.tensor(np.array(vis), dtype=torch.float32)
        self.normal = torch.tensor(np.array(normal), dtype=torch.float32)
        self.thickness = torch.tensor(np.array(thickness), dtype=torch.float32)
        self.view = torch.tensor(view, dtype=torch.float32)
        self.sun = torch.tensor(sun, dtype=torch.float32)

        # Number of pixels in each image
        num_pixels = self.color.shape[1] * self.color.shape[2]

        # Flatten
        self.color = self.color.view(-1, 4)
        self.pos = self.pos.view(-1, 3)
        self.vis = self.vis.view(-1, 1)
        self.normal = self.normal.view(-1, 3)
        self.thickness = self.thickness.view(-1, 1)
        self.view = self.view.repeat_interleave(num_pixels, dim=0)
        self.sun = self.sun.repeat_interleave(num_pixels, dim=0)

        evaluate = True
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

        print(f"Dataset size: {self.color.shape[0]} pixels")

    def __len__(self):
        return self.color.shape[0]

    def __getitem__(self, idx):
        return (
            self.color[idx],
            self.pos[idx],
            self.vis[idx],
            self.normal[idx],
            self.thickness[idx],
            self.view[idx],
            self.sun[idx],
        )


folders = [
    # "PATH TO FOLDER GENERATED FROM BLENDER",
]

datasets = [ThinCloudDataset(folder) for folder in folders]

globalDataset = {
    "color": torch.cat([dataset.color for dataset in datasets]),
    "pos": torch.cat([dataset.pos for dataset in datasets]),
    "vis": torch.cat([dataset.vis for dataset in datasets]),
    "normal": torch.cat([dataset.normal for dataset in datasets]),
    "thickness": torch.cat([dataset.thickness for dataset in datasets]),
    "view": torch.cat([dataset.view for dataset in datasets]),
    "sun": torch.cat([dataset.sun for dataset in datasets]),
    "resolution" : torch.tensor(datasets[0].resolution),
    "min_bound" : torch.tensor(datasets[0].min_bound),
    "max_bound" : torch.tensor(datasets[0].max_bound),
}

torch.save(globalDataset, "dataset.pt")
