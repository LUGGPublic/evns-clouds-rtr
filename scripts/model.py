import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import OrderedDict

import struct


THIN_CLOUD_MODEL_VERSION = 1


class Exp3(torch.nn.Module):
    def __init__(self):
        super(Exp3, self).__init__()

    @torch.compile
    def forward(self, x):
        return torch.exp(x - 3.0)


class TriplaneEncoding(nn.Module):
    """
    Triplane encoding:
      - three planes: XY, XZ, YZ
      - each plane stores features as a tensor (feature_dim, H, W)
    Args:
      resolution: int or (H,W) tuple for plane resolution (assumed same for all planes if int)
      feature_dim: int, number of channels per plane (final output will be either feature_dim (mode='sum')
                   or 3*feature_dim (mode='concat'))
      mode: 'sum' or 'concat'  (how to combine features from 3 planes)
      align_corners: bool passed to grid_sample (controls coord mapping)
    Input coords:
      positions: tensor (N,3), each coordinate assumed normalized in [0,1].
                 If your world coords differ, normalize to [0,1] first.
    Returns:
      enc: (N, feature_dim) if mode='sum', else (N, 3*feature_dim) for 'concat'.
    """
    def __init__(self, resolution, feature_dim, mode='sum', align_corners=True):
        super().__init__()
        if isinstance(resolution, int):
            H = W = resolution
        else:
            H, W = resolution

        self.H = H
        self.W = W
        self.feature_dim = feature_dim
        assert mode in ('sum', 'concat')
        self.mode = mode
        self.align_corners = align_corners

        # store planes as parameters shaped (1, C, H, W) for easy grid_sample
        self.plane_xy = nn.Parameter(torch.randn(1, feature_dim, H, W) * 0.01)
        self.plane_xz = nn.Parameter(torch.randn(1, feature_dim, H, W) * 0.01)
        self.plane_yz = nn.Parameter(torch.randn(1, feature_dim, H, W) * 0.01)

    def _get_mip_plane(self, plane, mip_level: int):
        if mip_level < 0:
            raise ValueError(f"mip_level must be >= 0, got {mip_level}")

        if mip_level == 0:
            return plane

        # Each mip level halves the resolution, clamped to at least 1x1.
        h = max(1, self.H // (2 ** mip_level))
        w = max(1, self.W // (2 ** mip_level))

        return F.interpolate(
            plane,
            size=(h, w),
            mode='bilinear',   # linear interpolation for 2D planes
            align_corners=self.align_corners,
        )

    @torch.compile
    def forward(self, positions, mip_level: int = 0):
        """
        positions: (N,3) with coordinates (x,y,z) each in [0,1].
        """
        assert positions.dim() == 2 and positions.shape[1] == 3
        N = positions.shape[0]
        pos = positions.clamp(0.0, 1.0)

        # Select mip level for each plane
        plane_xy = self._get_mip_plane(self.plane_xy, mip_level)
        plane_xz = self._get_mip_plane(self.plane_xz, mip_level)
        plane_yz = self._get_mip_plane(self.plane_yz, mip_level)

        # Build grids for each plane. grid shape: (1, N, 1, 2) => output (1, C, N, 1)
        x = (pos[:, 0] * 2.0 - 1.0).view(1, N, 1, 1)  # normalized to [-1,1]
        y = (pos[:, 1] * 2.0 - 1.0).view(1, N, 1, 1)
        z = (pos[:, 2] * 2.0 - 1.0).view(1, N, 1, 1)

        # grid layout is (x, y)
        grid_xy = torch.cat([x, y], dim=-1)  # (1, N, 1, 2)
        grid_xz = torch.cat([x, z], dim=-1)
        grid_yz = torch.cat([y, z], dim=-1)

        # sample each plane. input is (1, C, H, W), grid is (1, N, 1, 2) => out (1, C, N, 1)
        s_xy = F.grid_sample(plane_xy, grid_xy, mode='bilinear',
                                padding_mode='border', align_corners=self.align_corners)
        s_xz = F.grid_sample(plane_xz, grid_xz, mode='bilinear',
                                padding_mode='border', align_corners=self.align_corners)
        s_yz = F.grid_sample(plane_yz, grid_yz, mode='bilinear',
                                padding_mode='border', align_corners=self.align_corners)

        # squeeze to (N, C)
        s_xy = s_xy.view(self.feature_dim, N).transpose(0, 1).contiguous()
        s_xz = s_xz.view(self.feature_dim, N).transpose(0, 1).contiguous()
        s_yz = s_yz.view(self.feature_dim, N).transpose(0, 1).contiguous()

        if self.mode == 'sum':
            return s_xy + s_xz + s_yz
        else:  # concat
            return torch.cat([s_xy, s_xz, s_yz], dim=1)


class ThinCloudModel(nn.Module):
    def __init__(self, config):
        super(ThinCloudModel, self).__init__()

        # Triplane encoding
        triplane_features = int(config['model']['triplane_features'])
        triplane_resolution = int(config['model']['triplane_resolution'])
        self.triplanes = TriplaneEncoding(triplane_resolution, triplane_features)

        # MLP Layers
        layers = OrderedDict()
        width = int(config['model']['layer_width'])

        # Inputs
        self.use_thickness = 1 if config["model"]["use_thickness"].lower() in ("true", "yes", "1") else 0
        view_dim = 0 #3
        sun_dim = 2
        inputs = triplane_features + view_dim + sun_dim + self.use_thickness

        layers['linear1'] = torch.nn.Linear(inputs, width)
        layers['relu1'] = torch.nn.ReLU()

        # Hidden layers
        i = 2
        while i < int(config['model']['hidden_layers']):
            layers[f'linear{i}'] = torch.nn.Linear(width, width)
            layers[f'relu{i}'] = torch.nn.ReLU()
            i += 1

        # Outputs
        output_dim = 7  # RGB vis (3) + RGB non vis (3) + alpha (1)
        layers[f'linear{i}'] = torch.nn.Linear(width, output_dim)
        layers[f'exp3{i}'] = Exp3()

        self.network = torch.nn.Sequential(layers)

    @torch.compile
    def forward(self, x, mip_level: int = 0):
        pos, view, sun, thickness = x

        enc_pos = self.triplanes(pos, mip_level)
        if self.use_thickness:
            #x = torch.cat([enc_pos, view, sun, thickness], dim=-1)
            x = torch.cat([enc_pos, sun, thickness], dim=-1)
        else:
            #x = torch.cat([enc_pos, view, sun], dim=-1)
            x = torch.cat([enc_pos, sun], dim=-1)

        return self.network(x)

    def save(self, filename):
        content = bytearray()

        # Write model version
        content.extend(struct.pack("I", THIN_CLOUD_MODEL_VERSION))

        # Write bounding box
        #content.extend(struct.pack("fff"), self.bounding_box.x, self.bounding_box.y, self.bounding_box.z)

        # Write input encoding
        content.extend(struct.pack("II", self.triplanes.H, self.triplanes.feature_dim))
        plane_xy = self.triplanes.plane_xy.data.to("cpu").numpy()
        plane_xz = self.triplanes.plane_xz.data.to("cpu").numpy()
        plane_yz = self.triplanes.plane_yz.data.to("cpu").numpy()

        # (1, 8, 512, 512) to (512, 512, 8)
        plane_xy = plane_xy[0].transpose(1,2,0)
        plane_xz = plane_xz[0].transpose(1,2,0)
        plane_yz = plane_yz[0].transpose(1,2,0)

        content.extend(plane_xy.astype("float32").tobytes())
        content.extend(plane_xz.astype("float32").tobytes())
        content.extend(plane_yz.astype("float32").tobytes())

        # Write layer count
        n = sum(1 for name, _ in self.named_parameters() if "linear" in name) // 2
        content.extend(struct.pack("I", n))

        # Write each layer
        for name, param in self.named_parameters():
            if "triplane" in name:
                continue

            # Write param dimension
            for dim in param.size():
                content.extend(struct.pack("I", dim))

            # Write params
            array = param.data.to("cpu").numpy()
            content.extend(array.astype("float32").tobytes())

        with open(filename, "wb") as file:
            file.write(content)