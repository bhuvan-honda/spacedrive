"""Plücker ray embedding for the camera vision tokens (added to the depth-based 3D PE before the LLM).

Each vision token gets the Plücker coordinates (d, c x d) of the ray through its pixel centre (ego frame):
d = unit direction, c = camera centre, so the 6-D vector identifies the viewing ray independently of the
predicted depth (the depth-based 3D PE encodes c + depth * d and inherits the depth errors). The coordinates
are Fourier-encoded and mapped by an MLP whose output layer is zero-initialised, so the model is identical to
the one without it at iteration 0.
"""
import math

import torch
import torch.nn as nn

from .geometry import token_rays


class PluckerRayPE(nn.Module):
    def __init__(self, dim, num_freqs=6, hidden=512, moment_scale=2.0):
        super().__init__()
        self.moment_scale = moment_scale
        self.register_buffer('freqs', math.pi * 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False)
        in_dim = 6 * (1 + 2 * num_freqs)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, lidar2img, h, w, H, W):
        """lidar2img (B, N, 4, 4) ego->pixel of the (H, W) images; (h, w) vision-token grid.
        Returns (B, N*h*w, dim) float32 in vision-token order (camera-major, raster)."""
        B, N = lidar2img.shape[:2]
        with torch.no_grad():
            dirs, centres = token_rays(lidar2img.float(), h, w, H, W)  # (B, N, h, w, 3), (B, N, 3)
            moment = torch.cross(centres[:, :, None, None].expand_as(dirs), dirs, dim=-1) / self.moment_scale
            x = torch.cat([dirs, moment], -1).reshape(B, N * h * w, 6)
            xf = x[..., None] * self.freqs  # (B, P, 6, F)
            feat = torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)
        return self.mlp(feat).float()
