"""Auxiliary BEV occupancy head on LLM vision-token hidden states (training only).

The hidden states of the camera vision tokens at LLM layer `layer` are read out by a small BEV decoder:
learned 50x50 BEV queries cross-attend to the tokens and the result is upsampled to the 200x200 LiDAR
occupancy grid of tools/gen_bev_occ.py (ego frame, 0.5 m cells over [-50, 50) m, grid[i, j] with i along x
(forward) and j along y (left)). The GT has 3 labels (free / occupied / unknown from LiDAR ray casting); the
head predicts occupied-vs-free and unknown cells are ignored in the loss and the IoU.

The head gets direction-only geometry: keys carry a Fourier embedding of the token's ray direction and a
camera embedding, queries carry the polar position of their BEV cell. It is never told how far along the ray
a token is, so the range information has to come from the LLM hidden states.
"""
import math
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_bev_occ_targets(tokens, root, size, device):
    """Returns occupied (B, S, S) float {0, 1} and known (B, S, S) float {0, 1} (0 = unknown, ignored)."""
    n = size * size
    bits = np.stack([np.unpackbits(np.load(osp.join(root, t + '.npy')))[:2 * n] for t in tokens])
    bits = torch.from_numpy(bits).to(device=device, dtype=torch.float32).view(-1, 2, size, size)
    return bits[:, 0], bits[:, 1]


def fourier(x, num_freq):
    """x (..., d) roughly in [-1, 1] -> (..., d * (1 + 2 * num_freq))."""
    freqs = (2.0 ** torch.arange(num_freq, device=x.device, dtype=x.dtype)) * math.pi
    xf = x.unsqueeze(-1) * freqs
    return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, ffn_ratio=4):
        super().__init__()
        self.heads = heads
        self.nq, self.nk = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.nf = nn.LayerNorm(dim)
        self.ffn = mlp(dim, dim * ffn_ratio, dim)

    def forward(self, q, kv, k_pos):
        B, Nq, D = q.shape
        h = self.heads
        kvn = self.nk(kv)
        Q = self.q(self.nq(q)).view(B, Nq, h, D // h).transpose(1, 2)
        K = self.k(kvn + k_pos).view(B, -1, h, D // h).transpose(1, 2)
        V = self.v(kvn).view(B, -1, h, D // h).transpose(1, 2)
        a = F.scaled_dot_product_attention(Q, K, V).transpose(1, 2).reshape(B, Nq, D)
        q = q + self.o(a)
        return q + self.ffn(self.nf(q))


class BEVOccHead(nn.Module):
    def __init__(self,
                 in_dim,
                 layer=14,
                 embed_dims=256,
                 bev_size=50,
                 out_size=200,
                 pc_range=50.0,
                 num_layers=2,
                 num_heads=8,
                 num_cams=6,
                 num_freq=6,
                 loss_weight=0.5,
                 pos_weight=4.0,
                 dice_weight=1.0,
                 occ_root='data/nuscenes/bev_occ3',
                 detach_input=False):  # True: the occupancy loss trains only the head (no gradient into the LLM)
        super().__init__()
        D = embed_dims
        self.detach_input = detach_input
        self.layer, self.bev_size, self.out_size, self.pc_range = layer, bev_size, out_size, pc_range
        self.num_freq, self.loss_weight, self.dice_weight, self.occ_root = num_freq, loss_weight, dice_weight, occ_root
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)), persistent=False)

        self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, D))
        self.cam_embed = nn.Embedding(num_cams, D)
        self.ray_embed = mlp(3 * (1 + 2 * num_freq), D, D)
        self.bev_query = nn.Parameter(torch.randn(bev_size * bev_size, D) * 0.02)
        self.bev_pos = mlp(5 * (1 + 2 * num_freq), D, D)
        self.blocks = nn.ModuleList(CrossBlock(D, num_heads) for _ in range(num_layers))
        up = []
        c, s = D, bev_size
        while s < out_size:
            up += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                   nn.Conv2d(c, c // 2, 3, padding=1), nn.GroupNorm(8, c // 2), nn.GELU()]
            c, s = c // 2, s * 2
        assert s == out_size, 'out_size must be bev_size * 2^k'
        self.decoder = nn.Sequential(nn.Conv2d(D, D, 3, padding=1), nn.GroupNorm(8, D), nn.GELU(), *up,
                                     nn.Conv2d(c, 1, 1))

        # polar + cartesian position of each BEV query cell (ego frame, cell centres)
        cs = 2 * pc_range / bev_size
        x = -pc_range + (torch.arange(bev_size) + 0.5) * cs
        X, Y = torch.meshgrid(x, x, indexing='ij')  # [i, j]: i along x (forward), j along y (left)
        r = torch.sqrt(X ** 2 + Y ** 2)
        az = torch.atan2(Y, X)
        cell = torch.stack([X / pc_range, Y / pc_range, r / (pc_range * math.sqrt(2)), az.cos(), az.sin()], -1)
        self.register_buffer('cell_feat', cell.view(-1, 5), persistent=False)

    def forward(self, tokens, ray_dirs):
        """tokens (B, N*h*w, C) LLM hidden states of the vision tokens (camera-major, raster order);
        ray_dirs (B, N, h, w, 3) unit ego-frame ray directions. Returns occupancy logits (B, S, S)."""
        B = tokens.shape[0]
        N = ray_dirs.shape[1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            kv = self.in_proj(tokens.float())
            cam = self.cam_embed.weight[:N, None].expand(N, ray_dirs.shape[2] * ray_dirs.shape[3], -1).reshape(1, -1, kv.shape[-1])
            k_pos = self.ray_embed(fourier(ray_dirs.reshape(B, -1, 3), self.num_freq)) + cam
            q = (self.bev_query + self.bev_pos(fourier(self.cell_feat, self.num_freq))).unsqueeze(0).expand(B, -1, -1)
            for blk in self.blocks:
                q = blk(q, kv, k_pos)
            S = self.bev_size
            logits = self.decoder(q.transpose(1, 2).reshape(B, -1, S, S))
        return logits[:, 0].float()

    def loss(self, logits, target, known):
        """BCE (pos_weight) + dice over the known (free / occupied) cells only."""
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction='none')
        bce = (bce * known).sum() / known.sum().clamp(min=1)
        p = logits.sigmoid() * known
        inter = (p * target).flatten(1).sum(-1)
        dice = 1 - ((2 * inter + 1) / (p.flatten(1).sum(-1) + target.flatten(1).sum(-1) + 1)).mean()
        with torch.no_grad():
            k = known > 0.5
            pred = (logits > 0) & k
            t = target > 0.5
            iou = (pred & t).sum().float() / (pred | t).sum().clamp(min=1).float()
        return dict(loss_occ=self.loss_weight * (bce + self.dice_weight * dice), occ_iou=iou)
