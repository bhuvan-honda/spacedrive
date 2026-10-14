"""Depth-guided epipolar cross-view adapter for the camera vision tokens (before they enter the LLM).

For a token of camera n with UniDepth depth d, the 3D points at depths d * s (s in `scales`, a short segment
of its epipolar line around the predicted depth) are projected into every other camera m. Keys / values are
bilinearly sampled from camera m's token map at those pixels. Each sample's attention logit gets a learnable
penalty  -alpha * |log z_m - log D_m(u, v)|, where z_m is the hypothesis depth seen from camera m and D_m is
camera m's own UniDepth depth at that pixel: both views agreeing that the surface is there makes a sample
attractive. Samples behind camera m or outside its image are masked; a learnable null key lets tokens without
cross-view evidence attend to nothing. The result is added back through a zero-initialised projection, so the
adapter starts as an identity.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import project, token_pixel_centers, unproject


class DepthEpipolarAdapter(nn.Module):
    def __init__(self, dim, inner=256, heads=8, scales=(0.8, 0.9, 1.0, 1.1, 1.25), init_alpha=2.0, min_depth=0.5):
        super().__init__()
        assert inner % heads == 0
        self.inner, self.heads, self.min_depth = inner, heads, min_depth
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32), persistent=False)
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, inner)
        self.kv = nn.Linear(dim, 2 * inner)
        self.hyp_embed = nn.Parameter(torch.zeros(len(scales), inner))
        self.null_k = nn.Parameter(torch.zeros(inner))
        self.null_v = nn.Parameter(torch.zeros(inner))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.out = nn.Linear(inner, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.last_valid_frac = None

    @torch.no_grad()
    def _geometry(self, depth_tok, lidar2img, H, W):
        B, N, h, w = depth_tok.shape
        img2lidar = lidar2img.double().inverse().float()
        u, v = token_pixel_centers(h, w, H, W, depth_tok.device)
        d = depth_tok.float().clamp(min=self.min_depth)[..., None] * self.scales  # (B, N, h, w, S)
        pts = unproject(img2lidar[:, :, None, None, None], u[..., None], v[..., None], d)  # (B, N, h, w, S, 3)
        pu, pv, pz = project(lidar2img.float()[:, None, None, None, None], pts[..., None, :])  # (B, N, h, w, S, M)
        M = lidar2img.shape[1]
        other = ~torch.eye(N, M, dtype=torch.bool, device=pu.device)[None, :, None, None, None, :]
        valid = (pz > self.min_depth) & (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H) & other
        grid = torch.stack([pu / W * 2 - 1, pv / H * 2 - 1], -1)  # (B, N, h, w, S, M, 2)
        return grid, pz, valid

    def forward(self, x, depth_tok, lidar2img, img_hw):
        """x (B*N*h*w, C) vision tokens (camera-major, raster); depth_tok (B, N, h, w) min-pooled metric depth;
        lidar2img (B, N, 4, 4) ego->pixel of the (H, W) = img_hw images. Returns x + adapter(x)."""
        B, N, h, w = depth_tok.shape
        H, W = img_hw
        I, nh = self.inner, self.heads
        S = self.scales.numel()
        P = h * w

        grid, pz, valid = self._geometry(depth_tok, lidar2img, H, W)
        M = grid.shape[-2]

        with torch.autocast('cuda', dtype=torch.bfloat16):
            xn = self.norm(x.view(B, N, P, -1))
            q = self.q(xn).float()                                   # (B, N, P, I)
            kv = self.kv(xn).float()                                 # (B, N, P, 2I)

        # sample every target camera m at the projections of all source tokens / hypotheses
        kv_maps = kv.view(B, N, h, w, 2 * I).permute(0, 1, 4, 2, 3).reshape(B * N, 2 * I, h, w)
        g = grid.permute(0, 5, 1, 2, 3, 4, 6).reshape(B * M, 1, N * P * S, 2)
        samp = F.grid_sample(kv_maps, g, mode='bilinear', padding_mode='zeros', align_corners=False)
        samp = samp.view(B, M, 2 * I, N, P, S).permute(0, 3, 4, 5, 1, 2).reshape(B, N, P, S * M, 2 * I)
        d_maps = depth_tok.float().reshape(B * N, 1, h, w)
        d_tgt = F.grid_sample(d_maps, g, mode='nearest', padding_mode='border', align_corners=False)
        d_tgt = d_tgt.view(B, M, N, P, S).permute(0, 2, 3, 4, 1)    # (B, N, P, S, M)

        k, val = samp.split(I, dim=-1)
        k = (k.view(B, N, P, S, M, I) + self.hyp_embed[:, None]).view(B, N, P, S * M, nh, I // nh)
        val = val.reshape(B, N, P, S * M, nh, I // nh)
        qh = q.view(B, N, P, nh, I // nh)
        scale = 1.0 / math.sqrt(I // nh)

        logits = torch.einsum('bnphd,bnpkhd->bnphk', qh, k) * scale
        vmask = valid.reshape(B, N, P, S * M)
        geo = (pz.reshape(B, N, P, S, M).clamp(min=self.min_depth).log() - d_tgt.clamp(min=self.min_depth).log()).abs()
        bias = (-self.log_alpha.exp() * geo.reshape(B, N, P, S * M)).masked_fill(~vmask, float('-inf'))
        logits = logits + bias[:, :, :, None, :]
        null = (qh * self.null_k.view(nh, -1)).sum(-1, keepdim=True) * scale  # (B, N, P, nh, 1)
        attn = torch.cat([logits, null], -1).softmax(-1)
        out = torch.einsum('bnphk,bnpkhd->bnphd', attn[..., :-1], val) + attn[..., -1:] * self.null_v.view(nh, -1)
        out = self.out(out.reshape(B, N, P, I)).reshape(B * N * P, -1)

        self.last_valid_frac = vmask.any(-1).float().mean().detach()
        return x + out.to(x.dtype)
