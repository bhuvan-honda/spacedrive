"""Small geometry helpers shared by the BEV occupancy head and the depth-guided epipolar adapter.

Conventions (SpaceDrive): `lidar2img` (B, N, 4, 4) maps the EGO frame (x forward, y left, z up; the dataset
builds it from the camera sensor2ego) to pixels of the resized (H, W) image. Vision tokens of each camera form
an (h, w) grid in raster order after the 2x2 merger; token (i, j) covers pixels [i*H/h, (i+1)*H/h) x
[j*W/w, (j+1)*W/w).
"""
import torch


def token_pixel_centers(h, w, H, W, device, dtype=torch.float32):
    v = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (H / h)
    u = (torch.arange(w, device=device, dtype=dtype) + 0.5) * (W / w)
    return torch.meshgrid(u, v, indexing='xy')  # each (h, w): u varies along w, v along h


def unproject(img2lidar, u, v, d):
    """img2lidar (..., 4, 4); u, v, d broadcastable to (...,) -> ego points (..., 3)."""
    u, v, d = torch.broadcast_tensors(u, v, d)
    p = torch.stack([u * d, v * d, d, torch.ones_like(d)], -1)
    return (img2lidar @ p.unsqueeze(-1))[..., :3, 0]


def project(lidar2img, pts, eps=1e-5):
    """lidar2img (..., 4, 4); pts (..., 3) -> (u, v, z), each (...,)."""
    p = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)
    q = (lidar2img @ p.unsqueeze(-1))[..., 0]
    z = q[..., 2]
    zc = z.clamp(min=eps)
    return q[..., 0] / zc, q[..., 1] / zc, z


def token_rays(lidar2img, h, w, H, W):
    """Unit ray direction (ego frame) through every token centre and the camera centres.
    Returns dirs (B, N, h, w, 3), centres (B, N, 3)."""
    img2lidar = lidar2img.double().inverse()
    u, v = token_pixel_centers(h, w, H, W, lidar2img.device, torch.float64)
    m = img2lidar[:, :, None, None]
    p1 = unproject(m, u, v, torch.ones_like(u))
    p2 = unproject(m, u, v, 2 * torch.ones_like(u))
    dirs = torch.nn.functional.normalize(p2 - p1, dim=-1)
    centres = img2lidar[..., :3, 3]  # camera centre = img2lidar applied to the origin with d = 0
    return dirs.float(), centres.float()


def vis_token_mask(input_ids, image_token_id, n_vis):
    """(B, L) bool mask of the first `n_vis` image-placeholder positions of every sample (the camera vision
    tokens; extra placeholders inserted later hold the ego / depth-concat features)."""
    is_img = input_ids == image_token_id
    rank = is_img.long().cumsum(-1)
    return is_img & (rank <= n_vis)
