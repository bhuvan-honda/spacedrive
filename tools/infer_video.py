#!/usr/bin/env python3
# -------------------------------------------------------------------------
# SpaceDrive — Video inference over NuScenes samples directory
# -------------------------------------------------------------------------
# Usage:
#   python tools/infer_video.py \
#       --samples-dir data/nuscenes/samples \
#       --ckpt spacedrive_qwen/iter_21096.pth \
#       [--camera-params tools/camera_params.yaml] \
#       [--command-desc "Go straight."] \
#       [--output-dir ./infer_video_output] \
#       [--max-frames 100] \
#       [--fps 2]
#
# Iterates CAM_FRONT images (sorted), loads matching frames from other
# camera folders by sorted index, runs inference, and writes an MP4 video
# of annotated front-camera frames.
# -------------------------------------------------------------------------

import os
import sys
import argparse
import copy
import json
import re
import warnings

import cv2
import numpy as np
from nuscenes.eval.common.utils import Quaternion

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

# ── Project root ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
import mmdet
import mmdet3d
from mmdet3d.models import build_model

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'projects'))
import projects.mmdet3d_plugin  # noqa: F401

# ── Reuse helpers from infer_single_image ─────────────────────────────────
from infer_single_image import (
    CAM_NAMES,
    DEFAULT_IMAGE_TOKEN,
    load_camera_params,
    resize_image_and_intrinsic,
    build_tokenised_input,
    parse_waypoints,
    project_bev_to_image,
    draw_waypoints_on_image,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="SpaceDrive video inference — NuScenes samples → annotated MP4"
    )
    p.add_argument(
        "--samples-dir", type=str, required=True,
        help="Path to NuScenes samples/ directory containing CAM_FRONT/, CAM_BACK/, etc.",
    )
    p.add_argument(
        "--ckpt", type=str, default="spacedrive_qwen/iter_21096.pth",
    )
    p.add_argument(
        "--config", type=str,
        default="projects/configs/spacedrive/spacedrive_qwen.py",
    )
    p.add_argument(
        "--camera-params", type=str, default="tools/camera_params.yaml",
    )
    p.add_argument(
        "--command-desc", type=str, default="Move forward.",
    )
    p.add_argument(
        "--output-dir", type=str, default="./infer_video_output",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
    )
    p.add_argument(
        "--target-size", nargs=2, type=int, default=[1600, 900],
        metavar=("W", "H"),
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Process at most N frames (default: all).",
    )
    p.add_argument(
        "--start-frame", type=int, default=0,
        help="Index of first CAM_FRONT frame to process (sorted order).",
    )
    p.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for the output video.",
    )
    return p.parse_args()


def build_frame_lists(samples_dir):
    """Build sorted file lists for each camera, indexed by position.

    Returns:
        frame_paths: list[dict] — one dict per frame-index, mapping
                     camera name → absolute image path.
    """
    cam_files = {}
    for cam in CAM_NAMES:
        cam_dir = os.path.join(samples_dir, cam)
        if not os.path.isdir(cam_dir):
            raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
        files = sorted(os.listdir(cam_dir))
        cam_files[cam] = [os.path.join(cam_dir, f) for f in files]

    n_frames = len(cam_files[CAM_NAMES[0]])
    for cam in CAM_NAMES:
        if len(cam_files[cam]) != n_frames:
            warnings.warn(
                f"{cam} has {len(cam_files[cam])} files vs "
                f"{CAM_NAMES[0]} has {n_frames}. Using min."
            )
    n_frames = min(len(cam_files[cam]) for cam in CAM_NAMES)

    frame_paths = []
    for i in range(n_frames):
        frame_paths.append({cam: cam_files[cam][i] for cam in CAM_NAMES})
    return frame_paths


def run_single_frame(model, processor, tokenizer, frame_cam_paths,
                     intrinsics_orig, extrinsics, target_w, target_h,
                     command_desc, device):
    """Run inference on one set of 6 camera images. Returns (annotated_pil, result_dict)."""

    # Load & resize images
    images_bgr = []
    intrinsics = []
    for cam_name in CAM_NAMES:
        path = frame_cam_paths[cam_name]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img_resized, K_scaled = resize_image_and_intrinsic(
            img_bgr, intrinsics_orig[cam_name], target_size=(target_w, target_h)
        )
        images_bgr.append(img_resized)
        intrinsics.append(K_scaled)

    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    # Tokenise
    extrinsics_list = [extrinsics[cam] for cam in CAM_NAMES]
    batch = build_tokenised_input(
        images_rgb, intrinsics, extrinsics_list,
        processor, tokenizer,
        command_desc=command_desc,
    )

    # Build tensors
    img_tensor = torch.from_numpy(
        np.stack([i.astype(np.float32).transpose(2, 0, 1) for i in images_rgb], axis=0)
    ).unsqueeze(0).to(device)

    intrinsics_t = torch.from_numpy(
        np.stack(intrinsics, axis=0).astype(np.float32)
    ).unsqueeze(0).to(device)

    extrinsics_t = torch.from_numpy(
        np.stack(extrinsics_list, axis=0).astype(np.float32)
    ).unsqueeze(0).to(device)

    lidar2img_t = intrinsics_t @ extrinsics_t

    for k in batch:
        batch[k] = batch[k].to(device)

    B = 1
    extra_data = dict(
        intrinsics    = intrinsics_t,
        extrinsics    = extrinsics_t,
        lidar2img     = lidar2img_t,
        timestamp     = torch.zeros(B, dtype=torch.float32, device=device),
        img_timestamp = torch.zeros(B, 6, dtype=torch.float32, device=device),
        ego_pose      = torch.eye(4, device=device).unsqueeze(0),
        ego_pose_inv  = torch.eye(4, device=device).unsqueeze(0),
        command       = torch.zeros(B, dtype=torch.long, device=device),
        can_bus       = torch.zeros(B, 13, dtype=torch.float32, device=device),
        **batch,
    )

    with torch.no_grad():
        results = model(
            return_loss=False,
            img=img_tensor,
            img_metas=[{
                "sample_idx": os.path.basename(frame_cam_paths["CAM_FRONT"]),
                "filename": [frame_cam_paths[c] for c in CAM_NAMES],
            }],
            rescale=True,
            **extra_data,
        )

    # Parse answer
    raw_answer = ""
    if isinstance(results, list) and len(results) > 0:
        r = results[0]
        if isinstance(r, dict):
            raw_answer = r.get("A", "")
            if isinstance(raw_answer, list):
                raw_answer = raw_answer[0] if raw_answer else ""
        elif isinstance(r, str):
            raw_answer = r

    waypoints_xy = parse_waypoints(raw_answer)

    # Project & annotate on front camera
    front_bgr = images_bgr[0]
    front_K = intrinsics[0]
    front_E = extrinsics_list[0]

    if waypoints_xy:
        uv_list, valid = project_bev_to_image(waypoints_xy, front_K, front_E)
        annotated_pil = draw_waypoints_on_image(front_bgr, uv_list, valid, waypoints_xy)
    else:
        annotated_pil = Image.fromarray(cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB))

    return annotated_pil, {"answer": raw_answer, "waypoints_xy_bev": waypoints_xy}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Camera params (same for all frames) ────────────────────────────
    cam_yaml = os.path.join(PROJECT_ROOT, args.camera_params) \
        if not os.path.isabs(args.camera_params) else args.camera_params
    intrinsics_list, extrinsics_list = load_camera_params(cam_yaml)
    # Map by camera name for convenience
    intrinsics_orig = {cam: K for cam, K in zip(CAM_NAMES, intrinsics_list)}
    extrinsics = {cam: E for cam, E in zip(CAM_NAMES, extrinsics_list)}

    target_w, target_h = args.target_size

    # ── Build frame list ───────────────────────────────────────────────
    samples_dir = os.path.join(PROJECT_ROOT, args.samples_dir) \
        if not os.path.isabs(args.samples_dir) else args.samples_dir
    frame_paths = build_frame_lists(samples_dir)
    total = len(frame_paths)
    start = args.start_frame
    end = total if args.max_frames is None else min(start + args.max_frames, total)
    frame_paths = frame_paths[start:end]
    print(f"[INFO] Processing frames {start}..{end-1} out of {total} total")

    # ── Build model ────────────────────────────────────────────────────
    config_path = os.path.join(PROJECT_ROOT, args.config) \
        if not os.path.isabs(args.config) else args.config
    cfg = Config.fromfile(config_path)
    # Model internally writes JSON to save_path + sample_idx;
    # put those in a subdirectory so they don't pollute the output.
    model_json_dir = os.path.join(args.output_dir, "_model_json/")
    os.makedirs(model_json_dir, exist_ok=True)
    cfg.model.save_path = model_json_dir

    print("[INFO] Building model …")
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    model.eval()

    ckpt_path = os.path.join(PROJECT_ROOT, args.ckpt) \
        if not os.path.isabs(args.ckpt) else args.ckpt
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)
    model = model.to(device)

    processor = model.processor
    tokenizer = model.tokenizer

    # ── Frames directory for saved PNGs ─────────────────────────────
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    all_results = []
    saved_frame_paths = []

    # ── Inference loop ─────────────────────────────────────────────────
    for idx, frame_cam_paths in enumerate(frame_paths):
        frame_name = os.path.basename(frame_cam_paths["CAM_FRONT"])
        print(f"[{idx+1}/{len(frame_paths)}] {frame_name}")

        annotated_pil, result = run_single_frame(
            model, processor, tokenizer, frame_cam_paths,
            intrinsics_orig, extrinsics,
            target_w, target_h, args.command_desc, device,
        )
        result["frame"] = frame_name
        all_results.append(result)

        # Save annotated frame as PNG
        frame_png = os.path.join(frames_dir, f"{idx:06d}.png")
        annotated_pil.save(frame_png)
        saved_frame_paths.append(frame_png)

    # ── Assemble video from saved PNGs using ffmpeg ──────────────────
    video_path = os.path.join(args.output_dir, "waypoints_video.mp4")
    if saved_frame_paths:
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(frames_dir, "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path,
        ]
        print(f"[INFO] Running: {' '.join(ffmpeg_cmd)}")
        ret = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if ret.returncode == 0:
            print(f"[INFO] Video saved: {video_path}")
        else:
            print(f"[WARN] ffmpeg failed (rc={ret.returncode}): {ret.stderr[:500]}")
            print("[INFO] Frames are still available in:", frames_dir)

    # Save all results JSON
    json_path = os.path.join(args.output_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Results JSON saved: {json_path}")
    print(f"[INFO] Annotated frames saved in: {frames_dir}/")


if __name__ == "__main__":
    main()
