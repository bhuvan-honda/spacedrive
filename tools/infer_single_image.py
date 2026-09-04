#!/usr/bin/env python3
# -------------------------------------------------------------------------
# SpaceDrive — Standalone single-image inference script
# -------------------------------------------------------------------------
# Usage:
#   python tools/infer_single_image.py  \
#       --image path/to/image.jpg       \
#       --ckpt  spacedrive_qwen/iter_21096.pth \
#       [--camera-params tools/camera_params.yaml] \
#       [--command-desc "Turn right."]  \
#       [--output-dir ./infer_output]   \
#       [--device cuda:0]
#
# The script:
#   1. Loads the pre-trained SpaceDrive (non-plus) weights
#   2. Replicates the single input image across all 6 camera slots
#      (you can supply separate per-camera images via --images, see below)
#   3. Runs depth prediction + 3D PE + autoregressive generation
#   4. Parses the 6 predicted (x, y) BEV waypoints from the generated text
#   5. Projects them back into the image using the front-camera intrinsic
#   6. Saves an annotated image with the waypoints overlaid
#
# Optional multi-image mode:
#   --images front.jpg front_right.jpg front_left.jpg \
#            back.jpg  back_left.jpg   back_right.jpg
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

# ── Make sure the project root is on the Python path ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ── mmdet3d plugin registration ────────────────────────────────────────────
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint

import mmdet
import mmdet3d
from mmdet3d.models import build_model

# Register the SpaceDrive plugin
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'projects'))
import projects.mmdet3d_plugin  # noqa: F401  triggers __init__ auto-registration

# ── Constants (must match constants.py) ────────────────────────────────────
IGNORE_INDEX               = -100
IMAGE_TOKEN_INDEX          = 151655
VISION_START_TOKEN_INDEX   = 151653
VISION_END_TOKEN_INDEX     = 151653
POS_INDICATOR_TOKEN        = "<POS_INDICATOR>"
POS_EMBEDDING_TOKEN        = "<POS_EMBEDDING>"
POS_INDICATOR_TOKEN_INDEX  = 151665
POS_EMBEDDING_TOKEN_INDEX  = 151666
DEFAULT_IMAGE_TOKEN        = "<image>"

# NuScenes camera order expected by the model
CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# ── Helpers ────────────────────────────────────────────────────────────────
# These reproduce exactly what nuscenes_dataset.py does to compute
# intrinsics (viewpad 4×4) and extrinsics (lidar2cam 4×4) from the raw
# sensor2ego_rotation / sensor2ego_translation / camera_intrinsic values
# stored in NuScenes calibrated_sensor.json.

def _convert_egopose_to_matrix(rotation, translation):
    """Build 4×4 transform from 3×3 rotation matrix + translation."""
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3,  3] = translation
    mat[3, 3] = 1.0
    return mat


def _invert_matrix_egopose(egopose):
    """Invert a 4×4 rigid-body transform."""
    inv = np.zeros((4, 4), dtype=np.float64)
    R = egopose[:3, :3]
    t = egopose[:3,  3]
    inv[:3, :3] = R.T
    inv[:3,  3] = -R.T @ t
    inv[3, 3] = 1.0
    return inv


def load_camera_params(yaml_path: str):
    """Load camera params from YAML and compute intrinsic / extrinsic matrices.

    The YAML stores raw NuScenes calibration values:
      sensor2ego_rotation    : [w, x, y, z]   (quaternion, cam→ego)
      sensor2ego_translation : [x, y, z]       (metres, cam→ego)
      camera_intrinsic       : 3×3 matrix

    This function reproduces exactly what nuscenes_dataset.py does:
      cam2lidar_r   = Quaternion(sensor2ego_rotation).rotation_matrix
      cam2lidar_rt  = build_matrix(cam2lidar_r, sensor2ego_translation)
      lidar2cam_rt  = invert(cam2lidar_rt)         # = extrinsic
      viewpad       = eye(4); viewpad[:3,:3] = K    # = intrinsic

    Returns:
        intrinsics : list[np.ndarray] — 6 × (4, 4) viewpad matrices
        extrinsics : list[np.ndarray] — 6 × (4, 4) lidar2cam matrices
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    cam_map = {c["name"]: c for c in cfg["cameras"]}
    intrinsics, extrinsics = [], []
    for name in CAM_NAMES:
        if name not in cam_map:
            raise KeyError(
                f"Camera '{name}' not found in {yaml_path}. "
                f"Available: {list(cam_map.keys())}"
            )
        c = cam_map[name]

        # --- extrinsic: lidar2cam  (same path as nuscenes_dataset.py) ---
        cam2lidar_r = Quaternion(c["sensor2ego_rotation"]).rotation_matrix
        cam2lidar_t = c["sensor2ego_translation"]
        cam2lidar_rt = _convert_egopose_to_matrix(cam2lidar_r, cam2lidar_t)
        lidar2cam_rt = _invert_matrix_egopose(cam2lidar_rt)
        extrinsics.append(lidar2cam_rt)

        # --- intrinsic: 3×3 → 4×4 viewpad ---
        K3 = np.array(c["camera_intrinsic"], dtype=np.float64)  # 3×3
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:K3.shape[0], :K3.shape[1]] = K3
        intrinsics.append(viewpad)

    return intrinsics, extrinsics


def resize_image_and_intrinsic(img_bgr, intrinsic, target_size=(640, 640)):
    """Resize image to target_size and scale the intrinsic accordingly."""
    h0, w0 = img_bgr.shape[:2]
    tw, th = target_size
    w_scale = tw / w0
    h_scale = th / h0
    img_resized = cv2.resize(img_bgr, (tw, th))

    K = intrinsic.copy()
    K[0, 0] *= w_scale
    K[0, 2] *= w_scale
    K[1, 1] *= h_scale
    K[1, 2] *= h_scale
    return img_resized, K


def build_tokenised_input(images_rgb, intrinsics, extrinsics,
                           processor, tokenizer, command_desc=""):
    """Replicate what LoadAnnoatationVQATest does for a planning query.

    Returns a dict with:
        input_ids, attention_mask, pixel_values, image_grid_thw
        (all as torch tensors, batch-dim=1)
    """
    import copy
    from projects.mmdet3d_plugin.datasets.qwen_utils.data_qwen import (
        preprocess_qwen_2_visual,
    )

    # build the question text — same as the test pipeline
    traj_question = "Please provide the planning trajectory for the ego car without reasons."
    if command_desc:
        traj_question = command_desc + traj_question

    prompt = "You are driving in boston. "  # generic; model ignores the city name at inference
    question_content = (DEFAULT_IMAGE_TOKEN + "\n") * len(images_rgb) + prompt + traj_question

    sources = [
        [{"from": "human", "value": question_content}]
    ]

    # ── visual processing ──────────────────────────────────────────────
    uint8_imgs = [img.astype(np.uint8) for img in images_rgb]
    visual = processor.image_processor(images=uint8_imgs, return_tensors="pt")
    pixel_values   = visual["pixel_values"]       # (total_patches, C_patch)
    image_grid_thw = visual["image_grid_thw"]     # (N_imgs, 3)

    merge_size = processor.image_processor.merge_size
    grid_thw_merged = [
        thw.prod() // (merge_size ** 2) for thw in image_grid_thw
    ]

    # ── text tokenisation ──────────────────────────────────────────────
    data_dict = preprocess_qwen_2_visual(
        copy.deepcopy(sources),
        processor.tokenizer,
        grid_thw_image=grid_thw_merged,
        add_generation_prompt=True,
    )
    input_ids = data_dict["input_ids"][0]          # 1D tensor,  shape (L,)

    # All tensors returned here must have shape (1, ...) so that
    # forward_test's  data[k] = data[k][0].unsqueeze(0)  is a no-op.
    return dict(
        input_ids      = input_ids.unsqueeze(0),              # (1, L)
        attention_mask = torch.ones_like(input_ids).unsqueeze(0),  # (1, L)
        pixel_values   = pixel_values.unsqueeze(0),           # (1, total_patches, D)
        image_grid_thw = image_grid_thw.unsqueeze(0),         # (1, N_imgs, 3)
    )


def parse_waypoints(generated_text: str):
    """Extract (x, y) pairs from model output.

    Handles both decoded coord strings like '(-0.20, +1.50)' or raw
    POS_EMBEDDING tokens (replaced to coords by the model itself).
    """
    coords = re.findall(
        r'\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)',
        generated_text,
    )
    waypoints = []
    for x_str, y_str in coords:
        waypoints.append((float(x_str), float(y_str)))
    return waypoints


def project_bev_to_image(waypoints_xy, intrinsic, extrinsic):
    """Project BEV (x_lidar, y_lidar, z=0) waypoints into the image plane.

    Args:
        waypoints_xy : list of (x, y) in LiDAR / ego-BEV frame  [metres]
        intrinsic    : (4, 4) camera intrinsic
        extrinsic    : (4, 4) lidar2cam matrix

    Returns:
        uv_list : list of (u, v) pixel coordinates  (may be outside image)
        valid   : bool mask — True where the point is in front of the camera
    """
    uv_list = []
    valid   = []
    for x, y in waypoints_xy:
        pt_lidar  = np.array([x, y, 0.0, 1.0], dtype=np.float64)
        pt_cam    = extrinsic @ pt_lidar          # (4,)
        z_cam     = pt_cam[2]
        if z_cam <= 0.1:
            uv_list.append((None, None))
            valid.append(False)
            continue
        pt_img    = intrinsic @ pt_cam            # (4,)
        u = pt_img[0] / pt_img[2]
        v = pt_img[1] / pt_img[2]
        uv_list.append((u, v))
        valid.append(True)
    return uv_list, valid


def draw_waypoints_on_image(img_bgr, uv_list, valid, waypoints_xy):
    """Draw projected waypoints on image and return annotated PIL image."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    h, w = img_bgr.shape[:2]
    r = max(6, w // 80)    # dot radius scales with image width

    # colour gradient: green (near) → red (far)
    n = len(uv_list)
    for i, ((u, v), ok, (wx, wy)) in enumerate(zip(uv_list, valid, waypoints_xy)):
        t = i / max(n - 1, 1)
        colour = (int(255 * t), int(255 * (1 - t)), 0)   # BGR→RGB
        label  = f"({wx:.1f},{wy:.1f})"

        if ok and u is not None:
            u_px, v_px = int(round(u)), int(round(v))
            draw.ellipse(
                [u_px - r, v_px - r, u_px + r, v_px + r],
                fill=colour, outline=(255, 255, 255),
            )
            draw.text((u_px + r + 2, v_px - r), label, fill=colour)
        else:
            # waypoint behind camera — annotate in corner
            draw.text((10, 10 + i * 20), f"WP{i}: {label} (behind cam)", fill=colour)

    return pil


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SpaceDrive single-image inference — predicts BEV waypoints "
                    "and overlays them on the image."
    )
    # input
    img_grp = p.add_mutually_exclusive_group(required=True)
    img_grp.add_argument(
        "--image", type=str,
        help="Path to a single image (will be replicated across all 6 cameras).",
    )
    img_grp.add_argument(
        "--images", nargs=6, type=str, metavar="IMG",
        help="Paths to exactly 6 images: FRONT FRONT_RIGHT FRONT_LEFT BACK BACK_LEFT BACK_RIGHT.",
    )

    p.add_argument(
        "--ckpt", type=str,
        default="spacedrive_qwen/iter_21096.pth",
        help="Path to the SpaceDrive checkpoint (.pth).",
    )
    p.add_argument(
        "--config", type=str,
        default="projects/configs/spacedrive/spacedrive_qwen.py",
        help="mmcv config file for the model.",
    )
    p.add_argument(
        "--camera-params", type=str,
        default="tools/camera_params.yaml",
        help="YAML file with per-camera intrinsics and extrinsics.",
    )
    p.add_argument(
        "--command-desc", type=str, default="Turn right.",
        help='High-level driving command prepended to question, e.g. "Turn right."',
    )
    p.add_argument(
        "--output-dir", type=str, default="./infer_output_turn_right",
        help="Directory where the annotated image and result JSON are saved.",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device string.",
    )
    p.add_argument(
        "--target-size", nargs=2, type=int, default=[1600, 900],
        metavar=("W", "H"),
        help="Image size fed to the model (default: 640 640).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load camera parameters ──────────────────────────────────────
    cam_yaml = os.path.join(PROJECT_ROOT, args.camera_params) \
        if not os.path.isabs(args.camera_params) else args.camera_params
    print(f"[INFO] Loading camera params from: {cam_yaml}")
    intrinsics_orig, extrinsics = load_camera_params(cam_yaml)

    # ── 2. Load and resize images ──────────────────────────────────────
    target_w, target_h = args.target_size

    if args.image:
        img_paths = [args.image] * 6
    else:
        img_paths = args.images

    images_bgr = []
    intrinsics  = []
    for path, K_orig in zip(img_paths, intrinsics_orig):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img_resized, K_scaled = resize_image_and_intrinsic(
            img_bgr, K_orig, target_size=(target_w, target_h)
        )
        images_bgr.append(img_resized)
        intrinsics.append(K_scaled)

    # Convert BGR → RGB for the model
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    # ── 3. Build model from config ─────────────────────────────────────
    config_path = os.path.join(PROJECT_ROOT, args.config) \
        if not os.path.isabs(args.config) else args.config
    print(f"[INFO] Loading config: {config_path}")
    cfg = Config.fromfile(config_path)

    # Model internally writes JSON to save_path + sample_idx;
    # put those in a subdirectory so they don't pollute the output.
    model_json_dir = os.path.join(args.output_dir, "_model_json/")
    os.makedirs(model_json_dir, exist_ok=True)
    cfg.model.save_path = model_json_dir

    print("[INFO] Building model …")
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    model.eval()

    # ── 4. Load checkpoint ─────────────────────────────────────────────
    ckpt_path = os.path.join(PROJECT_ROOT, args.ckpt) \
        if not os.path.isabs(args.ckpt) else args.ckpt
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)
    model = model.to(device)

    processor = model.processor
    tokenizer = model.tokenizer

    # ── 5. Prepare tensors ─────────────────────────────────────────────
    print("[INFO] Preparing inputs …")
    batch = build_tokenised_input(
        images_rgb, intrinsics, extrinsics,
        processor, tokenizer,
        command_desc=args.command_desc,
    )

    # ── Standard shapes (batch=1) ──────────────────────────────────────
    # img tensor: (1, 6, 3, H, W)  float32, values 0-255
    img_tensor = torch.from_numpy(
        np.stack([i.astype(np.float32).transpose(2, 0, 1) for i in images_rgb], axis=0)
    ).unsqueeze(0).to(device)   # (1, 6, 3, H, W)

    # intrinsic / extrinsic tensors: (1, 6, 4, 4)
    intrinsics_t = torch.from_numpy(
        np.stack(intrinsics,  axis=0).astype(np.float32)
    ).unsqueeze(0).to(device)   # (1, 6, 4, 4)

    extrinsics_t = torch.from_numpy(
        np.stack(extrinsics, axis=0).astype(np.float32)
    ).unsqueeze(0).to(device)   # (1, 6, 4, 4)

    # lidar2img = intrinsic @ extrinsic  (K @ lidar2cam)
    lidar2img_t = intrinsics_t @ extrinsics_t  # (1, 6, 4, 4)

    # Move batch tensors to device
    for k in batch:
        batch[k] = batch[k].to(device)

    # Fake scalar states (ego_status is '' in spacedrive_qwen — not used)
    B = 1
    timestamp     = torch.zeros(B, dtype=torch.float32, device=device)
    img_timestamp = torch.zeros(B, 6,  dtype=torch.float32, device=device)
    ego_pose      = torch.eye(4, device=device).unsqueeze(0)    # (1, 4, 4)
    ego_pose_inv  = torch.eye(4, device=device).unsqueeze(0)    # (1, 4, 4)
    command       = torch.zeros(B, dtype=torch.long,  device=device)
    can_bus       = torch.zeros(B, 13, dtype=torch.float32, device=device)

    # img_metas: list[dict] — one dict per batch element (batch_size=1)
    img_metas = [{
        "sample_idx"  : "single_image_infer",
        "filename"    : img_paths,
        "ori_shape"   : [(target_h, target_w, 3)] * 6,
        "img_shape"   : [(target_h, target_w, 3)] * 6,
        "pad_shape"   : [(target_h, target_w, 3)] * 6,
        "scale_factor": [np.ones(4, dtype=np.float32)] * 6,
        "flip"        : False,
        "img_norm_cfg": dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False),
        "scene_token" : "single_image_infer",
        "box_mode_3d" : None,
        "box_type_3d" : None,
    }]

    # ── 6. Run inference ───────────────────────────────────────────────
    # SpaceDrive.forward_test() signature:
    #   forward_test(self, img, img_metas, rescale, **data)
    # It does:  data[k] = data[k][0].unsqueeze(0)  for every key.
    # All tensors in extra_data already have shape (1, ...), so the unwrap
    # is a shape-preserving no-op.
    # img is passed as the raw tensor (1,6,3,H,W); forward_test passes it
    # straight through to depth_prediction which expects img.shape to work.
    print("[INFO] Running inference …")

    extra_data = dict(
        intrinsics    = intrinsics_t,      # (1, 6, 4, 4)
        extrinsics    = extrinsics_t,      # (1, 6, 4, 4)
        lidar2img     = lidar2img_t,       # (1, 6, 4, 4)
        timestamp     = timestamp,         # (1,)
        img_timestamp = img_timestamp,     # (1, 6)
        ego_pose      = ego_pose,          # (1, 4, 4)
        ego_pose_inv  = ego_pose_inv,      # (1, 4, 4)
        command       = command,           # (1,)
        can_bus       = can_bus,           # (1, 13)
        **batch,    # input_ids (1,L), attention_mask (1,L),
                    # pixel_values (1,P,D), image_grid_thw (1,N,3)
    )

    with torch.no_grad():
        results = model(
            return_loss   = False,
            img           = img_tensor,      # (1, 6, 3, H, W) — passed as-is to depth_prediction
            img_metas     = img_metas,       # list[list[dict]]
            rescale       = True,
            **extra_data,
        )

    print("[INFO] Generation done.")

    # ── 7. Parse waypoints ─────────────────────────────────────────────
    raw_answer = ""
    if isinstance(results, list) and len(results) > 0:
        r = results[0]
        if isinstance(r, dict):
            raw_answer = r.get("A", "")
            if isinstance(raw_answer, list):
                raw_answer = raw_answer[0] if raw_answer else ""
        elif isinstance(r, str):
            raw_answer = r

    print(f"\n[INFO] Model answer:\n{raw_answer}\n")
    waypoints_xy = parse_waypoints(raw_answer)
    print(f"[INFO] Parsed {len(waypoints_xy)} waypoints: {waypoints_xy}")

    # ── 8. Project & visualise on the front camera image ──────────────
    front_bgr = images_bgr[0]          # CAM_FRONT (BGR, already at target size)
    front_K   = intrinsics[0]          # scaled 4×4 intrinsic
    front_E   = extrinsics[0]          # lidar2cam 4×4

    if waypoints_xy:
        uv_list, valid = project_bev_to_image(waypoints_xy, front_K, front_E)
        print(f"[INFO] Projected waypoints to image coordinates: {uv_list}, valid={valid}")
        annotated_pil  = draw_waypoints_on_image(front_bgr, uv_list, valid, waypoints_xy)
    else:
        warnings.warn("No waypoints parsed from the model output — saving unannotated image.")
        annotated_pil = Image.fromarray(cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB))

    # ── 9. Save outputs ────────────────────────────────────────────────
    out_img_path  = os.path.join(args.output_dir, "annotated_waypoints.png")
    out_json_path = os.path.join(args.output_dir, "result.json")

    annotated_pil.save(out_img_path)
    print(f"[INFO] Annotated image saved to: {out_img_path}")

    with open(out_json_path, "w") as f:
        json.dump(
            {"answer": raw_answer, "waypoints_xy_bev": waypoints_xy},
            f, indent=2,
        )
    print(f"[INFO] Result JSON  saved to: {out_json_path}")


if __name__ == "__main__":
    main()
