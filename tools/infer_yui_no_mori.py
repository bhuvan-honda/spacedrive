#!/usr/bin/env python3
# -------------------------------------------------------------------------
# SpaceDrive+ — video inference on the yui-no-mori-processed capture
# -------------------------------------------------------------------------
# Usage:
#   python tools/infer_yui_no_mori.py \
#       --data-root yui-no-mori-processed \
#       --ckpt      results/spacedrive_plus_qwen_depth_orth_hard/latest.pth \
#       --config    projects/configs/spacedrive/spacedrive_plus_qwen_depth_orth.py \
#       --output-dir ./infer_yui_no_mori_output \
#       --stride 5
#
# What this does, and why it differs from tools/infer_single_image.py /
# tools/infer_video.py:
#
#  1. Camera calibration format is NOT the nuScenes-style cam2ego quaternion +
#     translation used by tools/camera_params.yaml. Each
#     camera_params/00_0N.json stores "reference-to-<sensor>", i.e. the
#     REFERENCE (vehicle body, x-forward/y-left/z-up — same convention as the
#     nuScenes ego/lidar frame) → camera transform *directly*. This is already
#     what the model calls "lidar2cam" — inverting it again (as
#     infer_single_image.load_camera_params does for the old YAML format)
#     would silently rotate every projected point into the wrong half-space.
#     Intrinsics live in a separate camera_params/intrinsic/camera/data/0N.json
#     (pinhole, zero distortion, native 3840x2160).
#
#     The image folders (camera/Camera_01..06) carry no sensor name, so the
#     folder → sensor mapping below was resolved by decoding each camera's
#     position/heading in the reference frame from its extrinsic (front-center
#     points +x with ~0 lateral offset, the "rear" camera sits ~2.4m behind
#     the reference origin facing -x, etc.) and cross-checking Camera_01
#     against the extrinsic-derived front-center pose (clear forward vanishing
#     point, no motion blur). If your capture rig numbers the folders
#     differently, fix FOLDER_TO_SENSOR — everything below is driven by it.
#
#  2. SpaceDrive+ (unlike plain SpaceDrive/spacedrive_qwen) is built with
#     ego_status='feature+PE': it keeps a StreamPETR-style running memory of
#     past ego poses / can_bus vectors across consecutive forward_test() calls
#     on the SAME model instance (see SpaceDrive.pre_update_memory /
#     post_update_memory). Feeding a fixed identity ego_pose and command=0 per
#     frame (what infer_single_image.py / infer_video.py do, since the base
#     model doesn't use ego_status) would still run, but the temporal ego
#     feature would be meaningless. Here we instead:
#       - build a REAL ego_pose (ego→ENU-world SE3) per frame from
#         trajectory.npz (p_enu position + q_b2e body→ENU quaternion),
#       - process frames strictly in order on one persistent model instance
#         (model.reset_memory() once, then never rebuilt), so the memory
#         queue is temporally meaningful,
#       - derive the discrete high-level `command` (and its language
#         description `command_desc`, which the model reads inside the
#         question because the config sets load_ego_command_in_question=True)
#         from the REAL recorded future trajectory 3s ahead, using the exact
#         bucketing in tools/command_generation.py (same angle thresholds),
#         instead of a fixed "Turn right." string.
#     `can_bus` (a 13-dim vector whose exact original semantics — accel /
#     rotation-rate / velocity slices — are only produced by a converter not
#     present in this repo) is left at zero, matching the precedent already
#     established in infer_single_image.py / infer_video.py.
#
#  3. Camera frames have no per-frame timestamp file. trajectory.npz is INS
#     data at 50 Hz; frame_i is assumed to align with INS sample i*decimation
#     (default decimation=5 → 10 Hz), confirmed by:  3942 frames / 10 fps =
#     394.2s, ~13s less than the 407.1s INS span — consistent with the INS
#     logger starting slightly before camera capture. Override with
#     --decimation/--ins-hz if you know the true values.
#
#  4. The model's own vision input is resized to a square (default 640x640,
#     matching this config's training img_scale) with independent x/y
#     scaling — correct for the model, but it would visibly squash a 16:9
#     frame if used for the output video. So the front camera is loaded
#     TWICE per frame: once resized (squashed) for the model, once resized
#     to --display-size (aspect-ratio preserved) for drawing/output video,
#     each with its own correctly-scaled intrinsic.
#
#  5. The model's predicted (x, y) waypoints are meant to be projected onto
#     z=0 in the calibration frame — that's correct for nuScenes, whose ego
#     frame origin sits at road level (so sensor extrinsics have POSITIVE z).
#     Here all 6 decoded camera extrinsics have NEGATIVE z relative to the
#     "reference" origin, meaning the origin sits ABOVE every camera (roof /
#     GNSS-antenna height), not at the road. Projecting at z=0 therefore hits
#     a virtual plane near roof height instead of the ground, which visually
#     collapses near-range waypoints toward the horizon. See
#     project_bev_to_image_at_height()'s docstring for the numeric check;
#     --ground-height (default 1.5m) controls this and should be corrected if
#     the true reference-to-road offset for this rig is known.
# -------------------------------------------------------------------------

import os
import sys
import time
import argparse
import json
import warnings

import cv2
import numpy as np
import torch
from PIL import ImageDraw
from nuscenes.eval.common.utils import Quaternion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'projects'))
import projects.mmdet3d_plugin  # noqa: F401

from infer_single_image import (
    CAM_NAMES,
    resize_image_and_intrinsic,
    build_tokenised_input,
    parse_waypoints,
    draw_waypoints_on_image,
    _invert_matrix_egopose,
)

# ── Camera calibration file layout for yui-no-mori-processed ──────────────
FOLDER_TO_SENSOR = {
    'Camera_01': 'front_center_camera',
    'Camera_02': 'left_front_camera',
    'Camera_03': 'right_front_camera',
    'Camera_04': 'left_rear_camera',
    'Camera_05': 'right_rear_camera',
    'Camera_06': 'rear_camera',
}
SENSOR_TO_FOLDER = {v: k for k, v in FOLDER_TO_SENSOR.items()}

# CAM_NAMES (order the model / lidar2img stacking expects) -> physical sensor
SENSOR_FOR_CAM = {
    'CAM_FRONT':       'front_center_camera',
    'CAM_FRONT_RIGHT': 'right_front_camera',
    'CAM_FRONT_LEFT':  'left_front_camera',
    'CAM_BACK':        'rear_camera',
    'CAM_BACK_LEFT':   'left_rear_camera',
    'CAM_BACK_RIGHT':  'right_rear_camera',
}

EXTRINSIC_FILES = {
    'front_center_camera': '00_00.json',
    'left_front_camera':   '00_01.json',
    'right_front_camera':  '00_02.json',
    'left_rear_camera':    '00_03.json',
    'right_rear_camera':   '00_04.json',
    'rear_camera':         '00_05.json',
}
INTRINSIC_FILES = {
    'front_center_camera': 'intrinsic/camera/data/00.json',
    'left_front_camera':   'intrinsic/camera/data/01.json',
    'right_front_camera':  'intrinsic/camera/data/02.json',
    'left_rear_camera':    'intrinsic/camera/data/03.json',
    'right_rear_camera':   'intrinsic/camera/data/04.json',
    'rear_camera':         'intrinsic/camera/data/05.json',
}


def _load_extrinsic(path, sensor_name):
    with open(path, 'r') as f:
        d = json.load(f)
    key = f'reference-to-{sensor_name}'
    mat = np.array(d[key]['param']['sensor_calib']['data'], dtype=np.float64)
    return mat  # reference(=ego/lidar frame) -> camera, i.e. lidar2cam directly


def _load_intrinsic(path):
    with open(path, 'r') as f:
        d = json.load(f)
    p = next(iter(d.values()))['param']
    K3 = np.array(p['cam_K']['data'], dtype=np.float64)
    dist = np.array(p['cam_dist']['data'], dtype=np.float64).reshape(-1)
    if np.abs(dist).sum() > 1e-6:
        warnings.warn(f"{path}: non-zero distortion {dist} is ignored (pinhole projection only)")
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = K3
    size = (int(p['img_dist_w']), int(p['img_dist_h']))
    return viewpad, size


def load_yui_camera_params(camera_params_dir):
    """Returns dict[sensor_name] -> dict(K=4x4 viewpad, E=4x4 lidar2cam, size=(w,h))."""
    out = {}
    for sensor in EXTRINSIC_FILES:
        E = _load_extrinsic(os.path.join(camera_params_dir, EXTRINSIC_FILES[sensor]), sensor)
        K, size = _load_intrinsic(os.path.join(camera_params_dir, INTRINSIC_FILES[sensor]))
        out[sensor] = dict(K=K, E=E, size=size)
    return out


# ── Command bucketing — identical thresholds/text to tools/command_generation.py,
#    applied to the real future ego trajectory instead of a stored GT pickle.
#
#    NOTE on the left/right label: (dx, dy) below is computed in the INS body frame
#    with x=forward, y=left, z=up — this is forced by construction (x=forward is
#    verified against real velocity direction; y=left follows from x-forward +
#    z-up + right-handedness, i.e. cross(z, x) = y). command_generation.py's
#    positive-angle="right"/negative-angle="left" labelling was written against
#    nuScenes' own `gt_planning` array, whose y-sign convention relative to this
#    INS frame was not independently verifiable from this repo. The turn
#    *magnitude* (straight/slight/sharp) is convention-independent and reliable;
#    the *word* "left" vs "right" could in principle be mirrored. This does not
#    affect the projected trajectory drawn on the video, which uses the model's
#    raw predicted (x, y) unmodified, exactly as project_bev_to_image already did
#    in the reference scripts. ──
def classify_command(dx, dy):
    if (dx ** 2 + dy ** 2) ** 0.5 < 0.5:
        angle = 0.0
    else:
        angle = float(np.degrees(np.arctan2(dy, dx)))

    if -0.4 <= angle <= 0.4:
        return 0, "The ego vehicle is moving strictly forward. "
    elif (0.4 < angle <= 2) or (-2 <= angle < -0.4):
        return 1, "The ego vehicle is moving forward. "
    elif 2 < angle <= 10:
        return 2, "The ego vehicle is moving slightly right. "
    elif 10 < angle <= 20:
        return 3, "The ego vehicle is turning right. "
    elif angle > 20:
        return 4, "The ego vehicle is turning sharp right. "
    elif -10 <= angle < -2:
        return 5, "The ego vehicle is moving slightly left. "
    elif -20 <= angle < -10:
        return 6, "The ego vehicle is turning left. "
    else:
        return 7, "The ego vehicle is turning sharp left. "


def project_bev_to_image_at_height(waypoints_xy, intrinsic, extrinsic, ground_z):
    """Same as infer_single_image.project_bev_to_image, but the ground plane is
    z=ground_z instead of z=0.

    z=0 (nuScenes' convention: the ego/lidar frame origin sits at road level, so
    sensors have POSITIVE z translations) is wrong for this rig: decoding the
    extrinsics shows ALL 6 cameras sit at NEGATIVE z relative to the "reference"
    origin (front-center: -0.30, corners: -0.7 to -0.93) — i.e. every camera is
    BELOW the reference origin, which is only possible if the origin itself is
    mounted above the cameras (roof/GNSS-antenna height), not at the road.
    Projecting the model's waypoints at z=0 therefore projects them onto a
    virtual plane near roof height instead of the road, which visually collapses
    near waypoints toward the horizon (verified numerically: z=0 gives a
    non-monotonic, unphysical image-row/distance relationship; z=-1.5 restores
    the expected monotonic near-low/far-high behavior with the nearest waypoint
    landing just past the hood, exactly where a ~2m ground point should be).
    ground_z is exposed via --ground-height since the true reference-to-road
    offset isn't in the calibration data; 1.5m is an engineering estimate
    (roof/antenna-height origin, dash/bumper-mounted cameras below it).
    """
    uv_list = []
    valid = []
    for x, y in waypoints_xy:
        pt_ref = np.array([x, y, ground_z, 1.0], dtype=np.float64)
        pt_cam = extrinsic @ pt_ref
        z_cam = pt_cam[2]
        if z_cam <= 0.1:
            uv_list.append((None, None))
            valid.append(False)
            continue
        pt_img = intrinsic @ pt_cam
        u = pt_img[0] / pt_img[2]
        v = pt_img[1] / pt_img[2]
        uv_list.append((u, v))
        valid.append(True)
    return uv_list, valid


class Trajectory:
    """Wraps trajectory.npz: p_enu (world position), q_b2e (body->ENU quaternion, xyzw)."""

    def __init__(self, npz_path, ins_hz=50.0):
        d = np.load(npz_path)
        self.t = d['t_tai'].astype(np.float64)
        self.p = d['p_enu'].astype(np.float64)
        self.q = d['q_b2e'].astype(np.float64)  # [x, y, z, w]
        self.ins_hz = ins_hz
        self.n = len(self.t)

    def _clamp(self, idx):
        return min(max(idx, 0), self.n - 1)

    def rotation(self, idx):
        idx = self._clamp(idx)
        qx, qy, qz, qw = self.q[idx]
        return Quaternion([qw, qx, qy, qz]).rotation_matrix  # body -> ENU

    def pose_matrix(self, idx):
        """ego (body) -> world (ENU) SE3, 4x4."""
        idx = self._clamp(idx)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.rotation(idx)
        M[:3, 3] = self.p[idx]
        return M

    def future_delta_body(self, idx, horizon_s=3.0):
        """Displacement `horizon_s` seconds ahead, expressed in the ego frame at `idx`."""
        idx = self._clamp(idx)
        fut_idx_raw = idx + int(round(horizon_s * self.ins_hz))
        fut_idx = self._clamp(fut_idx_raw)
        R_now = self.rotation(idx)
        delta_world = self.p[fut_idx] - self.p[idx]
        delta_body = R_now.T @ delta_world
        truncated = fut_idx_raw > self.n - 1
        return float(delta_body[0]), float(delta_body[1]), truncated

    def rel_time(self, idx):
        idx = self._clamp(idx)
        return float(self.t[idx] - self.t[0])


def load_config(config_path):
    """Config.fromfile, tolerant of configs dumped into results/<exp>/ by training,
    which contain an unloadable `gpu_ids = [device(type='cuda', ...), ...]` block."""
    with open(config_path, 'r') as f:
        text = f.read()
    if 'device(type=' not in text:
        return Config.fromfile(config_path)
    import re
    import tempfile
    text = re.sub(r"^gpu_ids\s*=\s*\[.*?^\]\s*$", "gpu_ids = [0]", text, flags=re.S | re.M)
    with tempfile.NamedTemporaryFile('w', suffix='.py', dir=os.path.dirname(config_path), delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        return Config.fromfile(tmp_path)
    finally:
        os.remove(tmp_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="SpaceDrive+ inference on yui-no-mori-processed -> annotated front-camera MP4"
    )
    p.add_argument('--data-root', type=str, default='yui-no-mori-processed')
    p.add_argument('--ckpt', type=str,
                    default='results/spacedrive_plus_qwen_depth_orth_hard/latest.pth')
    p.add_argument('--config', type=str,
                    default='projects/configs/spacedrive/spacedrive_plus_qwen_depth_orth.py')
    p.add_argument('--depth-skip', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=None,
                    help="Override model.depth_skip (use if the checkpoint's variant differs from --config).")
    p.add_argument('--depth-skip-mode', type=str, default=None, choices=['concat', 'add'],
                    help="Override model.depth_skip_mode, e.g. 'add' for the add_v2 checkpoints.")
    p.add_argument('--depth-orth-mode', type=str, default=None, choices=['none', 'hard', 'soft'],
                    help="Override model.depth_orth_mode.")
    p.add_argument('--output-dir', type=str, default='./infer_yui_no_mori_output')
    p.add_argument('--device', type=str, default='cuda:0')

    p.add_argument('--model-input-size', nargs=2, type=int, default=[640, 640],
                    metavar=('W', 'H'), help="Resize fed to the model (must match training img_scale).")
    p.add_argument('--display-size', nargs=2, type=int, default=[1280, 720],
                    metavar=('W', 'H'), help="Aspect-ratio-preserving resize used for the output video.")
    p.add_argument('--ground-height', type=float, default=1.5,
                    help="Height (metres) of the camera_params 'reference' origin above the road. "
                         "All 6 cameras decode to NEGATIVE z in that frame, so the origin sits above "
                         "them (roof/antenna height), not at road level as nuScenes assumes — waypoints "
                         "are projected at z=-ground_height instead of z=0. See "
                         "project_bev_to_image_at_height() docstring. Override if you know the rig's "
                         "true reference-to-road offset.")

    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--stride', type=int, default=1,
                    help="Process every Nth frame (frame indices still map through --decimation to "
                         "the trajectory). Use >1 to cut runtime for a long capture.")

    p.add_argument('--decimation', type=int, default=5,
                    help="frame_idx -> INS sample index = frame_idx * decimation.")
    p.add_argument('--ins-hz', type=float, default=50.0, help="Trajectory sample rate.")
    p.add_argument('--horizon-s', type=float, default=3.0,
                    help="Look-ahead horizon (seconds) used for the high-level command, "
                         "matching the 3s/6-waypoint planning horizon used in training.")

    p.add_argument('--location', type=str, default='boston',
                    help="City token in 'You are driving in {location}.' — the model was only ever "
                         "conditioned on nuScenes city names; this text is not semantically load-bearing.")
    p.add_argument('--fps', type=float, default=None,
                    help="Output video fps. Default: (ins_hz / decimation) / stride, i.e. real-time.")
    return p.parse_args()


def build_model_and_inputs(cfg, ckpt_path, device, output_dir):
    model_json_dir = os.path.join(output_dir, "_model_json/")
    os.makedirs(model_json_dir, exist_ok=True)
    cfg.model.save_path = model_json_dir

    print("[INFO] Building model …")
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    model.eval()

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)
    model = model.to(device)
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(PROJECT_ROOT, args.data_root)
    camera_params_dir = os.path.join(data_root, 'camera_params')
    camera_dir = os.path.join(data_root, 'camera')
    traj_path = os.path.join(data_root, 'trajectory.npz')

    print(f"[INFO] Loading camera params from: {camera_params_dir}")
    cam_params = load_yui_camera_params(camera_params_dir)

    print(f"[INFO] Loading trajectory from: {traj_path}")
    traj = Trajectory(traj_path, ins_hz=args.ins_hz)

    # total available frames = min frame count across the 6 camera folders
    n_frames = min(
        len([f for f in os.listdir(os.path.join(camera_dir, folder)) if f.endswith('.jpg')])
        for folder in FOLDER_TO_SENSOR
    )
    start = args.start_frame
    end = n_frames if args.max_frames is None else min(start + args.max_frames, n_frames)
    frame_indices = list(range(start, end, args.stride))
    print(f"[INFO] {n_frames} frames available; processing {len(frame_indices)} "
          f"(frames {start}..{end - 1}, stride {args.stride})")

    model_w, model_h = args.model_input_size
    disp_w, disp_h = args.display_size

    config_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    print(f"[INFO] Loading config: {config_path}")
    cfg = load_config(config_path)
    for key, val in (('depth_skip', args.depth_skip), ('depth_skip_mode', args.depth_skip_mode),
                     ('depth_orth_mode', args.depth_orth_mode)):
        if val is not None:
            cfg.model[key] = val
            print(f"[INFO] Override model.{key} = {val}")

    model = build_model_and_inputs(cfg, args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt),
                                    device, args.output_dir)
    processor = model.processor
    tokenizer = model.tokenizer

    if not hasattr(model, 'ego_status') or model.ego_status is None:
        warnings.warn(
            "The loaded model has ego_status=None (not a SpaceDrive+ config) — the real ego_pose / "
            "command signals built here will be ignored by forward_test."
        )
    model.reset_memory()

    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    all_results = []

    fps = args.fps if args.fps is not None else (args.ins_hz / args.decimation) / args.stride
    print(f"[INFO] Output video fps: {fps:.3f}")

    t_start = time.time()
    for step, frame_idx in enumerate(frame_indices):
        # ── load + resize the 6 views for the model ───────────────────
        images_rgb = []
        intrinsics_model = []
        extrinsics_list = []
        for cam in CAM_NAMES:
            sensor = SENSOR_FOR_CAM[cam]
            folder = SENSOR_TO_FOLDER[sensor]
            path = os.path.join(camera_dir, folder, f"frame_{frame_idx:05d}.jpg")
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            img_resized, K_scaled = resize_image_and_intrinsic(
                img_bgr, cam_params[sensor]['K'], target_size=(model_w, model_h))
            images_rgb.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            intrinsics_model.append(K_scaled)
            extrinsics_list.append(cam_params[sensor]['E'])

        # ── front camera, separately resized (aspect preserved) for display ──
        front_sensor = 'front_center_camera'
        front_path = os.path.join(camera_dir, SENSOR_TO_FOLDER[front_sensor], f"frame_{frame_idx:05d}.jpg")
        front_bgr_orig = cv2.imread(front_path)
        front_disp_bgr, front_disp_K = resize_image_and_intrinsic(
            front_bgr_orig, cam_params[front_sensor]['K'], target_size=(disp_w, disp_h))
        front_disp_E = cam_params[front_sensor]['E']

        # ── ego state from the real trajectory ─────────────────────────
        ins_idx = frame_idx * args.decimation
        ego_pose_np = traj.pose_matrix(ins_idx)
        ego_pose_inv_np = _invert_matrix_egopose(ego_pose_np)
        dx, dy, truncated = traj.future_delta_body(ins_idx, horizon_s=args.horizon_s)
        command_idx, command_desc = classify_command(dx, dy)
        t_now = traj.rel_time(ins_idx)

        # ── tokenised text input (image tokens + location + command_desc + question) ──
        batch = build_tokenised_input(
            images_rgb, intrinsics_model, extrinsics_list,
            processor, tokenizer, command_desc=command_desc,
        )

        img_tensor = torch.from_numpy(
            np.stack([i.astype(np.float32).transpose(2, 0, 1) for i in images_rgb], axis=0)
        ).unsqueeze(0).to(device)

        intrinsics_t = torch.from_numpy(
            np.stack(intrinsics_model, axis=0).astype(np.float32)
        ).unsqueeze(0).to(device)
        extrinsics_t = torch.from_numpy(
            np.stack(extrinsics_list, axis=0).astype(np.float32)
        ).unsqueeze(0).to(device)
        lidar2img_t = intrinsics_t @ extrinsics_t

        ego_pose_t = torch.from_numpy(ego_pose_np.astype(np.float32)).unsqueeze(0).to(device)
        ego_pose_inv_t = torch.from_numpy(ego_pose_inv_np.astype(np.float32)).unsqueeze(0).to(device)
        command_t = torch.tensor([float(command_idx)], dtype=torch.float32, device=device)
        can_bus_t = torch.zeros(1, 13, dtype=torch.float32, device=device)
        timestamp_t = torch.tensor([t_now], dtype=torch.float32, device=device)
        img_timestamp_t = timestamp_t.unsqueeze(1).repeat(1, 6)

        for k in batch:
            batch[k] = batch[k].to(device)

        extra_data = dict(
            intrinsics=intrinsics_t,
            extrinsics=extrinsics_t,
            lidar2img=lidar2img_t,
            timestamp=timestamp_t,
            img_timestamp=img_timestamp_t,
            ego_pose=ego_pose_t,
            ego_pose_inv=ego_pose_inv_t,
            command=command_t,
            can_bus=can_bus_t,
            **batch,
        )

        img_metas = [{
            "sample_idx": f"frame_{frame_idx:05d}",
            "filename": [os.path.join(camera_dir, SENSOR_TO_FOLDER[SENSOR_FOR_CAM[c]],
                                       f"frame_{frame_idx:05d}.jpg") for c in CAM_NAMES],
            "ori_shape": [(model_h, model_w, 3)] * 6,
            "img_shape": [(model_h, model_w, 3)] * 6,
            "pad_shape": [(model_h, model_w, 3)] * 6,
            "scale_factor": [np.ones(4, dtype=np.float32)] * 6,
            "flip": False,
            "img_norm_cfg": dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False),
            "scene_token": "yui_no_mori",
            "box_mode_3d": None,
            "box_type_3d": None,
        }]

        with torch.no_grad():
            results = model(
                return_loss=False,
                img=img_tensor,
                img_metas=img_metas,
                rescale=True,
                **extra_data,
            )

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

        if waypoints_xy:
            uv_list, valid = project_bev_to_image_at_height(
                waypoints_xy, front_disp_K, front_disp_E, ground_z=-args.ground_height)
            annotated_pil = draw_waypoints_on_image(front_disp_bgr, uv_list, valid, waypoints_xy)
        else:
            uv_list, valid = [], []
            from PIL import Image
            annotated_pil = Image.fromarray(cv2.cvtColor(front_disp_bgr, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(annotated_pil)
        overlay = f"frame {frame_idx:05d}  |  {command_desc.strip()}"
        if truncated:
            overlay += "  [future truncated]"
        draw.rectangle([0, disp_h - 22, disp_w, disp_h], fill=(0, 0, 0))
        draw.text((6, disp_h - 20), overlay, fill=(255, 255, 0))

        frame_png = os.path.join(frames_dir, f"{step:06d}.png")
        annotated_pil.save(frame_png)

        all_results.append({
            "frame": frame_idx,
            "ins_idx": ins_idx,
            "answer": raw_answer,
            "waypoints_xy_bev": waypoints_xy,
            "command": command_idx,
            "command_desc": command_desc,
            "future_truncated": truncated,
        })

        if step == 0:
            dt = time.time() - t_start
            eta = dt * len(frame_indices)
            print(f"[INFO] First frame took {dt:.1f}s -> ETA for {len(frame_indices)} frames: "
                  f"{eta / 60:.1f} min")
        print(f"[{step + 1}/{len(frame_indices)}] frame {frame_idx:05d}  cmd={command_idx} "
              f"({command_desc.strip()})  waypoints={len(waypoints_xy)}")

    # ── assemble video (cv2.VideoWriter — no ffmpeg dependency) ─────────
    video_path = os.path.join(args.output_dir, "waypoints_video.mp4")
    if all_results:
        writer = None
        for step in range(len(all_results)):
            frame_png = os.path.join(frames_dir, f"{step:06d}.png")
            frame_bgr = cv2.imread(frame_png)
            if frame_bgr is None:
                continue
            if writer is None:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            writer.write(frame_bgr)
        if writer is not None:
            writer.release()
            print(f"[INFO] Video saved: {video_path}")
        else:
            print("[WARN] No frames written; video not created.")

    json_path = os.path.join(args.output_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Results JSON saved: {json_path}")
    print(f"[INFO] Annotated frames saved in: {frames_dir}/")


if __name__ == "__main__":
    main()
