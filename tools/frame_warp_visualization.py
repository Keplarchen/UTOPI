#!/usr/bin/env python3
"""
Frame Warp Visualization Video Generator (GPU-accelerated)

Generate a 3-panel video showing frame-to-frame warping:
  Left:   Previous frame
  Center: Current frame
  Right:  Previous frame warped to current viewpoint, blended with current frame

Reads MP4 video + trajectory CSV from AEA dataset.
No projectaria_tools dependency. Uses PyTorch GPU for warping.
"""

import os
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
import time
import argparse
import torch
import torch.nn.functional as F


def find_recording_path(video_path: str) -> Path:
    """Locate the AEA recording directory from a video file path."""
    video_path = Path(video_path)
    parts = video_path.stem.split('_')
    if len(parts) >= 7 and parts[0] == 'AriaEverydayActivities':
        recording_name = '_'.join(parts[2:-2])
    else:
        recording_name = video_path.parent.name

    if video_path.parent.name == recording_name:
        return video_path.parent

    current = video_path.parent
    while current.name not in ('', '/'):
        candidate = current / recording_name
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def ensure_trajectory(recording_path: Path) -> Path:
    """Extract trajectory CSV if needed, return path."""
    traj_file = recording_path / "mps" / "slam" / "closed_loop_trajectory.csv"
    if traj_file.exists():
        return traj_file

    traj_file.parent.mkdir(parents=True, exist_ok=True)
    name = recording_path.name
    traj_zip = recording_path / f"AriaEverydayActivities_1.0.0_{name}_mps_slam_trajectories.zip"
    if not traj_zip.exists():
        raise FileNotFoundError(f"Trajectory zip not found: {traj_zip}")

    with zipfile.ZipFile(traj_zip, 'r') as zf:
        for fi in zf.filelist:
            if 'closed_loop_trajectory.csv' in fi.filename:
                fi.filename = os.path.basename(fi.filename)
                zf.extract(fi, traj_file.parent)
    return traj_file


def load_trajectory(traj_file: Path):
    """Load trajectory, return sorted timestamps and pose arrays."""
    df = pd.read_csv(traj_file).sort_values('tracking_timestamp_us').reset_index(drop=True)
    timestamps = df['tracking_timestamp_us'].values.astype(np.int64)
    poses = df[['tx_world_device', 'ty_world_device', 'tz_world_device',
                'qx_world_device', 'qy_world_device', 'qz_world_device',
                'qw_world_device']].values.astype(np.float32)
    return timestamps, poses


def lookup_poses_batch(query_ts: np.ndarray, traj_ts: np.ndarray, traj_poses: np.ndarray) -> np.ndarray:
    """Vectorized nearest-neighbor pose lookup for all query timestamps."""
    idx = np.searchsorted(traj_ts, query_ts, side='left')
    idx = np.clip(idx, 1, len(traj_ts) - 1)
    # Choose closer of idx-1 and idx
    left_diff = np.abs(query_ts - traj_ts[idx - 1])
    right_diff = np.abs(query_ts - traj_ts[idx])
    idx = np.where(left_diff <= right_diff, idx - 1, idx)
    return traj_poses[idx]


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    """Convert Nx7 [tx,ty,tz,qx,qy,qz,qw] to Nx4x4 matrices."""
    n = len(poses)
    mats = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    mats[:, :3, :3] = R.from_quat(poses[:, 3:]).as_matrix().astype(np.float32)
    mats[:, :3, 3] = poses[:, :3]
    return mats


class GPUWarper:
    """GPU-accelerated depth-based frame warping."""

    def __init__(self, h: int, w: int, K: np.ndarray, T_device_camera: np.ndarray,
                 depth: float = 2.0, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h, self.w = h, w

        # Precompute pixel grid → 3D points at fixed depth (HW x 4, homogeneous)
        u, v = torch.meshgrid(
            torch.arange(w, device=self.device, dtype=torch.float32),
            torch.arange(h, device=self.device, dtype=torch.float32),
            indexing='xy',
        )
        ones = torch.ones_like(u)
        pixel_grid = torch.stack([u, v, ones], dim=-1).reshape(-1, 3)  # HW x 3

        K_inv_t = torch.from_numpy(np.linalg.inv(K)).to(self.device)
        pts_cam = (K_inv_t @ pixel_grid.T).T * depth  # HW x 3
        self.pts_homo = torch.cat([pts_cam, torch.ones(h * w, 1, device=self.device)], dim=1)  # HW x 4

        self.K_t = torch.from_numpy(K).to(self.device)
        self.T_dc = torch.from_numpy(T_device_camera).to(self.device)
        self.T_dc_inv = torch.inverse(self.T_dc)

        # Morphological close kernel size
        self.morph_k = 5
        self.morph_p = 2

    def warp(self, src_img: torch.Tensor, T_ref: torch.Tensor, T_curr: torch.Tensor) -> torch.Tensor:
        """
        Warp src_img (H,W,3 uint8 tensor on GPU) from T_ref to T_curr viewpoint.
        Returns (H,W,3 uint8 tensor on GPU).
        """
        h, w = self.h, self.w

        # T_rel = T_dc_inv @ T_curr_inv @ T_ref @ T_dc
        T_rel = self.T_dc_inv @ torch.inverse(T_curr) @ T_ref @ self.T_dc

        # Transform all points: (HW x 4) @ (4 x 4)^T → (HW x 4), take :3
        pts_curr = (self.pts_homo @ T_rel.T)[:, :3]  # HW x 3

        # Project: K @ pts^T → 3 x HW
        proj = (self.K_t @ pts_curr.T).T  # HW x 3
        z = proj[:, 2].clamp(min=1e-6)
        u_proj = torch.round(proj[:, 0] / z).long()
        v_proj = torch.round(proj[:, 1] / z).long()

        valid = (pts_curr[:, 2] > 0.001) & (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)

        # Filter non-black source pixels
        src_flat = src_img.reshape(-1, 3).float()
        non_black = (src_flat > 5).any(dim=1)
        valid = valid & non_black

        # Scatter write
        warped = torch.zeros(h, w, 3, device=self.device, dtype=torch.float32)
        warped[v_proj[valid], u_proj[valid]] = src_flat[valid]

        # Morphological close via max_pool: close = erode(dilate(img))
        warped_nchw = warped.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        dilated = F.max_pool2d(warped_nchw, self.morph_k, stride=1, padding=self.morph_p)
        closed = -F.max_pool2d(-dilated, self.morph_k, stride=1, padding=self.morph_p)

        return closed.squeeze(0).permute(1, 2, 0).to(torch.uint8)


def generate_video(
    video_path: str,
    output_video_path: str,
    max_frames: int = None,
    sample_frames: int = None,
    fps: float = 20.0,
    frame_gap: int = 1,
    blend_alpha: float = 0.5,
    process_size: tuple = (532, 532),
    output_size: tuple = (532, 532),
) -> dict:
    # ---- Setup ----
    rec_path = find_recording_path(video_path)
    if rec_path is None:
        return {"error": "Recording directory not found"}
    print(f"[FrameWarp] Recording: {rec_path.name}")

    traj_file = ensure_trajectory(rec_path)
    traj_ts, traj_poses = load_trajectory(traj_file)
    print(f"[FrameWarp] Loaded {len(traj_ts)} trajectory poses")

    # T_device_camera (Aria RGB camera extrinsics in the device frame)
    cam_device_pos = np.array([
        -0.00428072, -0.0118417, -0.00511398,
        0.32414, 0.0402123, 0.0410768, 0.944261,
    ], dtype=np.float32)
    T_dc = np.eye(4, dtype=np.float32)
    T_dc[:3, :3] = R.from_quat(cam_device_pos[3:]).as_matrix()
    T_dc[:3, 3] = cam_device_pos[:3]

    # Compensate for raw image orientation: the AEA camera image is stored
    # rotated 90° CW relative to the camera coordinate system convention.
    # T_dc is right-multiplied by a 90° CW rotation around the camera z-axis
    # (optical axis) instead of rotating every frame.
    R_z_cw90 = np.eye(4, dtype=np.float32)
    R_z_cw90[:3, :3] = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]], dtype=np.float32)
    T_dc = T_dc @ R_z_cw90

    # ---- Read MP4 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print(f"[FrameWarp] Video: {total_video_frames} frames @ {video_fps:.1f} fps")

    t0 = time.time()
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()

    if sample_frames and sample_frames < len(frames):
        total = len(frames)
        idx = np.linspace(0, total - 1, sample_frames, dtype=int)
        frames = [frames[i] for i in idx]
        print(f"[FrameWarp] Sampled {sample_frames} from {total} frames")

    num_frames = len(frames)
    print(f"[FrameWarp] Loaded {num_frames} frames in {time.time()-t0:.1f}s")
    if num_frames < 2:
        return {"error": "Need at least 2 frames"}

    # ---- Compute timestamps & batch pose lookup ----
    rgb_start_us = traj_ts[0] - 850075  # fixed AEA offset
    frame_interval_us = 50000.0
    if sample_frames and sample_frames < total_video_frames:
        original_indices = np.linspace(0, total_video_frames - 1, sample_frames, dtype=int)
    elif max_frames:
        original_indices = np.arange(min(max_frames, total_video_frames))
    else:
        original_indices = np.arange(total_video_frames)
    query_ts = (rgb_start_us + original_indices[:num_frames] * frame_interval_us).astype(np.int64)
    all_poses_7 = lookup_poses_batch(query_ts, traj_ts, traj_poses)  # N x 7
    all_T = poses_to_matrices(all_poses_7)  # N x 4x4

    # ---- Sizes & intrinsics ----
    h_orig, w_orig = frames[0].shape[:2]
    h_proc, w_proc = process_size
    h_out, w_out = output_size

    fx, fy = 610.94096 * w_proc / w_orig, 610.94096 * h_proc / h_orig
    cx, cy = 704.0 * w_proc / w_orig, 704.0 * h_proc / h_orig
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # ---- GPU setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[FrameWarp] Device: {device}, Process: {w_proc}x{h_proc}, Output: {w_out}x{h_out}")

    warper = GPUWarper(h_proc, w_proc, K, T_dc, depth=2.0, device=device)

    # Upload all pose matrices to GPU
    all_T_gpu = torch.from_numpy(all_T).to(device)

    # ---- Resize all frames upfront ----
    print("[FrameWarp] Preprocessing frames ...")
    resized = np.empty((num_frames, h_proc, w_proc, 3), dtype=np.uint8)
    for i, img in enumerate(frames):
        resized[i] = cv2.resize(img, (w_proc, h_proc))
    frames = None  # free memory

    # Upload all frames to GPU
    all_imgs_gpu = torch.from_numpy(resized).to(device)

    # ---- Video writer ----
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 3 * w_out, h_out
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (panel_w, panel_h))

    # ---- Generate ----
    t_start = time.time()
    need_resize = (h_proc, w_proc) != (h_out, w_out)

    for i in range(num_frames):
        prev_i = max(0, i - frame_gap)

        # GPU warp
        warped = warper.warp(all_imgs_gpu[prev_i], all_T_gpu[prev_i], all_T_gpu[i])

        # Blend on GPU
        curr_f = all_imgs_gpu[i].float()
        blended = (curr_f * blend_alpha + warped.float() * (1.0 - blend_alpha)).to(torch.uint8)

        # Transfer to CPU for video writing
        prev_cpu = resized[prev_i]
        curr_cpu = resized[i]
        blended_cpu = blended.cpu().numpy()

        if need_resize:
            prev_cpu = cv2.resize(prev_cpu, (w_out, h_out))
            curr_cpu = cv2.resize(curr_cpu, (w_out, h_out))
            blended_cpu = cv2.resize(blended_cpu, (w_out, h_out))

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(prev_cpu, f"Prev (f={prev_i})", (10, 25), font, 0.6, (0, 255, 255), 2)
        cv2.putText(curr_cpu, f"Curr (f={i})", (10, 25), font, 0.6, (0, 255, 255), 2)
        cv2.putText(blended_cpu, f"Warp+Blend (a={blend_alpha:.1f})", (10, 25), font, 0.6, (0, 255, 255), 2)

        combined = np.hstack([prev_cpu, curr_cpu, blended_cpu])
        writer.write(combined)

        if i % 100 == 0:
            elapsed = time.time() - t_start
            print(f"[FrameWarp] Frame {i}/{num_frames} ({(i+1)/elapsed:.1f} fps)")

    writer.release()
    total_time = time.time() - t_start
    print(f"\n[FrameWarp] Done in {total_time:.1f}s ({num_frames/total_time:.1f} fps)")
    print(f"  Output: {output_video_path}")
    return {"total_frames": num_frames, "total_time_s": total_time}


def main():
    parser = argparse.ArgumentParser(description="3-panel frame warp visualization (GPU)")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="frame_warp_vis")
    parser.add_argument("--output_video", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--sample_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--frame_gap", type=int, default=1)
    parser.add_argument("--blend_alpha", type=float, default=0.5)
    parser.add_argument("--output_size", type=int, nargs=2, default=[532, 532])
    parser.add_argument("--process_size", type=int, nargs=2, default=[532, 532])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.output_video is None:
        seq_name = Path(args.video_path).stem
        parts = [seq_name, f"gap{args.frame_gap}", f"alpha{args.blend_alpha}"]
        if args.max_frames:
            parts.append(f"frames{args.max_frames}")
        if args.sample_frames:
            parts.append(f"sample{args.sample_frames}")
        output_video_path = str(Path(args.output_dir) / ("_".join(str(p) for p in parts) + ".mp4"))
    else:
        output_video_path = args.output_video

    generate_video(
        video_path=args.video_path,
        output_video_path=output_video_path,
        max_frames=args.max_frames,
        sample_frames=args.sample_frames,
        fps=args.fps,
        frame_gap=args.frame_gap,
        blend_alpha=args.blend_alpha,
        process_size=tuple(args.process_size),
        output_size=tuple(args.output_size),
    )


if __name__ == "__main__":
    main()
