#!/usr/bin/env python3
"""
Visualize per-frame token dropping (feature / DTD drop details).

Reads a per-video drop-details JSON (saved by scripts/eval_egoeye.py --drop_vis_dir)
and generates a 3-panel video:
  Left:   Current frame with dropped token patches masked to BLACK
  Center: Current frame with dropped patches REPLACED by matched previous-frame patches
  Right:  Full current frame (unmodified)

Usage:
    python visualize_token_drop.py --drop_json <path_to_json>
    python visualize_token_drop.py --drop_dir  <dir_of_jsons>  # batch mode
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load_drop_details(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def read_sampled_frames(video_path: str, sampled_indices: list, target_h: int, target_w: int) -> np.ndarray:
    """Read specific frames from video and resize to (target_h, target_w)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_set = set(sampled_indices)
    max_idx = max(sampled_indices)

    frames = {}
    frame_idx = 0
    while frame_idx <= max_idx:
        ret, img = cap.read()
        if not ret:
            break
        if frame_idx in sampled_set:
            frames[frame_idx] = cv2.resize(img, (target_w, target_h))
        frame_idx += 1
    cap.release()

    # Build ordered array matching sampled_indices
    result = np.zeros((len(sampled_indices), target_h, target_w, 3), dtype=np.uint8)
    for i, si in enumerate(sampled_indices):
        if si in frames:
            result[i] = frames[si]
    return result


def generate_vis_video(details: dict, output_path: str, fps: float = 5.0):
    """Generate 3-panel visualization video from drop details.

    Left:   Current frame with dropped patches masked
            - Saliency drops → BLACK (0,0,0)
            - Matched drops  → GRAY  (128,128,128)
    Center: Current frame with matched-dropped patches replaced by prev-frame patches
            - Saliency drops → BLACK (no prev-frame match)
    Right:  Full current frame (unmodified)
    """
    video_path = details['video_path']
    sampled_indices = details['sampled_indices']
    frame_h = details['frame_h']
    frame_w = details['frame_w']
    h_patches = details['h_patches']
    w_patches = details['w_patches']
    token_size = details['token_size']
    frame_details = details['frames']

    # Saliency drops: dict of str(frame_idx) -> list of [h, w]
    sal_drops = details.get('saliency_drops_per_frame', {})
    has_saliency = len(sal_drops) > 0

    if not os.path.exists(video_path):
        print(f"  Video not found: {video_path}")
        return

    print(f"  Reading {len(sampled_indices)} sampled frames from {video_path} ...")
    if has_saliency:
        print(f"  Saliency drops present for {len(sal_drops)} frames")
    all_frames = read_sampled_frames(video_path, sampled_indices, frame_h, frame_w)

    # Video writer: 3 panels side by side
    panel_w = frame_w * 3
    panel_h = frame_h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (panel_w, panel_h))

    # Build a lookup: frame_idx -> detail
    detail_by_frame = {d['frame_idx']: d for d in frame_details}

    num_temporal_frames = len(sampled_indices)
    for t_idx in range(num_temporal_frames):
        curr_frame = all_frames[t_idx].copy()

        # Saliency drops for this frame (set of (h,w) tuples for fast lookup)
        sal_positions = sal_drops.get(str(t_idx), [])
        sal_set = {(p[0], p[1]) for p in sal_positions}

        has_imu_detail = t_idx in detail_by_frame
        has_any_drop = has_imu_detail or len(sal_set) > 0

        if not has_any_drop:
            # No drops at all: all 3 panels are identical
            left = curr_frame.copy()
            center = curr_frame.copy()
            right = curr_frame.copy()
        else:
            # 75% transparent white overlay on left panel (retained tokens stay visible)
            left = cv2.addWeighted(curr_frame, 0.95, np.full_like(curr_frame, 255), 0.25, 0)
            center = curr_frame.copy()
            right = curr_frame.copy()

            # --- Saliency drops: BLACK in left and center ---
            for (dh, dw) in sal_positions:
                y0 = dh * token_size
                x0 = dw * token_size
                y1 = min(y0 + token_size, frame_h)
                x1 = min(x0 + token_size, frame_w)
                left[y0:y1, x0:x1] = 0       # BLACK
                center[y0:y1, x0:x1] = 0      # BLACK (no prev-frame match)

            # --- Matched drops: GRAY in left, prev-patch replace in center ---
            if has_imu_detail:
                detail = detail_by_frame[t_idx]
                prev_t_idx = detail['prev_frame_idx']
                prev_frame = all_frames[prev_t_idx].copy()
                dropped_positions = detail['dropped_positions']
                matched_prev_positions = detail['matched_prev_positions']

                for (dh, dw), (ph, pw) in zip(dropped_positions, matched_prev_positions):
                    # Skip if already handled by saliency (saliency takes priority: BLACK)
                    if (dh, dw) in sal_set:
                        continue

                    # Left panel: GRAY for matched drops
                    y0 = dh * token_size
                    x0 = dw * token_size
                    y1 = min(y0 + token_size, frame_h)
                    x1 = min(x0 + token_size, frame_w)
                    left[y0:y1, x0:x1] = 0     # BLACK

                    # Center panel: replace with prev-frame patch
                    if ph < 0 or pw < 0:
                        dy0 = dh * token_size
                        dx0 = dw * token_size
                        dy1 = min(dy0 + token_size, frame_h)
                        dx1 = min(dx0 + token_size, frame_w)
                        center[dy0:dy1, dx0:dx1] = 40
                        continue
                    sy0 = ph * token_size
                    sx0 = pw * token_size
                    sy1 = min(sy0 + token_size, frame_h)
                    sx1 = min(sx0 + token_size, frame_w)
                    dy0 = dh * token_size
                    dx0 = dw * token_size
                    dy1 = min(dy0 + token_size, frame_h)
                    dx1 = min(dx0 + token_size, frame_w)
                    copy_h = min(sy1 - sy0, dy1 - dy0)
                    copy_w = min(sx1 - sx0, dx1 - dx0)
                    center[dy0:dy0+copy_h, dx0:dx0+copy_w] = prev_frame[sy0:sy0+copy_h, sx0:sx0+copy_w]

        # Draw thin grid lines on left and center panels to show token boundaries
        for panel in [left, center]:
            for r in range(1, h_patches):
                y = r * token_size
                cv2.line(panel, (0, y), (frame_w, y), (60, 60, 60), 1)
            for c in range(1, w_patches):
                x = c * token_size
                cv2.line(panel, (x, 0), (x, frame_h), (60, 60, 60), 1)

        combined = np.hstack([left, center, right])
        writer.write(combined)

        # Save left/right panels as images for every frame
        img_dir = str(Path(output_path).parent)
        stem = Path(output_path).stem
        cv2.imwrite(os.path.join(img_dir, f"{stem}_f{t_idx:03d}_left.png"), left)
        cv2.imwrite(os.path.join(img_dir, f"{stem}_f{t_idx:03d}_right.png"), right)

    writer.release()
    print(f"  Written: {output_path}  ({num_temporal_frames} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize IMU token drop as 3-panel video")
    parser.add_argument('--drop_json', type=str, default=None,
                        help='Path to a single drop-details JSON file')
    parser.add_argument('--drop_dir', type=str, default=None,
                        help='Directory of drop-details JSON files (batch mode)')
    parser.add_argument('--output_dir', type=str, default='token_drop_vis',
                        help='Output directory for visualization videos')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='Output video FPS (default: 5)')
    args = parser.parse_args()

    if args.drop_json is None and args.drop_dir is None:
        parser.error("Must specify either --drop_json or --drop_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = []
    if args.drop_json:
        json_files.append(args.drop_json)
    if args.drop_dir:
        for f in sorted(os.listdir(args.drop_dir)):
            if f.endswith('.json'):
                json_files.append(os.path.join(args.drop_dir, f))

    print(f"Found {len(json_files)} drop-details JSON file(s)")

    for jf in json_files:
        basename = Path(jf).stem
        output_path = os.path.join(args.output_dir, f"{basename}_vis.mp4")
        print(f"\nProcessing: {jf}")
        try:
            details = load_drop_details(jf)
            # Validate: must be a dict with expected keys
            if not isinstance(details, dict) or 'video_path' not in details or 'frames' not in details:
                print(f"  Skipped (not a drop-details file)")
                continue
            generate_vis_video(details, output_path, fps=args.fps)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
