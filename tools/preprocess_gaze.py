"""Project AEA eye-gaze (yaw/pitch in CPF) to pixel coordinates of the preview RGB mp4.

Pipeline (projectaria_tools official calibration path):
  gaze point in CPF (yaw, pitch, depth)  ->  T_camera_cpf  ->  fisheye624 projection at 2880 res
  ->  rescale to 1408 (crop 32 px each side, 2x binning)  ->  rotate 90 deg CW (preview mp4 is upright)

Writes <seq_dir>/gaze_px.csv with one row per video frame:
  frame_idx, timestamp_us, gaze_x, gaze_y, ref_w, ref_h, yaw_rads_cpf, pitch_rads_cpf, valid
gaze_x / gaze_y are in 1408x1408 preview-video pixel coordinates (NaN when invalid).

Frame -> device timestamp mapping is the same convention used by the IVOC code:
  ts(frame) = closed_loop_trajectory.ts[0] - RGB_OFFSET_US + frame_idx * (1e6 / FPS)
"""
import argparse
import glob
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
from projectaria_tools.core import calibration as ac
from projectaria_tools.core import mps

RGB_OFFSET_US = 850075   # fixed AEA offset between trajectory start and RGB frame 0
FPS = 20.0
RAW_RES = 2880
PREVIEW_RES = 1408
CROP_OFFSET = 32.0       # 2880 -> crop 32 px each side -> 2816 -> 2x binning -> 1408
GAZE_TOL_US = 150_000    # max |gaze_ts - frame_ts| to accept a gaze sample (gaze is 10 Hz)


def _extract(zip_path, member_suffix, out_dir):
    """Extract the member ending with `member_suffix` from zip to out_dir (flat). Return path or None."""
    if not os.path.exists(zip_path):
        return None
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.filelist:
            if info.filename.endswith(member_suffix):
                info.filename = os.path.basename(info.filename)
                zf.extract(info, out_dir)
                return os.path.join(out_dir, info.filename)
    return None


def _ensure(seq_dir, rec, rel_path, zip_suffix, member_suffix):
    path = os.path.join(seq_dir, rel_path)
    if os.path.exists(path):
        return path
    zip_path = os.path.join(seq_dir, f"AriaEverydayActivities_1.0.0_{rec}_{zip_suffix}.zip")
    return _extract(zip_path, member_suffix, os.path.dirname(path))


def process_sequence(seq_dir, depth_m, device_subtype, overwrite):
    rec = os.path.basename(os.path.normpath(seq_dir))
    out_csv = os.path.join(seq_dir, "gaze_px.csv")
    if os.path.exists(out_csv) and not overwrite:
        print(f"[{rec}] gaze_px.csv exists, skip")
        return

    video = os.path.join(seq_dir, f"AriaEverydayActivities_1.0.0_{rec}_preview_rgb.mp4")
    gaze_csv = _ensure(seq_dir, rec, "mps/eye_gaze/general_eye_gaze.csv", "mps_eye_gaze", "general_eye_gaze.csv")
    traj_csv = _ensure(seq_dir, rec, "mps/slam/closed_loop_trajectory.csv", "mps_slam_trajectories", "closed_loop_trajectory.csv")
    calib_jsonl = _ensure(seq_dir, rec, "mps/slam/online_calibration.jsonl", "mps_slam_calibration", "online_calibration.jsonl")
    for name, p in [("video", video), ("gaze", gaze_csv), ("trajectory", traj_csv), ("calibration", calib_jsonl)]:
        if p is None or not os.path.exists(p):
            print(f"[{rec}] missing {name}, skip")
            return

    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (vw, vh) == (PREVIEW_RES, PREVIEW_RES), f"unexpected preview size {vw}x{vh}"

    gaze = pd.read_csv(gaze_csv).sort_values("tracking_timestamp_us").reset_index(drop=True)
    gaze_ts = gaze["tracking_timestamp_us"].values.astype(np.int64)
    traj_ts0 = int(pd.read_csv(traj_csv, usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].min())

    calibs = mps.read_online_calibration(calib_jsonl)
    calib_ts = np.array([c.tracking_timestamp.total_seconds() * 1e6 for c in calibs], dtype=np.int64)
    # T_device_cpf: CAD-based nominal transform (Gen1 default in projectaria_tools)
    cams0 = {c.get_label(): c for c in calibs[0].camera_calibs}
    dev = ac.DeviceCalibration(camera_calibs=cams0,
                               device_cad_extrinsics=ac.DeviceCadExtrinsics(device_subtype, "camera-slam-left"),
                               device_subtype=device_subtype, origin_label="camera-slam-left")
    T_device_cpf = dev.get_transform_device_cpf()

    rgb_cache = {}

    def rgb_calib_at(j):
        if j not in rgb_cache:
            rgb = [c for c in calibs[j].camera_calibs if c.get_label() == "camera-rgb"][0]
            assert list(rgb.get_image_size()) == [RAW_RES, RAW_RES]
            rgb1408 = rgb.rescale(np.array([PREVIEW_RES, PREVIEW_RES]), PREVIEW_RES / (RAW_RES - 2 * CROP_OFFSET),
                                  np.array([CROP_OFFSET, CROP_OFFSET]))
            T_cam_cpf = rgb.get_transform_device_camera().inverse() @ T_device_cpf
            rgb_cache[j] = (rgb1408, T_cam_cpf)
        return rgb_cache[j]

    frame_idx = np.arange(n_frames, dtype=np.int64)
    frame_ts = traj_ts0 - RGB_OFFSET_US + (frame_idx * (1e6 / FPS)).astype(np.int64)
    gi = np.clip(np.searchsorted(gaze_ts, frame_ts), 1, len(gaze_ts) - 1)
    gi = np.where(np.abs(frame_ts - gaze_ts[gi - 1]) <= np.abs(frame_ts - gaze_ts[gi]), gi - 1, gi)
    ci = np.clip(np.searchsorted(calib_ts, frame_ts), 1, len(calib_ts) - 1)
    ci = np.where(np.abs(frame_ts - calib_ts[ci - 1]) <= np.abs(frame_ts - calib_ts[ci]), ci - 1, ci)

    rows = []
    n_valid = 0
    for k in range(n_frames):
        g = gaze.iloc[gi[k]]
        yaw, pitch = float(g["yaw_rads_cpf"]), float(g["pitch_rads_cpf"])
        gx = gy = np.nan
        valid = False
        if abs(int(frame_ts[k]) - int(gaze_ts[gi[k]])) <= GAZE_TOL_US and np.isfinite(yaw) and np.isfinite(pitch):
            rgb1408, T_cam_cpf = rgb_calib_at(int(ci[k]))
            p_cpf = np.asarray(mps.get_eyegaze_point_at_depth(yaw, pitch, depth_m), dtype=np.float64).reshape(3, 1)
            p_cam = (T_cam_cpf @ p_cpf).flatten()
            uv = rgb1408.project(p_cam)
            if uv is not None:
                x_raw, y_raw = float(uv[0]), float(uv[1])
                # preview mp4 = raw image rotated 90 deg clockwise
                gx, gy = (PREVIEW_RES - 1) - y_raw, x_raw
                valid = True
                n_valid += 1
        rows.append((k, int(frame_ts[k]), gx, gy, PREVIEW_RES, PREVIEW_RES, yaw, pitch, valid))

    pd.DataFrame(rows, columns=["frame_idx", "timestamp_us", "gaze_x", "gaze_y", "ref_w", "ref_h",
                                "yaw_rads_cpf", "pitch_rads_cpf", "valid"]).to_csv(out_csv, index=False)
    print(f"[{rec}] {n_frames} frames, valid gaze {n_valid} ({100.0 * n_valid / n_frames:.1f}%) -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aria_video_base", required=True)
    ap.add_argument("--sequences", nargs="*", default=None, help="sequence names; default: all loc* dirs")
    ap.add_argument("--depth_m", type=float, default=1.0, help="gaze depth used to place the gaze point (m)")
    ap.add_argument("--device_subtype", default="DVT-L", choices=["DVT-S", "DVT-L"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    seqs = a.sequences or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.aria_video_base, "loc*")))
    for s in seqs:
        process_sequence(os.path.join(a.aria_video_base, s), a.depth_m, a.device_subtype, a.overwrite)
