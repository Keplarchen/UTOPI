"""Frame selection shared by the UTOPI preprocessing tools (tools/estimate_depth.py, tools/estimate_pose_megasam.py,
tools/predict_gaze_sum.py, tools/preprocess_egoschema.py). `--nframes` / `--fps` reproduce qwen_vl_utils.smart_nframes exactly, so the stored frames are the
ones the Qwen2.5-VL processor samples for the same setting."""
import math

import cv2
import torch

FRAME_FACTOR = 2  # qwen_vl_utils: frame counts are rounded to a multiple of the temporal patch size


def add_frame_args(ap):
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="every frame")
    g.add_argument("--stride", type=int, default=None, help="every K-th frame")
    g.add_argument("--nframes", type=int, default=None, help="N uniformly spaced frames (qwen_vl_utils nframes rule)")
    g.add_argument("--fps", type=float, default=None, help="target sampling fps (qwen_vl_utils fps rule; Qwen default 2.0)")
    g.add_argument("--indices", type=str, default=None, help="comma-separated frame indices")
    ap.add_argument("--fps_min_frames", type=int, default=4, help="[--fps] lower clamp (qwen_vl_utils FPS_MIN_FRAMES)")
    ap.add_argument("--fps_max_frames", type=int, default=768, help="[--fps] upper clamp (qwen_vl_utils FPS_MAX_FRAMES)")
    return ap


def _round_by_factor(n, f):
    return round(n / f) * f


def uniform_indices(total, n):
    return torch.linspace(0, total - 1, n).round().long().unique().tolist()


def select_frames(total, a, video_fps):
    if a.indices:
        idx = sorted({int(x) for x in a.indices.split(",")})
    elif a.nframes:
        idx = uniform_indices(total, max(FRAME_FACTOR, _round_by_factor(a.nframes, FRAME_FACTOR)))
    elif a.fps:
        min_frames = int(math.ceil(a.fps_min_frames / FRAME_FACTOR) * FRAME_FACTOR)
        max_frames = int(math.floor(min(a.fps_max_frames, total) / FRAME_FACTOR) * FRAME_FACTOR)
        n = total / video_fps * a.fps
        n = min(max(n, min_frames), max_frames)
        idx = uniform_indices(total, int(_round_by_factor(n, FRAME_FACTOR)))
    elif a.stride:
        idx = list(range(0, total, a.stride))
    elif a.all:
        idx = list(range(total))
    else:
        raise SystemExit("choose one of --all / --stride / --nframes / --fps / --indices")
    return [i for i in idx if 0 <= i < total]


def video_info(video):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total, fps, (h, w)


def read_frames(video, indices):
    cap = cv2.VideoCapture(video)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} of {video}")
        frames.append(f)
    cap.release()
    return frames
