"""Monocular video depth estimation for UTOPI/IVOC (paper App. Eq. 7: depth d at each token).

Implements the mono-depth prior stage of MegaSaM (Li et al., CVPR 2025): UniDepthV2 metric depth +
DepthAnythingV2 relative disparity, aligned with a global scale/shift (median over frames), i.e. the
same recipe as mega-sam/mono_depth_scripts, except that the final depth is kept in metres (MegaSaM
additionally normalises the scale for its SLAM stage). Only this depth stage is used: for AEA the camera
motion comes from the device trajectory (MPS). For MegaSaM camera tracking see estimate_pose_megasam.py.

Output: one .npz per video (default <video_dir>/depth_mono.npz) with
  frame_indices : int64 [F]      video frame index of each stored depth map
  depth         : float16 [F,h,w] depth in metres, stored at --store_res (aspect kept), 0 = invalid
  orig_hw       : int64 [2]      original frame size
  method, align_scale, align_shift, unidepth_focal_px (median) : metadata
The model (token_drop 'utopi') looks up the sampled frames by index and pools each map to the token
grid, so any frame-sampling scheme works as long as the needed frames are stored.

Frame selection (pick one): --all | --stride K | --nframes N | --fps F | --indices i,j,k
  --nframes / --fps reproduce qwen_vl_utils.smart_nframes (frame count rounded to a multiple of 2, fps mode clamped
  to [--fps_min_frames, --fps_max_frames]) followed by linspace(0, T-1, n).round(), i.e. exactly the frames the
  Qwen2.5-VL processor sees for the same `nframes` / `fps` setting.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utopi.frames import add_frame_args, read_frames, select_frames, video_info  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UNIDEPTH = os.path.join(ROOT, "..", "third_party", "UniDepth")
DEFAULT_DA = os.path.join(ROOT, "..", "third_party", "Depth-Anything")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="output npz (default: <video_dir>/depth_mono.npz)")
    add_frame_args(ap)
    ap.add_argument("--dry_run", action="store_true", help="only print the selected frame indices")
    ap.add_argument("--store_res", type=int, default=176, help="longer side of the stored depth maps")
    ap.add_argument("--source", choices=["aligned", "unidepth"], default="aligned",
                    help="aligned = DA disparity aligned to UniDepth (MegaSaM prior); unidepth = UniDepthV2 only")
    ap.add_argument("--da_ckpt", required=True,
                    help="DepthAnythingV2 vitl checkpoint depth_anything_v2_vitl.pth (HF depth-anything/Depth-Anything-V2-Large)")
    ap.add_argument("--unidepth_ckpt", required=True,
                    help="local dir with config.json + model.safetensors of HF lpiccinelli/unidepth-v2-vitl14 at revision "
                         "1d0d3c52f60b5164629d279bb9a7546458e6dcc4 (the vendored UniDepth code expects that config/weights; "
                         "the current HF revision uses a different config format)")
    ap.add_argument("--unidepth_root", default=DEFAULT_UNIDEPTH)
    ap.add_argument("--da_root", default=DEFAULT_DA)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vis", default=None, help="optional side-by-side rgb|depth mp4")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    out = a.out or os.path.join(os.path.dirname(a.video), "depth_mono.npz")
    if os.path.exists(out) and not a.overwrite and not a.dry_run:
        print(f"{out} exists, skip (use --overwrite)")
        return
    sys.path.insert(0, a.unidepth_root)
    sys.path.insert(0, a.da_root)
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose
    from unidepth.models import UniDepthV2

    total, video_fps, _ = video_info(a.video)
    indices = select_frames(total, a, video_fps)
    print(f"{a.video}: {total} frames @ {video_fps:.2f} fps, estimating {len(indices)}")
    if a.dry_run:
        print("selected:", indices[:10], "...", indices[-3:] if len(indices) > 10 else "")
        return

    dev = "cuda"
    print("Loading UniDepthV2...")
    ud = UniDepthV2.from_pretrained(a.unidepth_ckpt).to(dev).eval()
    da = None
    if a.source == "aligned":
        print("Loading DepthAnythingV2 (vitl)...")
        da = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]).to(dev).eval()
        da.load_state_dict(torch.load(a.da_ckpt, map_location="cpu"), strict=True)
        da_tf = Compose([
            Resize(width=768, height=768, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method="upper_bound", image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    ud_depths, da_disps, focals, scales, shifts = [], [], [], [], []
    orig_hw = None
    for b in range(0, len(indices), a.batch):
        frames = read_frames(a.video, indices[b:b + a.batch])
        if orig_hw is None:
            orig_hw = frames[0].shape[:2]
        # UniDepthV2: metric depth per frame (also predicts a pinhole focal)
        for f in frames:
            rgb = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            with torch.no_grad():
                pred = ud.infer(rgb)
            ud_depths.append(pred["depth"][0, 0].float().cpu().numpy())
            focals.append(float(pred["intrinsics"][0, 0, 0]))
        if da is not None:
            x = torch.stack([torch.from_numpy(da_tf({"image": cv2.cvtColor(f, cv2.COLOR_BGR2RGB) / 255.0})["image"]) for f in frames]).to(dev)
            with torch.no_grad():
                d = da(x)
            for j, f in enumerate(frames):
                h, w = ud_depths[b + j].shape
                disp = F.interpolate(d[j:j + 1][None], (h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
                da_disps.append(disp)
                # per-frame scale/shift of DA disparity to UniDepth disparity (MegaSaM recipe)
                md = ud_depths[b + j]
                gt_disp = 1.0 / (md + 1e-8)
                gt_disp[(md < 2.0) & (disp < 0.02)] = 1e-2
                gt_ms = gt_disp - np.median(gt_disp) + 1e-8
                da_ms = disp - np.median(disp) + 1e-8
                s = float(np.median(gt_ms / da_ms))
                t = float(np.median(gt_disp - s * disp))
                scales.append(s); shifts.append(t)
        print(f"  {min(b + a.batch, len(indices))}/{len(indices)}", flush=True)

    align_scale = align_shift = float("nan")
    if a.source == "aligned":
        # global scale/shift: the frame whose scale*shift is closest to the median (MegaSaM)
        prod = np.array(scales) * np.array(shifts)
        k = int(np.argmin(np.abs(prod - np.median(prod))))
        align_scale, align_shift = scales[k], shifts[k]
        depths = [np.clip(1.0 / np.maximum(align_scale * dd + align_shift, 1e-4), 1e-2, 1e2) for dd in da_disps]
    else:
        depths = [np.clip(md, 1e-2, 1e2) for md in ud_depths]

    # store at reduced resolution (area average), float16, aspect kept
    H, W = depths[0].shape
    sc = a.store_res / max(H, W)
    hs, ws = max(1, round(H * sc)), max(1, round(W * sc))
    stored = np.stack([cv2.resize(d, (ws, hs), interpolation=cv2.INTER_AREA) for d in depths]).astype(np.float16)
    np.savez_compressed(out, frame_indices=np.array(indices, dtype=np.int64), depth=stored,
                        orig_hw=np.array(orig_hw, dtype=np.int64), method=np.array(f"megasam_monodepth_prior:{a.source}"),
                        align_scale=np.array(align_scale), align_shift=np.array(align_shift),
                        unidepth_focal_px=np.array(float(np.median(focals))))
    d_all = stored.astype(np.float32)
    print(f"saved {out}: {stored.shape}, depth median {np.median(d_all):.2f} m, p5 {np.percentile(d_all, 5):.2f}, p95 {np.percentile(d_all, 95):.2f}, "
          f"unidepth focal(px, at {orig_hw[1]}x{orig_hw[0]}) {np.median(focals):.0f}")

    if a.vis:
        frames = read_frames(a.video, indices[:: max(1, len(indices) // 60)])
        vw = cv2.VideoWriter(a.vis, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (2 * 352, 352))
        for f, k in zip(frames, range(0, len(indices), max(1, len(indices) // 60))):
            d = d_all[k]
            lo, hi = np.percentile(d, 2), np.percentile(d, 98)
            dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            dc = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)
            vw.write(np.hstack([cv2.resize(f, (352, 352)), cv2.resize(dc, (352, 352))]))
        vw.release()
        print("vis:", a.vis)


if __name__ == "__main__":
    main()
