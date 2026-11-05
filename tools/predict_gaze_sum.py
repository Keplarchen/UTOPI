"""Predict per-frame gaze points with SUM for videos that have no eye tracking (paper Sec. 4.1: "For videos that do
not have gaze signals available, we estimate the user gaze location using SUM"; the gaze point is the arg-max of the
predicted saliency map).

SUM: Hosseini et al., "SUM: Saliency Unification through Mamba for Visual Attention Modeling", WACV 2025,
https://github.com/Arhosseini77/SUM (needs mamba-ssm / causal-conv1d CUDA kernels; official env torch 2.1 + cu121).
Model loading (one of):
  --sum_root <clone of the SUM repo> --sum_ckpt <net/pre_trained_weights/sum_model.pth>   (official weights, Google Drive)
  --sum_root <clone> --hf_id safe-models/SUM                                                (weights from the HF hub)
Per frame (official preprocessing): resize to 256x256, ImageNet normalisation, condition one-hot (default 1 =
natural scenes / eye-tracking data), forward -> saliency map [256,256]; gaze = arg-max cell centre mapped back to the
original frame pixels. `--blur_sigma` optionally smooths the map before the arg-max (0 = plain arg-max as in the paper).

Output: <video_dir>/gaze_px_sum.csv with the same columns as preprocess_gaze.py (frame_idx, timestamp_us, gaze_x,
gaze_y, valid, ...) plus ref_w / ref_h (the pixel frame the coordinates refer to), source and peak. Point the evaluator
at it with `scripts/eval_egoeye.py --gaze_file gaze_px_sum.csv`. Frame selection: --all | --stride K | --nframes N | --fps F |
--indices (same rules as the other tools; --nframes/--fps match the Qwen processor).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utopi.frames import add_frame_args, read_frames, select_frames, video_info  # noqa: E402

SUM_INPUT = 256


def load_sum(sum_root, sum_ckpt, hf_id, device):
    if sum_root:
        sys.path.insert(0, os.path.abspath(sum_root))
    from net.models.SUM import SUM  # noqa: E402  (SUM repo)
    if hf_id:
        model = SUM.from_pretrained(hf_id)
    else:
        # constructor arguments of net/configs/config_setting.py (model_config)
        model = SUM(num_classes=1, input_channels=3, depths=[2, 2, 27, 2], depths_decoder=[2, 2, 2, 1], drop_path_rate=0.3)
        model.load_state_dict(torch.load(sum_ckpt, map_location="cpu"))
    return model.to(device).eval()


_tf = T.Compose([T.Resize((SUM_INPUT, SUM_INPUT)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def predict_maps(model, frames_bgr, condition, device):
    """frames_bgr: list of HxWx3 uint8 (OpenCV). Returns float32 [B, 256, 256] saliency maps."""
    x = torch.stack([_tf(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames_bgr]).to(device)
    cond = torch.zeros((x.shape[0], 4), device=device)
    cond[:, condition] = 1
    out = model(x, cond)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(x.shape[0], SUM_INPUT, SUM_INPUT).float().cpu().numpy()


def maps_to_gaze(maps, orig_hw, blur_sigma=0.0):
    """arg-max of each 256x256 map -> (x, y) in original frame pixels (cell centres)."""
    H, W = orig_hw
    out = []
    for m in maps:
        if blur_sigma > 0:
            m = cv2.GaussianBlur(m, (0, 0), blur_sigma)
        r, c = np.unravel_index(int(np.argmax(m)), m.shape)
        out.append(((c + 0.5) * W / SUM_INPUT, (r + 0.5) * H / SUM_INPUT, float(m[r, c])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="default <video_dir>/gaze_px_sum.csv")
    add_frame_args(ap)
    ap.add_argument("--sum_root", default=os.environ.get("SUM_ROOT", "../third_party/SUM"), help="clone of the SUM repo")
    ap.add_argument("--sum_ckpt", default=None, help="sum_model.pth (official weights)")
    ap.add_argument("--hf_id", default=None, help="HF hub id, e.g. safe-models/SUM (used instead of --sum_ckpt)")
    ap.add_argument("--condition", type=int, default=1, choices=[0, 1, 2, 3],
                    help="SUM condition: 0 SALICON mouse-tracking, 1 natural eye-tracking (default), 2 e-commerce, 3 UI")
    ap.add_argument("--blur_sigma", type=float, default=0.0, help="Gaussian blur (in 256-map pixels) before the arg-max; 0 = none")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vis", default=None, help="optional mp4 with the predicted gaze drawn on the frames")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    total, fps, orig_hw = video_info(a.video)
    indices = select_frames(total, a, fps)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.video)), "gaze_px_sum.csv")
    print(f"{a.video}: {total} frames @ {fps:.2f} fps, {orig_hw[1]}x{orig_hw[0]}, predicting gaze for {len(indices)} frames")
    if a.dry_run:
        print("selected:", indices[:10], "...")
        return
    if os.path.exists(out) and not a.overwrite:
        print(f"{out} exists, skip (use --overwrite)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not (a.sum_ckpt or a.hf_id):
        raise SystemExit("give --sum_ckpt or --hf_id")
    model = load_sum(a.sum_root, a.sum_ckpt, a.hf_id, device)

    rows, vis_frames = [], []
    for b in range(0, len(indices), a.batch):
        idx = indices[b:b + a.batch]
        frames = read_frames(a.video, idx)
        maps = predict_maps(model, frames, a.condition, device)
        for i, (gx, gy, peak) in zip(idx, maps_to_gaze(maps, orig_hw, a.blur_sigma)):
            rows.append((i, int(i * 1e6 / fps), gx, gy, orig_hw[1], orig_hw[0], True, f"sum_c{a.condition}", peak))
        if a.vis:
            for f, (gx, gy, _) in zip(frames, maps_to_gaze(maps, orig_hw, a.blur_sigma)):
                g = f.copy()
                cv2.circle(g, (int(gx), int(gy)), max(6, orig_hw[1] // 100), (0, 255, 255), 3)
                vis_frames.append(cv2.resize(g, (512, int(512 * orig_hw[0] / orig_hw[1]))))
        print(f"  {min(b + a.batch, len(indices))}/{len(indices)}", flush=True)

    pd.DataFrame(rows, columns=["frame_idx", "timestamp_us", "gaze_x", "gaze_y", "ref_w", "ref_h", "valid", "source", "peak"]).to_csv(out, index=False)
    print(f"saved {out}: {len(rows)} rows")
    if a.vis:
        vw = cv2.VideoWriter(a.vis, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (vis_frames[0].shape[1], vis_frames[0].shape[0]))
        for g in vis_frames:
            vw.write(g)
        vw.release()
        print("vis:", a.vis)


if __name__ == "__main__":
    main()
