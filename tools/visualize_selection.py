"""Render UTOPI token selection: kept tokens (green), foreground tokens removed by IVOC (red), gaze point (yellow).
Usage: python tools/visualize_selection.py <drop_vis_json> <out_dir> [--frames 0,10,20,...]
"""
import argparse, json, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("vis_json"); ap.add_argument("out_dir")
ap.add_argument("--frames", default=None, help="comma-separated token-frame indices; default 8 evenly spaced")
ap.add_argument("--video", default=None, help="override video path (e.g. local copy)")
a = ap.parse_args()
d = json.load(open(a.vis_json))
os.makedirs(a.out_dir, exist_ok=True)
H, W, hp, wp, ts = d['frame_h'], d['frame_w'], d['h_patches'], d['w_patches'], d['token_size']
N = hp * wp; F = len(d['sampled_indices'])
keep = np.zeros(F * N, bool); keep[d['keep_flat_indices']] = True
red = np.zeros(F * N, bool); red[d['redundant_flat_indices']] = True
video = a.video or d['video_path']
cap = cv2.VideoCapture(video)
sel = [int(x) for x in a.frames.split(",")] if a.frames else np.linspace(0, F - 1, 8).astype(int).tolist()
tiles = []
for fi in sel:
    cap.set(cv2.CAP_PROP_POS_FRAMES, d['sampled_indices'][fi]); ok, img = cap.read()
    img = cv2.resize(img, (W, H)); ov = img.copy()
    for k in range(N):
        h, w = divmod(k, wp); x0, y0 = w * ts, h * ts
        if keep[fi * N + k]:
            cv2.rectangle(ov, (x0, y0), (x0 + ts - 1, y0 + ts - 1), (0, 255, 0), -1)
        elif red[fi * N + k]:
            cv2.rectangle(ov, (x0, y0), (x0 + ts - 1, y0 + ts - 1), (0, 0, 255), -1)
    out = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
    gx, gy = d['gaze_model_px'][fi]
    if gx >= 0: cv2.circle(out, (int(gx), int(gy)), 10, (0, 255, 255), 3)
    nk = int(keep[fi * N:(fi + 1) * N].sum()); nr = int(red[fi * N:(fi + 1) * N].sum())
    cv2.putText(out, f"f{fi} keep {nk} ivoc-rm {nr}", (6, 20), 0, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(a.out_dir, f"frame{fi:03d}.jpg"), out); tiles.append(out)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
if len(rows) > 1 and rows[-1].shape[1] != rows[0].shape[1]:
    rows[-1] = np.hstack([rows[-1], np.zeros((H, rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)])
cv2.imwrite(os.path.join(a.out_dir, "grid.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("stats:", json.dumps(d['stats'])); print("saved to", a.out_dir)
