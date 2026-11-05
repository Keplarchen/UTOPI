"""Camera pose (and depth) estimation with MegaSaM for videos without device pose tracking (paper App. A.2:
"For a few videos missing IMU or depth signals, we get camera motion and depth using MegaSaM").

This is a thin driver around the official MegaSaM pipeline (https://github.com/mega-sam/mega-sam):
  1. extract the selected video frames as jpgs           (this script)
  2. Depth-Anything (v1, vitl14) relative disparity       Depth-Anything/run_videos.py
  3. UniDepthV2 metric depth                             UniDepth/scripts/demo_mega-sam.py
  4. camera tracking (DROID-SLAM based)                   camera_tracking_scripts/test_demo.py
  5. RAFT optical flow + consistent video depth (CVD)     cvd_opt/preprocess_flow.py, cvd_opt/cvd_opt.py
  6. convert MegaSaM's npz to the UTOPI pose-file format (this script)

MegaSaM needs its own environment (Python 3.10 / CUDA 11.8 / PyTorch 2.0.1, xformers 0.0.22, the `base` CUDA extension
built with `cd base; python setup.py install`) and checkpoints (Depth-Anything/checkpoints/depth_anything_vitl14.pth,
cvd_opt/raft-things.pth, checkpoints/megasam_final.pth); see mega-sam/README.md. Run this script from that
environment with --megasam_root pointing at the clone. Stages 2-5 are executed verbatim through subprocess with the
working directory set to --megasam_root (the official scripts use relative output dirs: reconstructions/, outputs/,
cache_flow/, outputs_cvd/).
--convert_only converts an existing MegaSaM npz without running stages 2-5.

Output npz (default <video_dir>/pose_megasam.npz):
  frame_indices [F], cam_c2w [F,4,4] (OpenCV camera-to-world), K [3,3] (pixels of the original frame size), orig_hw [2],
  depth [F,h,w] float16 (metres, MegaSaM resolution), method.
The model can use it both as pose source (--pose_source file --pose_file pose_megasam.npz) and as depth source
(--depth_source file --depth_file pose_megasam.npz), since the keys are compatible with estimate_depth.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utopi.frames import add_frame_args, read_frames, select_frames, video_info  # noqa: E402


def run(cmd, cwd, env=None):
    print("$", " ".join(cmd), f"  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def convert(megasam_npz, frame_indices, orig_hw, out, method):
    z = np.load(megasam_npz)
    c2w = z["cam_c2w"].astype(np.float32)            # [F,4,4]
    K_rec = np.asarray(z["intrinsic"], dtype=np.float64)  # at MegaSaM's reconstruction resolution
    depths = z["depths"].astype(np.float16)           # [F,h,w]
    imgs = z["images"]
    h_rec, w_rec = imgs.shape[1], imgs.shape[2]
    F = c2w.shape[0]
    if F != len(frame_indices):
        print(f"warning: MegaSaM returned {F} poses for {len(frame_indices)} frames; truncating to the common prefix")
        n = min(F, len(frame_indices))
        c2w, depths, frame_indices = c2w[:n], depths[:n], frame_indices[:n]
    # intrinsics: MegaSaM resizes the input (h0,w0) to ~384x512 area and crops to a multiple of 8;
    # K_rec is expressed in that cropped image. Map back to original pixels (crop keeps the origin, so only scale).
    H0, W0 = orig_hw
    sx, sy = W0 / w_rec, H0 / h_rec
    K = np.array([[K_rec[0, 0] * sx, 0, K_rec[0, 2] * sx], [0, K_rec[1, 1] * sy, K_rec[1, 2] * sy], [0, 0, 1]], dtype=np.float32)
    np.savez_compressed(out, frame_indices=np.asarray(frame_indices, dtype=np.int64), cam_c2w=c2w, K=K,
                        orig_hw=np.array([H0, W0], dtype=np.int64), depth=depths,
                        megasam_hw=np.array([h_rec, w_rec], dtype=np.int64), method=np.array(method))
    print(f"saved {out}: {len(frame_indices)} frames, K(orig px) fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}, "
          f"depth median {float(np.median(depths.astype(np.float32))):.2f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="default <video_dir>/pose_megasam.npz")
    add_frame_args(ap)
    ap.add_argument("--megasam_root", default=os.environ.get("MEGASAM_ROOT", "../third_party/mega-sam"))
    ap.add_argument("--work_dir", default=None, help="where frames are written (default <megasam_root>/utopi_work)")
    ap.add_argument("--scene_name", default=None, help="MegaSaM scene name (default: video basename)")
    ap.add_argument("--skip_cvd", action="store_true", help="stop after camera tracking (use outputs/<scene>_droid.npz)")
    ap.add_argument("--opt_focal", action="store_true", help="let MegaSaM optimise the focal length (passed to test_demo.py)")
    ap.add_argument("--convert_only", default=None, help="path to an existing MegaSaM npz; skip stages 2-5")
    ap.add_argument("--python", default=sys.executable, help="python of the MegaSaM environment")
    a = ap.parse_args()

    total, fps, orig_hw = video_info(a.video)
    indices = select_frames(total, a, fps)
    scene = a.scene_name or os.path.splitext(os.path.basename(a.video))[0]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.video)), "pose_megasam.npz")

    if a.convert_only:
        convert(a.convert_only, indices, orig_hw, out, "megasam:convert_only")
        return

    root = os.path.abspath(a.megasam_root)
    for req in ["camera_tracking_scripts/test_demo.py", "cvd_opt/cvd_opt.py", "Depth-Anything/run_videos.py",
                "UniDepth/scripts/demo_mega-sam.py", "checkpoints/megasam_final.pth"]:
        if not os.path.exists(os.path.join(root, req)):
            raise SystemExit(f"MegaSaM file missing: {os.path.join(root, req)} (see docstring for setup)")
    work = os.path.abspath(a.work_dir or os.path.join(root, "utopi_work"))
    img_dir = os.path.join(work, scene)
    shutil.rmtree(img_dir, ignore_errors=True)
    os.makedirs(img_dir)
    # stage 1: frames as sorted jpgs (MegaSaM globs *.jpg in sorted order)
    for k, f in enumerate(read_frames(a.video, indices)):
        cv2.imwrite(os.path.join(img_dir, f"{k:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    json.dump({"video": os.path.abspath(a.video), "frame_indices": indices}, open(os.path.join(img_dir, "frame_indices.json"), "w"))
    print(f"stage 1: wrote {len(indices)} frames to {img_dir}")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}/UniDepth:{root}:" + env.get("PYTHONPATH", "")
    py = a.python
    # stage 2-3: mono depth priors (official scripts)
    run([py, "Depth-Anything/run_videos.py", "--encoder", "vitl", "--load-from", "Depth-Anything/checkpoints/depth_anything_vitl14.pth",
         "--img-path", img_dir, "--outdir", f"Depth-Anything/video_visualization/{scene}"], root, env)
    run([py, "UniDepth/scripts/demo_mega-sam.py", "--scene-name", scene, "--img-path", img_dir, "--outdir", "UniDepth/outputs"], root, env)
    # stage 4: camera tracking
    cmd = [py, "camera_tracking_scripts/test_demo.py", f"--datapath={img_dir}", "--weights=checkpoints/megasam_final.pth",
           "--scene_name", scene, "--mono_depth_path", f"{root}/Depth-Anything/video_visualization",
           "--metric_depth_path", f"{root}/UniDepth/outputs", "--disable_vis"]
    if a.opt_focal:
        cmd.append("--opt_focal")
    run(cmd, root, env)
    src = os.path.join(root, "outputs", f"{scene}_droid.npz")
    method = "megasam:camera_tracking"
    if not a.skip_cvd:
        # stage 5: flow + consistent video depth
        run([py, "cvd_opt/preprocess_flow.py", f"--datapath={img_dir}", "--model=cvd_opt/raft-things.pth", "--scene_name", scene, "--mixed_precision"], root, env)
        run([py, "cvd_opt/cvd_opt.py", "--scene_name", scene, "--w_grad", "2.0", "--w_normal", "5.0"], root, env)
        src = os.path.join(root, "outputs_cvd", f"{scene}_sgd_cvd_hr.npz")
        method = "megasam:cvd"
    # stage 6
    convert(src, indices, orig_hw, out, method)


if __name__ == "__main__":
    main()
