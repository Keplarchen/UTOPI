"""Generate UTOPI side files for EgoSchema videos (flat directory of <q_uid>.mp4):
  <q_uid>_gaze_px_sum.csv   via predict_gaze_sum.py   (SUM-predicted gaze)
  <q_uid>_pose_megasam.npz  via estimate_pose_megasam.py (MegaSaM poses + depth; needs the MegaSaM environment)
Frame selection defaults to --nframes 300 (the evaluation setting); pass any option of utopi/frames.py.

Example:
  python preprocess_egoschema.py --video_dir /data/egoschema/videos --questions_json questions.json --answers_json subset_answers.json \\
      --gaze --sum_root ../third_party/SUM --hf_id safe-models/SUM
  python preprocess_egoschema.py --video_dir ... --pose --megasam_root ../third_party/mega-sam --python /path/to/megasam/env/python
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--video_dir', required=True)
ap.add_argument('--questions_json', default=None, help='restrict to the videos of these questions')
ap.add_argument('--answers_json', default=None, help='with --questions_json: only the questions with known answers (subset)')
ap.add_argument('--uids', default=None, help='comma-separated q_uids (overrides the json filters)')
ap.add_argument('--gaze', action='store_true', help='run predict_gaze_sum.py')
ap.add_argument('--pose', action='store_true', help='run estimate_pose_megasam.py')
ap.add_argument('--nframes', type=int, default=300)
ap.add_argument('--sum_root', default=None); ap.add_argument('--sum_ckpt', default=None); ap.add_argument('--hf_id', default=None)
ap.add_argument('--megasam_root', default=None); ap.add_argument('--python', default=sys.executable, help='[pose] python of the MegaSaM env')
ap.add_argument('--skip_cvd', action='store_true')
ap.add_argument('--overwrite', action='store_true')
a = ap.parse_args()

if a.uids:
    uids = a.uids.split(',')
elif a.questions_json:
    qs = json.load(open(a.questions_json))
    if a.answers_json:
        known = set(json.load(open(a.answers_json)))
        qs = [q for q in qs if q['q_uid'] in known]
    uids = sorted({q['q_uid'] for q in qs})
else:
    uids = sorted(os.path.splitext(f)[0] for f in os.listdir(a.video_dir) if f.endswith('.mp4'))
videos = [(u, os.path.join(a.video_dir, f"{u}.mp4")) for u in uids]
videos = [(u, v) for u, v in videos if os.path.exists(v)]
print(f"{len(videos)} videos", flush=True)

for k, (u, v) in enumerate(videos):
    print(f"== [{k+1}/{len(videos)}] {u}", flush=True)
    if a.gaze:
        cmd = [sys.executable, os.path.join(HERE, 'predict_gaze_sum.py'), '--video', v, '--nframes', str(a.nframes),
               '--out', os.path.join(a.video_dir, f"{u}_gaze_px_sum.csv")]
        if a.sum_root: cmd += ['--sum_root', a.sum_root]
        if a.sum_ckpt: cmd += ['--sum_ckpt', a.sum_ckpt]
        if a.hf_id: cmd += ['--hf_id', a.hf_id]
        if a.overwrite: cmd.append('--overwrite')
        subprocess.run(cmd, check=True)
    if a.pose:
        cmd = [a.python, os.path.join(HERE, 'estimate_pose_megasam.py'), '--video', v, '--nframes', str(a.nframes),
               '--out', os.path.join(a.video_dir, f"{u}_pose_megasam.npz")]
        if a.megasam_root: cmd += ['--megasam_root', a.megasam_root]
        if a.skip_cvd: cmd.append('--skip_cvd')
        subprocess.run(cmd, check=True)
print("done")
