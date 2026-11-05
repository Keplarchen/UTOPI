"""Evaluate UTOPI (and the baselines) on EgoSchema with Qwen2.5-VL (paper Table 1 "ES" column, Table 3/4).

Data (official repository https://github.com/egoschema/EgoSchema, or the lmms-eval/egoschema mirror on the HF hub):
  --questions_json  official questions.json: list of {q_uid, question, "option 0".."option 4", google_drive_id}
  --answers_json    official subset_answers.json: {q_uid: answer index 0-4} for the 500-question subset (optional)
  --video_dir       flat directory with <q_uid>.mp4
  (or --hf_parquet  a lmms-eval/egoschema parquet: columns question_idx, question, video_idx, option[5], answer)
By default only the questions whose answer is known (the subset) are evaluated and accuracy is reported; with
--all_questions every question with a video is run and a submission file {q_uid: answer} is written for the official
validation server (--submission_out).

EgoSchema videos have no eye tracking, IMU or depth, so UTOPI's inputs come from side files next to each video,
named with the video stem (videos share one directory):
  <q_uid>_gaze_px_sum.csv      tools/predict_gaze_sum.py      (SUM-predicted gaze; required for AGTS' gaze prior)
  <q_uid>_pose_megasam.npz     tools/estimate_pose_megasam.py (camera poses + depth from MegaSaM; required for IVOC)
Without a pose file use --pose_source none (AGTS only, Table 4 ablation "AGTS"). Without a gaze file the model falls
back to G = 1 (MLP scores only). tools/preprocess_egoschema.py generates the side files for a list of videos.

Frames: 300 uniformly sampled frames at 532x532 per video as in the paper (--nframes / --resolution).
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from utopi.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # noqa: E402
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--questions_json', default=None)
ap.add_argument('--answers_json', default=None)
ap.add_argument('--hf_parquet', default=None, help='alternative to --questions_json/--answers_json')
ap.add_argument('--video_dir', required=True)
ap.add_argument('--all_questions', action='store_true', help='run every question with a video, not only those with known answers')
ap.add_argument('--submission_out', default=None, help='write {q_uid: predicted index} for the validation server')
ap.add_argument('--results_out', default=None, help='per-question results json (default: next to this script)')
ap.add_argument('--drop_method', default='utopi', choices=[None, 'feature', 'random', 'temporal_ds', 'utopi'])
ap.add_argument('--drop_threshold', type=float, default=0.7, help='utopi: theta; feature: cosine threshold; random/temporal_ds: drop ratio')
ap.add_argument('--drop_absolute', action='store_true', default=False)
ap.add_argument('--token_budget', type=float, default=0.15)
ap.add_argument('--sigma_px', type=float, default=1408.0 / 6, help='AGTS Gaussian std in 1408-pixel units (scaled to the model frame)')
ap.add_argument('--imu_depth', type=float, default=2.0, help='constant depth when no depth file')
ap.add_argument('--gaze_file', default='{stem}_gaze_px_sum.csv')
ap.add_argument('--depth_source', default='auto', choices=['auto', 'file', 'constant'])
ap.add_argument('--depth_file', default='{stem}_pose_megasam.npz', help='npz with frame_indices + depth (MegaSaM output)')
ap.add_argument('--pose_source', default='none', choices=['none', 'auto', 'file'])
ap.add_argument('--pose_file', default='{stem}_pose_megasam.npz')
ap.add_argument('--saliency_ckpt', default=None, help='saliency predictor checkpoint from scripts/train_saliency_predictor.py; required for utopi')
ap.add_argument('--model_name', default='Qwen/Qwen2.5-VL-7B-Instruct', help='HuggingFace model name or local path')
ap.add_argument('--nframes', type=int, default=300)
ap.add_argument('--resolution', type=int, default=532)
ap.add_argument('--max_questions', type=int, default=None)
ap.add_argument('--seed', type=int, default=42, help='question order and option shuffling')
ap.add_argument('--no_shuffle_options', action='store_true', help='keep the original option order (A = option 0)')
ap.add_argument('--torch_seed', type=int, default=1234)
args = ap.parse_args()

# ── questions ──
if args.hf_parquet:
    import pandas as pd
    df = pd.read_parquet(args.hf_parquet)
    questions = [{'q_uid': r['video_idx'], 'question': r['question'], 'options': list(r['option']),
                  'answer': (int(r['answer']) if r.get('answer') not in (None, '', 'nan') and str(r.get('answer')) != 'nan' else None)}
                 for _, r in df.iterrows()]
else:
    if not args.questions_json:
        raise SystemExit('give --questions_json (+ --answers_json) or --hf_parquet')
    raw = json.load(open(args.questions_json))
    answers = json.load(open(args.answers_json)) if args.answers_json else {}
    questions = [{'q_uid': q['q_uid'], 'question': q['question'], 'options': [q[f'option {k}'] for k in range(5)],
                  'answer': answers.get(q['q_uid'])} for q in raw]
n_total = len(questions)
questions = [q for q in questions if os.path.exists(os.path.join(args.video_dir, f"{q['q_uid']}.mp4"))]
n_with_video = len(questions)
if not args.all_questions:
    questions = [q for q in questions if q['answer'] is not None]
random.seed(args.seed)
random.shuffle(questions)
if args.max_questions:
    questions = questions[:args.max_questions]
print(f"EgoSchema: {n_total} questions, {n_with_video} with a video in {args.video_dir}, evaluating {len(questions)} "
      f"({'all' if args.all_questions else 'with known answers'})", flush=True)

# ── model ──
torch.manual_seed(args.torch_seed)
nframes = args.nframes
if args.drop_method == 'temporal_ds':
    nframes = max(2, int(args.nframes * (1 - args.drop_threshold)))
FIXED_RESOLUTION = args.resolution * args.resolution
print("Loading model...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,
                                                            attn_implementation="flash_attention_2",
                                                            max_position_embeddings=131072, device_map="auto")
processor = AutoProcessor.from_pretrained(args.model_name)
processor.tokenizer.model_max_length = 131072
drop_absolute = args.drop_absolute
if args.drop_method == 'utopi':
    if not args.saliency_ckpt:
        ap.error('--saliency_ckpt is required for --drop_method utopi (train one with scripts/train_saliency_predictor.py)')
    model.model.saliency_predictor.load_state_dict(torch.load(args.saliency_ckpt, map_location="cpu"))
    model.model.saliency_predictor.eval()
    m = model.model
    m.utopi_keep_ratio, m.utopi_sigma_px, m.utopi_depth_m = args.token_budget, args.sigma_px, args.imu_depth
    m.utopi_gaze_file, m.utopi_depth_source, m.utopi_depth_file = args.gaze_file, args.depth_source, args.depth_file
    m.utopi_pose_source, m.utopi_pose_file = args.pose_source, args.pose_file
    drop_absolute = True
    print(f"UTOPI: theta={args.drop_threshold} budget={args.token_budget} gaze={args.gaze_file} "
          f"pose={args.pose_source}:{args.pose_file} depth={args.depth_source}:{args.depth_file}", flush=True)
print("Model loaded!", flush=True)

# ── loop ──
results, predictions = [], {}
correct = answered = skipped = 0
drop_ratios = []
dr_file = f"/tmp/drop_ratio_egoschema_{os.getpid()}.jsonl"
t0 = time.time()
for i, q in enumerate(questions):
    video_path = os.path.join(args.video_dir, f"{q['q_uid']}.mp4")
    options, order = list(q['options']), list(range(5))
    if not args.no_shuffle_options:
        random.shuffle(order)
        options = [q['options'][k] for k in order]
    prompt = (f"{q['question']}\n\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\n\n"
              f"Please watch the video and answer with only the letter of the correct option (A/B/C/D/E).")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}", "nframes": nframes,
         "min_pixels": FIXED_RESOLUTION, "max_pixels": FIXED_RESOLUTION},
        {"type": "text", "text": prompt}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        if os.path.exists(dr_file):
            os.remove(dr_file)
        gen = model.generate(**inputs, max_new_tokens=16,
                             drop_method=None if args.drop_method == 'temporal_ds' else args.drop_method,
                             drop_threshold=args.drop_threshold, drop_absolute=drop_absolute,
                             dr_save_path=dr_file, video_path=video_path)
        out = processor.batch_decode([gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()
        dr = None
        if os.path.exists(dr_file):
            dr = json.loads(open(dr_file).readlines()[-1])['ratio']; drop_ratios.append(dr)
        mt = re.search(r'\b([A-E])\b', out.upper())
        if not mt:
            skipped += 1; print(f"[{i+1}/{len(questions)}] SKIPPED (unparsable): {q['q_uid']} -> {out!r}", flush=True); continue
        letter = mt.group(1)
        pred_orig = order[ord(letter) - 65]               # index in the original option order
        predictions[q['q_uid']] = pred_orig
        is_correct = (q['answer'] is not None and pred_orig == q['answer'])
        if q['answer'] is not None:
            answered += 1; correct += int(is_correct)
        results.append({'q_uid': q['q_uid'], 'question': q['question'], 'options_shown': {chr(65 + k): options[k] for k in range(5)},
                        'model_answer': letter, 'predicted_option_index': pred_orig, 'answer_index': q['answer'],
                        'is_correct': is_correct if q['answer'] is not None else None, 'drop_ratio': dr,
                        **({'utopi': getattr(model.model, '_utopi_last_stats', None)} if args.drop_method == 'utopi' else {})})
        acc = f"{100 * correct / answered:.2f}%" if answered else "n/a"
        print(f"[{i+1}/{len(questions)}] {'✓' if is_correct else ('✗' if q['answer'] is not None else '·')} {q['q_uid']} "
              f"pred {letter}->{pred_orig} gt {q['answer']} | drop {100 * (dr or 0):.1f}% | acc {acc} | {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        skipped += 1; print(f"[{i+1}/{len(questions)}] SKIPPED (error): {q['q_uid']}: {e}", flush=True)

tag = f"{args.drop_method}_thr{args.drop_threshold}_nf{nframes}_res{args.resolution}"
if args.drop_method == 'utopi':
    tag += f"_budget{args.token_budget}_pose{args.pose_source}_depth{args.depth_source}"
if args.max_questions:
    tag += f"_max{args.max_questions}"
os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
results_out = args.results_out or os.path.join(REPO_ROOT, 'results', f"egoschema_results_{tag}.json")
json.dump(results, open(results_out, 'w'), indent=2)
if args.submission_out:
    json.dump(predictions, open(args.submission_out, 'w'))
    print(f"submission ({len(predictions)} answers) -> {args.submission_out}")
print("=" * 60)
print(f"Answered with known answer: {answered} | correct {correct} | skipped {skipped}")
if answered:
    print(f"Accuracy (subset): {100 * correct / answered:.2f}%")
if drop_ratios:
    print(f"Average drop ratio: {100 * sum(drop_ratios) / len(drop_ratios):.2f}%")
print(f"Per-question results -> {results_out}")
