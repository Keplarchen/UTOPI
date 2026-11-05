from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import time
from datetime import datetime
import torch
import sys
import os
import json
import re
import random
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from utopi.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Parse command line arguments
parser = argparse.ArgumentParser(description='VQA evaluation with various token drop methods')
parser.add_argument('--drop_method', type=str, default='utopi',
                    choices=[None, 'feature', 'random', 'temporal_ds', 'utopi'],
                    help='utopi = AGTS (gaze x saliency MLP) -> IVOC (paper Sec. 3); feature = DTD baseline; '
                         'random = Random Drop baseline; temporal_ds = Frame Subsampling (default: utopi)')
parser.add_argument('--drop_threshold', type=float, default=0.85,
                    help='Drop threshold (default: 0.85)')
parser.add_argument('--drop_absolute', action='store_true', default=False,
                    help='Use absolute mode for dropping (default: relative mode)')
parser.add_argument('--nframes', type=int, default=300,
                    help='Number of frames to sample from video (default: 300)')
parser.add_argument('--resolution', type=int, default=532,
                    help='Fixed resolution side length, must be multiple of 28 (default: 532)')
parser.add_argument('--drop_vis_dir', type=str, default=None,
                    help='Directory to save per-video token drop details for visualization (default: None = disabled)')
parser.add_argument('--saliency_ckpt', type=str, default=None,
                    help='Saliency predictor (MLP) checkpoint written by scripts/train_saliency_predictor.py; required for --drop_method utopi')
parser.add_argument('--token_budget', type=float, default=0.15,
                    help='[utopi] fraction of visual tokens kept per video; rho is adapted to hit it exactly (default: 0.15)')
parser.add_argument('--sigma_px', type=float, default=1408.0 / 6,
                    help='[utopi] Gaussian std of the gaze prior (Eq. 1), in 1408x1408 preview-video pixels (default: 1408/6)')
parser.add_argument('--imu_depth', type=float, default=2.0,
                    help='[utopi] constant depth d (m) for IVOC re-projection, Eq. 7, used when no depth map (default: 2.0)')
parser.add_argument('--depth_source', type=str, default='file', choices=['file', 'auto', 'constant'],
                    help='[utopi] per-token depth for Eq. 7: file = require <video_dir>/depth_mono.npz '
                         '(run estimate_depth.py first), auto = use it if present else --imu_depth, '
                         'constant = always --imu_depth (default: file)')
parser.add_argument('--depth_file', type=str, default='depth_mono.npz',
                    help='[utopi] depth npz file name inside each video directory (default: depth_mono.npz)')
parser.add_argument('--pose_source', type=str, default='mps', choices=['mps', 'auto', 'file', 'none'],
                    help='[utopi] camera poses for Eq. 7-10: mps = AEA device trajectory (default), file = '
                         '<video_dir>/--pose_file from estimate_pose_megasam.py, auto = file if present, '
                         'none = skip IVOC (AGTS only)')
parser.add_argument('--gaze_file', type=str, default='gaze_px.csv',
                    help='[utopi] gaze csv inside each video directory: gaze_px.csv (recorded, preprocess_gaze.py) or '
                         'gaze_px_sum.csv (SUM-predicted, predict_gaze_sum.py); missing file = no gaze prior')
parser.add_argument('--pose_file', type=str, default='pose_megasam.npz',
                    help='[utopi] pose npz file name inside each video directory (default: pose_megasam.npz)')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                    help='HuggingFace model name or local path (default: Qwen/Qwen2.5-VL-7B-Instruct)')
parser.add_argument('--max_questions', type=int, default=None,
                    help='Maximum number of VQA questions to evaluate (default: all)')
parser.add_argument('--vqa_json', type=str,
                    default=os.path.join(REPO_ROOT, 'data', 'egoeye', 'VQA.json'),
                    help='Path to the EgoEye annotations (default: data/egoeye/VQA.json)')
parser.add_argument('--aria_video_base', type=str, required=True,
                    help='Root directory of the AriaEverydayActivities sequences (<root>/<sequence>/...)')
parser.add_argument('--nymeria_video_base', type=str, default=None,
                    help='Root directory of the Nymeria videos (<root>/<video>/<video>.mp4); if not given, Nymeria questions are skipped')
parser.add_argument('--exclude_train', type=str, default=None,
                    help='split.json written by train_saliency_predictor.py (or utopi/split.py): questions used to train the '
                         'saliency MLP are skipped, so the evaluation runs on held-out questions only (paper Sec. 4.1)')
parser.add_argument('--seed', type=int, default=42,
                    help='Evaluation seed: question shuffle and option order (keep identical across methods)')
parser.add_argument('--torch_seed', type=int, default=1234,
                    help='Torch seed (affects the stochastic random drop)')
args = parser.parse_args()

# Model parameters
DROP_METHOD = args.drop_method
DROP_THRESHOLD = args.drop_threshold
DROP_ABSOLUTE = args.drop_absolute
if DROP_METHOD == 'utopi':
    DROP_ABSOLUTE = True  # drop_threshold is theta of Eq. 5 (absolute cosine threshold)

# Video sampling parameters
VIDEO_NFRAMES = args.nframes
if DROP_METHOD == 'temporal_ds':
    VIDEO_NFRAMES = max(1, int(args.nframes * (1 - DROP_THRESHOLD)))
    print(f"temporal_ds: reducing frames from {args.nframes} to {VIDEO_NFRAMES} (threshold={DROP_THRESHOLD})")
# Fixed resolution (must be multiples of 28)
FIXED_RESOLUTION = args.resolution * args.resolution

# Paths
VQA_JSON_PATH = args.vqa_json
ARIA_VIDEO_BASE = args.aria_video_base
NYMERIA_VIDEO_BASE = args.nymeria_video_base

# Torch-level seed (matters for the stochastic 'random' drop method)
torch.manual_seed(args.torch_seed)

DROP_VIS_DIR = args.drop_vis_dir
if DROP_VIS_DIR:
    os.makedirs(DROP_VIS_DIR, exist_ok=True)
_COLLECT_VIS = bool(DROP_VIS_DIR)

MAX_QUESTIONS = args.max_questions

print(f"Config: drop_method={DROP_METHOD}, threshold={DROP_THRESHOLD}, absolute={DROP_ABSOLUTE}, "
      f"nframes={VIDEO_NFRAMES}, resolution={args.resolution}x{args.resolution}"
      + (f", token_budget={args.token_budget}, sigma_px={args.sigma_px:.1f}, imu_depth={args.imu_depth}" if DROP_METHOD == 'utopi' else "")
      + (f", drop_vis_dir={DROP_VIS_DIR}" if DROP_VIS_DIR else "")
      + (f", max_questions={MAX_QUESTIONS}" if MAX_QUESTIONS else ""))

# Load model
MODEL_NAME = args.model_name
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    max_position_embeddings=131072,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.model_max_length = 131072
# Enable vis data collection only when --drop_vis_dir is set
if hasattr(model, 'model'):
    model.model._collect_drop_vis = _COLLECT_VIS
# Load the AGTS saliency predictor (MLP)
if args.drop_method == 'utopi':
    if not args.saliency_ckpt:
        parser.error('--saliency_ckpt is required for --drop_method utopi (train one with scripts/train_saliency_predictor.py)')
    sal_ckpt = torch.load(args.saliency_ckpt, map_location="cpu")
    model.model.saliency_predictor.load_state_dict(sal_ckpt)
    model.model.saliency_predictor.eval()
    print(f"Saliency head loaded: ckpt={args.saliency_ckpt}")
    model.model.utopi_keep_ratio = args.token_budget
    model.model.utopi_sigma_px = args.sigma_px
    model.model.utopi_depth_m = args.imu_depth
    model.model.utopi_depth_source = args.depth_source
    model.model.utopi_depth_file = args.depth_file
    model.model.utopi_pose_source = args.pose_source
    model.model.utopi_pose_file = args.pose_file
    model.model.utopi_gaze_file = args.gaze_file
    print(f"UTOPI: theta={DROP_THRESHOLD}, token_budget={args.token_budget}, sigma_px={args.sigma_px:.1f}, depth={args.imu_depth}")
print("Model loaded!\n")

# Load VQA data
print("Loading VQA.json...")
with open(VQA_JSON_PATH, 'r') as f:
    vqa_data = json.load(f)
print(f"Total questions in VQA.json: {len(vqa_data)}")
for _i, _q in enumerate(vqa_data):
    _q['_orig_idx'] = _i  # position in the original VQA.json (used by --exclude_train)

# Train / evaluation separation (paper Sec. 4.1: MLP training questions are not evaluated)
EXCLUDE_TRAIN = None
if args.exclude_train:
    from utopi.split import load_split
    _split = load_split(args.exclude_train, vqa_data)
    EXCLUDE_TRAIN = _split['train_set']
    print(f"Excluding {len(EXCLUDE_TRAIN)} MLP training questions listed in {args.exclude_train} "
          f"(seed {_split['seed']}, eval_size {_split['eval_size']}); {len(vqa_data) - len(EXCLUDE_TRAIN)} questions eligible.")

# Shuffle questions randomly
random.seed(args.seed)
random.shuffle(vqa_data)
print("Questions shuffled randomly.\n")
excluded_train_count = 0

# Statistics
total_questions = 0
correct_count = 0
wrong_count = 0
skipped_count = 0

# Per-question results log
question_results = []

# Drop ratio statistics
drop_ratios = []
drop_ratio_file = f"/tmp/drop_ratio_temp_{os.getpid()}.jsonl"

# Clear previous drop ratio file if exists
if os.path.exists(drop_ratio_file):
    os.remove(drop_ratio_file)

# Process each question
for idx, item in enumerate(vqa_data):
    # Early stop if max_questions reached
    if MAX_QUESTIONS is not None and total_questions >= MAX_QUESTIONS:
        print(f"\nReached max_questions={MAX_QUESTIONS}, stopping early.")
        break

    if EXCLUDE_TRAIN is not None and item['_orig_idx'] in EXCLUDE_TRAIN:
        excluded_train_count += 1
        continue

    video_name = item['video_name']
    question = item['question']
    options = item['options']
    correct_answer_idx = item['correct_answer']

    # Shuffle options
    indices = list(range(5))
    random.shuffle(indices)
    options = [options[i] for i in indices]
    correct_answer_idx = indices.index(correct_answer_idx)

    # Construct video path based on dataset
    if video_name[0].isdigit():
        # Nymeria video
        if NYMERIA_VIDEO_BASE is None:
            skipped_count += 1
            continue
        video_path = os.path.join(NYMERIA_VIDEO_BASE, video_name, f"{video_name}.mp4")
    else:
        # AriaEveryday video
        video_path = os.path.join(ARIA_VIDEO_BASE, video_name, f"AriaEverydayActivities_1.0.0_{video_name}_preview_rgb.mp4")

    # Check if video exists
    if not os.path.exists(video_path):
        skipped_count += 1
        print(f"[{idx+1}/{len(vqa_data)}] SKIPPED (video not found): {video_path}")
        continue

    # Construct prompt with question and options
    prompt = f"{question}\n\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\n\nPlease watch the video and answer with only the letter of the correct option (A/B/C/D/E)."

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "nframes": VIDEO_NFRAMES,  # Fixed number of frames
                    "min_pixels": FIXED_RESOLUTION,  # Fixed resolution (min = max for square)
                    "max_pixels": FIXED_RESOLUTION,  # Fixed resolution
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    try:
        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Get actual sampled frames and resolution
        actual_sampled_frames = None
        actual_resolution = None
        if video_inputs is not None and len(video_inputs) > 0:
            # video_inputs[0] has shape (nframes, C, H, W)
            actual_sampled_frames = video_inputs[0].shape[0]
            height = video_inputs[0].shape[2]
            width = video_inputs[0].shape[3]
            actual_resolution = f"{width}x{height}"

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate answer
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            drop_method=None if DROP_METHOD == 'temporal_ds' else DROP_METHOD,
            drop_threshold=DROP_THRESHOLD,
            drop_absolute=DROP_ABSOLUTE,
            dr_save_path=drop_ratio_file,
            video_path=video_path,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Read drop ratio from file
        current_drop_ratio = None
        if os.path.exists(drop_ratio_file):
            try:
                with open(drop_ratio_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        drop_info = json.loads(last_line)
                        current_drop_ratio = drop_info['ratio']
                        drop_ratios.append(current_drop_ratio)
            except Exception as e:
                print(f"  Warning: Failed to read drop ratio: {e}")

        # Save per-video token drop details for visualization
        # token_drop() lives on model.model (Qwen2_5_VLModel), not the top-level wrapper
        _inner = model.model if hasattr(model, 'model') else model
        if DROP_VIS_DIR and hasattr(_inner, '_imu_drop_details') and _inner._imu_drop_details is not None:
            vis_data = _inner._imu_drop_details
            vis_filename = f"{video_name}_q{idx}.json"
            vis_path = os.path.join(DROP_VIS_DIR, vis_filename)
            with open(vis_path, 'w') as f:
                json.dump(vis_data, f)
            print(f"  Saved drop vis details: {vis_path}")
            _inner._imu_drop_details = None

        # Parse answer (extract A/B/C/D/E)
        output_text = output_text.strip()
        predicted_letter = None

        # Try to find A, B, C, D, or E in the output
        match = re.search(r'\b([A-E])\b', output_text.upper())
        if match:
            predicted_letter = match.group(1)

        if predicted_letter is None:
            skipped_count += 1
            print(f"[{idx+1}/{len(vqa_data)}] SKIPPED (could not parse answer): {video_name}")
            print(f"  Model output: {output_text}")
            continue

        # Convert letter to index
        letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        predicted_idx = letter_to_idx[predicted_letter]

        # Check if correct
        is_correct = (predicted_idx == correct_answer_idx)
        total_questions += 1

        if is_correct:
            correct_count += 1
            result = "✓ CORRECT"
        else:
            wrong_count += 1
            result = "✗ WRONG"

        # Log per-question result
        correct_letter = chr(65 + correct_answer_idx)
        question_results.append({
            "index": idx,
            "video_name": video_name,
            "question": question,
            "options": {chr(65+i): opt for i, opt in enumerate(options)},
            "model_answer": predicted_letter,
            "correct_answer": correct_letter,
            "is_correct": is_correct,
            "orig_idx": item['_orig_idx'],
            **({"utopi": getattr(model.model, '_utopi_last_stats', None)} if DROP_METHOD == 'utopi' else {}),
        })

        # Calculate accuracy
        accuracy = correct_count / total_questions * 100 if total_questions > 0 else 0

        # Calculate average drop ratio
        avg_drop_ratio = sum(drop_ratios) / len(drop_ratios) * 100 if drop_ratios else 0

        # Print progress
        print(f"[{idx+1}/{len(vqa_data)}] {result}: {video_name}")
        print(f"  Question: {question}")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Model answer: {predicted_letter} | Correct answer: {chr(65 + correct_answer_idx)}")

        # Print sampled frames and resolution info
        if actual_sampled_frames is not None:
            print(f"  Sampled: {actual_sampled_frames} frames @ {actual_resolution}")

        # Print drop ratio info
        if current_drop_ratio is not None:
            print(f"  Drop ratio: {current_drop_ratio * 100:.2f}% | Average drop ratio: {avg_drop_ratio:.2f}%")

        print(f"  Progress: {total_questions} answered | {correct_count} correct | {wrong_count} wrong | {skipped_count} skipped")
        print(f"  Current accuracy: {accuracy:.2f}%\n")

    except Exception as e:
        skipped_count += 1
        print(f"[{idx+1}/{len(vqa_data)}] SKIPPED (error): {video_name}")
        print(f"  Error: {str(e)}\n")
        continue

# Save per-question results log
results_log_name = f"vqa_results_{DROP_METHOD}_thr{DROP_THRESHOLD}_nf{VIDEO_NFRAMES}_res{args.resolution}"
if DROP_METHOD == 'utopi':
    results_log_name += f"_budget{args.token_budget}_sigma{args.sigma_px:.0f}_d{args.imu_depth}"
    if args.depth_source != 'constant':
        results_log_name += f"_depth{args.depth_source}"
    if args.pose_source != 'mps':
        results_log_name += f"_pose{args.pose_source}"
    if args.gaze_file != 'gaze_px.csv':
        results_log_name += f"_gaze{os.path.splitext(args.gaze_file)[0]}"
if EXCLUDE_TRAIN is not None:
    results_log_name += "_heldout"
if MAX_QUESTIONS:
    results_log_name += f"_max{MAX_QUESTIONS}"
results_log_name += ".json"
os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
results_log_path = os.path.join(REPO_ROOT, 'results', results_log_name)
with open(results_log_path, 'w') as f:
    json.dump(question_results, f, indent=2, ensure_ascii=False)
print(f"Per-question results saved to: {results_log_path}")

# Final statistics
print("="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Total questions in dataset: {len(vqa_data)}")
print(f"Questions answered: {total_questions}")
print(f"Questions skipped: {skipped_count}")
if EXCLUDE_TRAIN is not None:
    print(f"Training questions excluded (--exclude_train): {excluded_train_count}")
print(f"\nCorrect: {correct_count}")
print(f"Wrong: {wrong_count}")
print(f"Final Accuracy: {correct_count / total_questions * 100:.2f}%" if total_questions > 0 else "N/A")

# Drop ratio statistics
if drop_ratios:
    avg_drop_ratio = sum(drop_ratios) / len(drop_ratios) * 100
    min_drop_ratio = min(drop_ratios) * 100
    max_drop_ratio = max(drop_ratios) * 100
    print(f"\nDrop Ratio Statistics:")
    print(f"  Average: {avg_drop_ratio:.2f}%")
    print(f"  Min: {min_drop_ratio:.2f}%")
    print(f"  Max: {max_drop_ratio:.2f}%")
    print(f"  Total samples: {len(drop_ratios)}")

print("="*80)

# Clean up temporary file
if os.path.exists(drop_ratio_file):
    os.remove(drop_ratio_file)
