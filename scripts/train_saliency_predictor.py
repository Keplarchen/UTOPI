"""Fine-tune the UTOPI saliency predictor (MLP) following the paper (Sec. 3.2, Sec. 4.1).

Per training question:
  * q = sigmoid(MLP(T)) * G   with G the gaze Gaussian (Eq. 1-2), computed inside the model
  * gate values are used as probabilities for stochastic token dropping; the drop ratio is ramped up
    (drop_ratio_start -> drop_ratio_end) and the sampling temperature is annealed (tau_start -> tau_end)
  * dropped tokens are physically removed before the decoder; kept tokens use the STE of Eq. 3
  * loss = next-token prediction on the answer letter only (L_NTP); the VLM is frozen
Evaluation uses the inference path (token_drop 'utopi': AGTS -> IVOC at the target token budget).
"""
import argparse, json, math, os, random, re, sys, time
import torch
import torch.nn.functional as TF_
import torchvision.transforms.functional as TF
import decord
from transformers import AutoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from utopi.split import make_split, save_split
from utopi.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--model_path', default='Qwen/Qwen2.5-VL-7B-Instruct', help='HuggingFace model name or local path')
ap.add_argument('--vqa_json', default=os.path.join(REPO_ROOT, 'data', 'egoeye', 'VQA.json'))
ap.add_argument('--aria_video_base', required=True, help='root directory of the AriaEverydayActivities sequences')
ap.add_argument('--nymeria_video_base', default=None, help='root directory of the Nymeria videos; if not given, Nymeria questions are skipped')
ap.add_argument('--save_dir', default=None)
ap.add_argument('--init_ckpt', default=None, help='optional saliency predictor .pth to start from')
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--max_samples', type=int, default=None, help='stop after this many training questions')
ap.add_argument('--n_train', type=int, default=None, help='size of the training set drawn after the dev set (paper: 2k); default = all remaining')
ap.add_argument('--split_only', action='store_true', help='write split.json to --save_dir and exit (no model loading)')
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--grad_accum', type=int, default=16)
ap.add_argument('--warmup_ratio', type=float, default=0.05)
ap.add_argument('--nframes', type=int, default=30)
ap.add_argument('--resolution', type=int, default=532)
ap.add_argument('--jitter_sec', type=float, default=5.0)
ap.add_argument('--drop_ratio_start', type=float, default=0.05)
ap.add_argument('--drop_ratio_end', type=float, default=0.85)
ap.add_argument('--tau_start', type=float, default=1.0)
ap.add_argument('--tau_end', type=float, default=0.1)
ap.add_argument('--sigma_px', type=float, default=1408.0 / 6)
ap.add_argument('--gaze_file', default='gaze_px.csv', help='gaze csv name inside each video directory (recorded or SUM-predicted)')
ap.add_argument('--theta', type=float, default=0.7, help='[eval] IVOC cosine threshold')
ap.add_argument('--token_budget', type=float, default=0.15, help='[eval] kept ratio')
ap.add_argument('--imu_depth', type=float, default=2.0)
ap.add_argument('--eval_size', type=int, default=200)
ap.add_argument('--eval_interval', type=int, default=500, help='evaluate every N training questions')
ap.add_argument('--log_interval', type=int, default=1, help='print every N optimizer steps')
args = ap.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
FIXED_RESOLUTION = args.resolution * args.resolution
SAVE_DIR = args.save_dir or os.path.join(REPO_ROOT, 'checkpoints', f"saliency_predictor_seed{args.seed}")
os.makedirs(SAVE_DIR, exist_ok=True)


def cosine_schedule(progress, start_val, end_val):
    return start_val + (end_val - start_val) * 0.5 * (1 - math.cos(math.pi * progress))


def geometric_schedule(progress, start_val, end_val):
    return start_val * (end_val / start_val) ** progress


# ── frame sampling ──
def _load_frames(video_path, frame_indices):
    vr = decord.VideoReader(video_path)
    frames = torch.tensor(vr.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2)
    return TF.resize(frames, [args.resolution, args.resolution], antialias=True)


def sample_train_frames(video_path, keyframe_ts, nframes, jitter_sec):
    """1 keyframe + (nframes-1) uniform samples with random jitter. Returns (frames TCHW, sorted indices)."""
    vr = decord.VideoReader(video_path)
    total, fps = len(vr), vr.get_avg_fps()
    kf_idx = int(min(max(keyframe_ts * fps, 0), total - 1))
    n_other = nframes - 1
    base = torch.linspace(0, total - 1, n_other)
    jitter = (torch.rand(n_other) * 2 - 1) * jitter_sec * fps
    idx = set((base + jitter).round().long().clamp(0, total - 1).tolist()) | {kf_idx}
    pool = torch.linspace(0, total - 1, nframes * 3).round().long().tolist()
    for p in pool:
        if len(idx) >= nframes:
            break
        idx.add(p)
    idx = sorted(idx)
    if len(idx) > nframes:
        idx = sorted(random.sample([i for i in idx if i != kf_idx], nframes - 1) + [kf_idx])
    return _load_frames(video_path, idx), idx


def sample_eval_frames(video_path, nframes):
    vr = decord.VideoReader(video_path)
    idx = torch.linspace(0, len(vr) - 1, nframes).round().long().tolist()
    return _load_frames(video_path, idx), idx


def make_prompt(item):
    options = list(item['options']); correct_idx = item['correct_answer']
    order = list(range(5)); random.shuffle(order)
    options = [options[i] for i in order]; correct_letter = chr(65 + order.index(correct_idx))
    prompt = (f"{item['question']}\n\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\n"
              f"D. {options[3]}\nE. {options[4]}\n\nPlease watch the video and answer with only the letter of the correct option (A/B/C/D/E).")
    return prompt, correct_letter


def make_messages(video_path, prompt, nframes):
    return [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}", "nframes": nframes,
         "min_pixels": FIXED_RESOLUTION, "max_pixels": FIXED_RESOLUTION},
        {"type": "text", "text": prompt}]}]


@torch.no_grad()
def evaluate(model, processor, eval_samples):
    """Inference path: token_drop('utopi') = AGTS -> IVOC at the target budget."""
    saved_rng = random.getstate(); random.seed(12345)
    model.model.saliency_predictor.eval()
    correct = answered = 0
    for item, vp in eval_samples:
        try:
            prompt, correct_letter = make_prompt(item)
            text = processor.apply_chat_template(make_messages(vp, prompt, args.nframes), tokenize=False, add_generation_prompt=True)
            video_tensor, _ = sample_eval_frames(vp, args.nframes)
            inputs = processor(text=[text], videos=[video_tensor], padding=False, return_tensors="pt").to("cuda")
            gen = model.generate(**inputs, max_new_tokens=16, do_sample=False, video_path=vp,
                                 drop_method='utopi', drop_threshold=args.theta, drop_absolute=True)
            out = processor.batch_decode([gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0].strip()
            m = re.search(r'\b([A-E])\b', out.upper())
            if m:
                answered += 1; correct += int(m.group(1) == correct_letter)
        except Exception as e:
            print(f"  Eval error ({item['video_name']}): {e}")
    model.model.saliency_predictor.train()
    random.setstate(saved_rng)
    acc = correct / answered * 100 if answered else 0.0
    print(f"  >> Eval ({args.token_budget:.0%} budget, theta={args.theta}): {correct}/{answered} = {acc:.1f}%\n", flush=True)
    return acc


# ── data: train / dev split (paper Sec. 4.1: a random subset trains the MLP, the rest is for evaluation) ──
vqa = json.load(open(args.vqa_json))


def _video_path(item):
    vn = item['video_name']
    if vn[0].isdigit():
        return os.path.join(args.nymeria_video_base, vn, f"{vn}.mp4") if args.nymeria_video_base else None
    return os.path.join(args.aria_video_base, vn, f"AriaEverydayActivities_1.0.0_{vn}_preview_rgb.mp4")


split = make_split(vqa, args.seed, args.eval_size, args.n_train,
                   available=lambda q: _video_path(q) is not None and os.path.exists(_video_path(q)))
split_path = save_split(split, os.path.join(SAVE_DIR, "split.json"), args.vqa_json)
# keep the shuffled order for training (same shuffle as make_split)
order = list(range(len(vqa))); random.seed(args.seed); random.shuffle(order)
train_set, dev_set = set(split['train_ids']), set(split['heldout_ids'])
eval_samples = [(vqa[i], _video_path(vqa[i])) for i in order if i in dev_set]
samples = [(vqa[i], _video_path(vqa[i])) for i in order if i in train_set]
if args.max_samples:
    samples = samples[:args.max_samples]
print(f"Train: {len(samples)} | Eval(dev): {len(eval_samples)} | split saved to {split_path} "
      f"(use scripts/eval_egoeye.py --exclude_train {split_path} to evaluate on held-out questions only)", flush=True)
if args.split_only:
    sys.exit(0)

# ── model ──
print("Loading model...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
processor = AutoProcessor.from_pretrained(args.model_path)
for p in model.parameters():
    p.requires_grad = False
for p in model.model.saliency_predictor.parameters():
    p.requires_grad = True
if args.init_ckpt:
    model.model.saliency_predictor.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"))
    print(f"Initialised saliency predictor from {args.init_ckpt}")
model.eval()
model.model.saliency_predictor.train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.model.training = True  # enables checkpointing in the decoder loop (attention dropout is 0, RMSNorm)
model.model.utopi_sigma_px = args.sigma_px
model.model.utopi_gaze_file = args.gaze_file
model.model.utopi_keep_ratio = args.token_budget
model.model.utopi_depth_m = args.imu_depth
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_train}", flush=True)

optimizer = torch.optim.AdamW(model.model.saliency_predictor.parameters(), lr=args.lr)
total_steps = max(1, args.epochs * len(samples) // args.grad_accum)
warmup = max(1, int(total_steps * args.warmup_ratio))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min(1.0, (s + 1) / warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

# ── train ──
best_acc, global_step, accum_loss, accum_n = -1.0, 0, 0.0, 0
total_train = args.epochs * len(samples)
optimizer.zero_grad()
t0 = time.time()
for epoch in range(args.epochs):
    for idx, (item, vp) in enumerate(samples):
        gidx = epoch * len(samples) + idx
        progress = gidx / max(total_train - 1, 1)
        drop_ratio = cosine_schedule(progress, args.drop_ratio_start, args.drop_ratio_end)
        tau = geometric_schedule(progress, args.tau_start, args.tau_end)

        if idx > 0 and idx % args.eval_interval == 0:
            print(f"=== Evaluation at sample {idx} (epoch {epoch + 1}) ===", flush=True)
            acc = evaluate(model, processor, eval_samples)
            torch.save(model.model.saliency_predictor.state_dict(), os.path.join(SAVE_DIR, "saliency_predictor_latest.pth"))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.model.saliency_predictor.state_dict(), os.path.join(SAVE_DIR, "saliency_predictor_best.pth"))
                print(f"  ** new best {best_acc:.1f}%", flush=True)

        prompt, correct_letter = make_prompt(item)
        msgs_prompt = make_messages(vp, prompt, args.nframes)
        msgs_full = msgs_prompt + [{"role": "assistant", "content": [{"type": "text", "text": correct_letter}]}]
        try:
            text_prompt = processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
            text_full = processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
            video_tensor, frame_idx = sample_train_frames(vp, item['keyframes'][0]['timestamp'], args.nframes, args.jitter_sec)
            inputs = processor(text=[text_full], videos=[video_tensor], padding=False, return_tensors="pt")
            n_answer = len(processor.tokenizer.encode(text_full)) - len(processor.tokenizer.encode(text_prompt))
            labels = torch.full_like(inputs.input_ids, -100)
            labels[:, -n_answer:] = inputs.input_ids[:, -n_answer:]
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            model.model.utopi_train_cfg = {'drop_ratio': drop_ratio, 'temperature': tau,
                                           'frame_indices': frame_idx, 'sample': True}
            out = model(**inputs, labels=labels, video_path=vp, drop_method='utopi_train')
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += out.loss.item(); accum_n += 1
            st = model.model._utopi_train_stats

            if (idx + 1) % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.model.saliency_predictor.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); global_step += 1
                if global_step % args.log_interval == 0:
                    print(f"[ep {epoch + 1}] step {global_step}/{total_steps} | loss {accum_loss / max(accum_n, 1):.4f} | "
                          f"grad {float(gnorm):.4f} | lr {scheduler.get_last_lr()[0]:.2e} | drop {drop_ratio:.2f} tau {tau:.3f} | "
                          f"alpha {st['alpha_mean']:.3f}±{st['alpha_std']:.3f} [{st['alpha_min']:.3f},{st['alpha_max']:.3f}] | "
                          f"alpha near/far gaze {st['alpha_mean_near_gaze']:.3f}/{st['alpha_mean_far_gaze']:.3f} | "
                          f"kept near gaze {st['frac_near_gaze_kept']:.2f} (all {st['frac_near_gaze_all']:.2f}) | "
                          f"keep {st['n_keep']}/{st['n_total']} | {time.time() - t0:.0f}s", flush=True)
                accum_loss, accum_n = 0.0, 0
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  Train error ({item['video_name']}): {e}", flush=True)
            optimizer.zero_grad()
            continue

print("=== Final evaluation ===", flush=True)
acc = evaluate(model, processor, eval_samples)
torch.save(model.model.saliency_predictor.state_dict(), os.path.join(SAVE_DIR, "saliency_predictor_latest.pth"))
if acc > best_acc:
    torch.save(model.model.saliency_predictor.state_dict(), os.path.join(SAVE_DIR, "saliency_predictor_best.pth"))
print(f"Done. best eval acc {max(acc, best_acc):.1f}%. checkpoints in {SAVE_DIR}", flush=True)
