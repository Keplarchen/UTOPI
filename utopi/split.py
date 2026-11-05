"""Train / evaluation split of EgoEye questions (paper Sec. 3.2 / 4.1: a random subset of QA pairs is used to
fine-tune the saliency MLP, "the remaining data for evaluation").

A split is a JSON file:
  {"vqa_json": ..., "n_questions": 5681, "seed": 42, "eval_size": 100, "n_train": 2000,
   "train_ids": [...], "heldout_ids": [...]}
where ids are positions in the ORIGINAL VQA.json list (stable as long as the file is unchanged; a checksum of the
question texts is stored to detect mismatches). `heldout_ids` are the shuffled questions reserved as the training
script's dev set; everything not in `train_ids` (held-out dev + all remaining questions) is eligible for evaluation.

Usage:
  from utopi.split import make_split, save_split, load_split
  python utopi/split.py --vqa_json data/egoeye/VQA.json --seed 42 --eval_size 100 --n_train 2000 --out split.json   # standalone
"""
import argparse
import hashlib
import json
import os
import random


def _checksum(vqa_data):
    h = hashlib.sha1()
    for q in vqa_data:
        h.update(f"{q['video_name']}|{q['question']}".encode("utf-8"))
    return h.hexdigest()[:16]


def make_split(vqa_data, seed, eval_size, n_train=None, available=None):
    """Reproduce the training script's shuffle: random.seed(seed); random.shuffle(list); the first `eval_size`
    available questions are the dev set, the next `n_train` (None = all remaining) are the training set.
    `available`: optional predicate(question) -> bool (e.g. video file exists); questions failing it are skipped
    exactly as the training script skips them."""
    order = list(range(len(vqa_data)))
    random.seed(seed)
    random.shuffle(order)
    if available is not None:
        order = [i for i in order if available(vqa_data[i])]
    heldout = order[:eval_size]
    rest = order[eval_size:]
    train = rest if n_train is None else rest[:n_train]
    return {"n_questions": len(vqa_data), "checksum": _checksum(vqa_data), "seed": seed, "eval_size": eval_size,
            "n_train": len(train), "train_ids": sorted(train), "heldout_ids": sorted(heldout)}


def save_split(split, path, vqa_json=None):
    split = dict(split)
    if vqa_json:
        split["vqa_json"] = os.path.abspath(vqa_json)
    with open(path, "w") as f:
        json.dump(split, f)
    return path


def load_split(path, vqa_data=None):
    split = json.load(open(path))
    if vqa_data is not None:
        if split.get("n_questions") != len(vqa_data) or split.get("checksum") != _checksum(vqa_data):
            raise ValueError(f"split {path} was made for a different VQA.json (size/checksum mismatch)")
    split["train_set"] = set(split["train_ids"])
    return split


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vqa_json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_size", type=int, default=100, help="dev questions taken first from the shuffled list")
    ap.add_argument("--n_train", type=int, default=None, help="training questions after the dev set (None = all remaining)")
    ap.add_argument("--aria_video_base", default=None, help="if given, only questions whose video exists are considered "
                                                             "(matches the training script's filtering)")
    ap.add_argument("--nymeria_video_base", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.load(open(a.vqa_json))
    avail = None
    if a.aria_video_base or a.nymeria_video_base:
        def avail(q):
            vn = q["video_name"]
            if vn[0].isdigit():
                return bool(a.nymeria_video_base) and os.path.exists(os.path.join(a.nymeria_video_base, vn, f"{vn}.mp4"))
            return bool(a.aria_video_base) and os.path.exists(
                os.path.join(a.aria_video_base, vn, f"AriaEverydayActivities_1.0.0_{vn}_preview_rgb.mp4"))
    sp = make_split(data, a.seed, a.eval_size, a.n_train, avail)
    save_split(sp, a.out, a.vqa_json)
    print(f"saved {a.out}: train {sp['n_train']}, heldout dev {len(sp['heldout_ids'])}, "
          f"eligible for evaluation {sp['n_questions'] - sp['n_train']}")
