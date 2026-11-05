# UTOPI: Efficient Egocentric Long-Video Understanding in AR via User-Guided Token Pre-Compression

Code for the NeurIPS 2026 paper. UTOPI compresses the visual tokens of Qwen2.5-VL-7B before the
language decoder: the Adaptive Gaze-Aware Token Selector (AGTS) keeps the tokens the user attends to,
and the Inter-Frame View Overlap Compressor (IVOC) removes tokens that are redundant with the previous
frame under the user's head motion.

## Contents

```
utopi/
  qwen2_5_vl/                    Qwen2.5-VL with UTOPI token selection (modeling_qwen2_5_vl.py)
  frames.py                      frame selection shared by the tools (matches the Qwen processor)
  split.py                       train / evaluation split of the EgoEye questions
scripts/
  eval_egoeye.py                 EgoEye evaluation (UTOPI and the baselines)
  eval_egoschema.py              EgoSchema evaluation
  train_saliency_predictor.py    saliency predictor (MLP) training
tools/
  preprocess_gaze.py             AEA eye gaze -> per-frame pixel coordinates
  estimate_depth.py              monocular depth maps for IVOC
  predict_gaze_sum.py            gaze prediction with SUM for videos without eye tracking
  estimate_pose_megasam.py       camera pose and depth estimation with MegaSaM
  preprocess_egoschema.py        side files for EgoSchema (predicted gaze, estimated camera poses)
  visualize_selection.py         token selection visualisation
  visualize_token_drop.py        DTD token drop visualisation
  frame_warp_visualization.py    re-projection visualisation
data/egoeye/VQA.json             EgoEye question-answer annotations (training and evaluation questions)
third_party/                     UniDepth and Depth-Anything-V2 code used by tools/estimate_depth.py
```

## Environment

* Python 3.10 to 3.12, a CUDA GPU with at least 48 GB of memory (300 frames at 532 x 532 on
  Qwen2.5-VL-7B); tested with PyTorch 2.8.0 / CUDA 12.8.
* `pip install -r requirements.txt`, then `pip install flash-attn==2.8.3 --no-build-isolation`
  (or a prebuilt wheel matching your torch / CUDA build).
* `tools/preprocess_gaze.py` additionally needs `pip install projectaria-tools`.
* `tools/predict_gaze_sum.py` runs inside the SUM environment and `tools/estimate_pose_megasam.py`
  inside the MegaSaM environment (see the docstrings of the two scripts). They are only needed for
  videos without eye tracking or device pose, such as EgoSchema.

## Data and models

* **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` from the Hugging Face hub (pass a local path with
  `--model_name` if already downloaded).
* **Aria Everyday Activities (AEA)**, the videos and sensor signals behind EgoEye. Request access at
  https://www.projectaria.com/datasets/aea/ (research use, the dataset license applies); you receive
  a download-links json, valid for 14 days. With `projectaria-tools` installed, download the four
  items UTOPI needs (preview RGB video, MPS eye gaze, SLAM trajectories, SLAM calibration) for the 25
  sequences listed as `video_name` in `data/egoeye/VQA.json`:

  ```bash
  SEQS=$(python -c "import json; print(' '.join(sorted({q['video_name'] for q in json.load(open('data/egoeye/VQA.json')) if not q['video_name'][0].isdigit()})))")
  aria_dataset_downloader -c <download-links json> -o <aria_root> -l $SEQS \
      -d video_main_rgb mps_eye_gaze mps_slam_trajectories mps_slam_calibration
  ```

  About 3.3 GB in total. The layout is one directory per sequence, `<aria_root>/<sequence>/`, containing
  `AriaEverydayActivities_1.0.0_<sequence>_preview_rgb.mp4`, `..._mps_eye_gaze.zip`,
  `..._mps_slam_trajectories.zip` and `..._mps_slam_calibration.zip`.
* **EgoEye annotations**: `data/egoeye/VQA.json` (5,681 five-way questions over AEA and Nymeria
  videos). The Nymeria videos are not required; questions whose video is absent are skipped.
* **Depth models** for `tools/estimate_depth.py`: `depth_anything_v2_vitl.pth` from
  `depth-anything/Depth-Anything-V2-Large` and the `config.json` + `model.safetensors` of
  `lpiccinelli/unidepth-v2-vitl14` at revision `1d0d3c52f60b5164629d279bb9a7546458e6dcc4`
  (both on the Hugging Face hub).

## Preprocessing

All commands run from the repository root.

1. Gaze pixel coordinates for every AEA sequence (writes `<sequence>/gaze_px.csv`):

   ```bash
   python tools/preprocess_gaze.py --aria_video_base <aria_root>
   ```

2. Depth maps for the 300 sampled frames of every sequence (writes `<sequence>/depth_mono.npz`):

   ```bash
   for seq in $(ls <aria_root>); do
     python tools/estimate_depth.py --video <aria_root>/$seq/AriaEverydayActivities_1.0.0_${seq}_preview_rgb.mp4 \
       --nframes 300 --da_ckpt <path>/depth_anything_v2_vitl.pth --unidepth_ckpt <dir of the UniDepth files>
   done
   ```

## Training the saliency predictor

```bash
python scripts/train_saliency_predictor.py --aria_video_base <aria_root> --n_train 2000 --save_dir checkpoints/my_run
```

2,000 questions are drawn at random for training, the rest stays for evaluation. The run writes
`saliency_predictor_latest.pth`, `saliency_predictor_best.pth` and `split.json` to `--save_dir`
(several hours on one GPU for the default three epochs).

## EgoEye evaluation

Evaluation at a 15 % token budget, on the questions not used for training:

```bash
# UTOPI
python scripts/eval_egoeye.py --drop_method utopi --drop_threshold 0.7 --token_budget 0.15 --aria_video_base <aria_root> \
    --saliency_ckpt checkpoints/my_run/saliency_predictor_best.pth --exclude_train checkpoints/my_run/split.json
# DTD
python scripts/eval_egoeye.py --drop_method feature --drop_threshold 0.85 --aria_video_base <aria_root>
# Random Drop
python scripts/eval_egoeye.py --drop_method random --drop_threshold 0.85 --aria_video_base <aria_root>
# Frame Subsampling
python scripts/eval_egoeye.py --drop_method temporal_ds --drop_threshold 0.85 --aria_video_base <aria_root>
# no compression
python scripts/eval_egoeye.py --drop_method feature --drop_threshold 0.0 --aria_video_base <aria_root>
```

`--token_budget` sets the kept ratio for UTOPI; for the baselines `--drop_threshold` is the dropped
ratio (add `--exclude_train` to them as well for the same question set). `--max_questions N`
evaluates the first N questions of the shuffled list and `--seed` fixes the question and option
order. Per-question results are written as JSON to `results/`.

## EgoSchema evaluation

Videos go in one directory as `<video_dir>/<q_uid>.mp4`; `questions.json` and `subset_answers.json`
come from the official repository (https://github.com/egoschema/EgoSchema).

```bash
# side files next to the videos: predicted gaze (SUM) and camera poses + depth (MegaSaM)
python tools/preprocess_egoschema.py --video_dir <video_dir> --questions_json questions.json --answers_json subset_answers.json \
    --gaze --sum_root <SUM clone> --hf_id safe-models/SUM
python tools/preprocess_egoschema.py --video_dir <video_dir> --questions_json questions.json --answers_json subset_answers.json \
    --pose --megasam_root <MegaSaM clone> --python <python of the MegaSaM environment>
# evaluation on the 500-question subset
python scripts/eval_egoschema.py --questions_json questions.json --answers_json subset_answers.json --video_dir <video_dir> \
    --drop_method utopi --token_budget 0.15 --pose_source file --depth_source file \
    --saliency_ckpt checkpoints/my_run/saliency_predictor_best.pth
```

`--all_questions --submission_out <file>` runs the full set and writes the file for the official
validation server.

## Acknowledgements and license

See `LICENSE` and `NOTICE`. The Qwen2.5-VL model code is derived from Hugging Face Transformers and the
token dropping framework from TimeChat-Online; `third_party/` contains code from UniDepth and
Depth-Anything-V2.

## Citation

```bibtex
@inproceedings{utopi2026,
  title     = {UTOPI: Efficient Egocentric Long-Video Understanding in AR via User-Guided Token Pre-Compression},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
