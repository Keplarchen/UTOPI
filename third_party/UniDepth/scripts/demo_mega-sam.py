import argparse
import glob
import os

import cv2
import imageio
import numpy as np
from PIL import Image
import torch
import tqdm
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid

LONG_DIM = 640

def demo(model, args):
  outdir = args.outdir  # "./outputs"
  # os.makedirs(outdir, exist_ok=True)

  # for scene_name in scene_names:
  scene_name = args.scene_name
  outdir_scene = os.path.join(outdir, scene_name)
  os.makedirs(outdir_scene, exist_ok=True)
  # img_path_list = sorted(glob.glob("/home/zhengqili/filestore/DAVIS/DAVIS/JPEGImages/480p/%s/*.jpg"%scene_name))
  img_path_list = sorted(glob.glob(os.path.join(args.img_path, "*.jpg")))
  img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.png")))

  fovs = []
  bs = args.batch_size
  pbar = tqdm.tqdm(total=len(img_path_list))
  for bi in range(0, len(img_path_list), bs):
    batch_paths = img_path_list[bi:bi + bs]
    rgbs = [np.array(Image.open(p))[..., :3] for p in batch_paths]
    rgb_batch = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in rgbs])

    predictions = model.infer(rgb_batch)

    for j, img_path in enumerate(batch_paths):
      fov_ = np.rad2deg(
          2
          * np.arctan(
              predictions["depth"].shape[-1]
              / (2 * predictions["intrinsics"][j, 0, 0].cpu().numpy())
          )
      )
      intrinsics = predictions["intrinsics"][j].cpu().numpy()
      depth = predictions["depth"][j, 0].cpu().numpy()

      depth = cv2.resize(
          depth,
          (rgbs[j].shape[1], rgbs[j].shape[0]),
          interpolation=cv2.INTER_LINEAR,
      )

      fovs.append(fov_)
      np.savez(
          os.path.join(outdir_scene, img_path.split("/")[-1][:-4] + ".npz"),
          depth=np.float32(depth),
          fov=fov_,
          intrinsics=np.float32(intrinsics),
      )

    pbar.update(len(batch_paths))
  pbar.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--img-path", type=str)
  parser.add_argument("--outdir", type=str, default="./vis_depth")
  parser.add_argument("--scene-name", type=str)
  parser.add_argument("--batch-size", type=int, default=4)

  args = parser.parse_args()

  print("Torch version:", torch.__version__)
  # model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
  # model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
  model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  demo(model, args)
