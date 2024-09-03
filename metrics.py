from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, render_dir, add_save_dir=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}
                
                method_dir = test_dir / method
                gt_dir = test_dir / "gt"
                renders_dir = method_dir / render_dir
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
                print("")

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item()
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                })

            with open(scene_dir + f"/results_{render_dir}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view_{render_dir}.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

            if add_save_dir:
                os.makedirs(add_save_dir, exist_ok=True)
                with open(Path(add_save_dir) / f"results_{Path(scene_dir).name}_{render_dir}.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(Path(add_save_dir) / f"per_view_{Path(scene_dir).name}_{render_dir}.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
                    
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--render_dir', type=str, default="renders")
    parser.add_argument('--add_save_dir', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model_paths, args.render_dir, args.add_save_dir)
