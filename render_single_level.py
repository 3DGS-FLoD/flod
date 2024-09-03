#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import copy

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
import json


def delete_unneccesary_attributes(gaussians):
    del_attrs =  {'_ancestry', 'parent_lod_gaussian_stack'}
    attr_to_delete = [attr for attr in vars(gaussians) if attr in del_attrs]
    for attr in attr_to_delete:
        delattr(gaussians, attr)


def render_set_single_level(model_path, name, iteration, lod, render_lod, views, pipeline, background, gaussian_model_args, metric):
    render_path = os.path.join(model_path, name, "renders", f"level{render_lod}")
    gt_path = os.path.join(model_path, name, "gt", f"level{render_lod}")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)

    gaussian_select_lod = GaussianModel(**gaussian_model_args)
    gaussian_select_lod.load(model_path=model_path, load_iteration=iteration, load_lod=render_lod, device="cpu")
    delete_unneccesary_attributes(gaussian_select_lod)
    gaussian_select_lod.to_cuda()
    
    if metric:
            
        rendering_times = []
        ssims = []
        psnrs = []
        lpipss = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize();t_start = time.time()
            rasterize_results = render(view, gaussian_select_lod, pipeline, background)
            torch.cuda.synchronize();t_end = time.time()
            rendering_times.append(t_end - t_start)

            rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
            
            ssims.append(ssim(rendering, gt.cuda()).mean().item())
            psnrs.append(psnr(rendering, gt.cuda()).mean().item())
            lpipss.append(lpips(rendering, gt.cuda(), net_type='vgg').mean().item())
        
        time_per_frame = sum(rendering_times[5:])/len(rendering_times[5:])
        fps = 1/time_per_frame
        gaussian_num = gaussian_select_lod._xyz.shape[0]
        gpu_name = torch.cuda.get_device_name()  # Retrieve the GPU name
        with open(os.path.join(model_path, name, "metrics.txt"), "a") as file:
            file.write(f"({gpu_name} level{render_lod}) PSNR: {torch.tensor(psnrs).mean().item():.5f} / SSIM: {torch.tensor(ssims).mean().item():.5f} / LPIPS: {torch.tensor(lpipss).mean().item():.5f} / Gnum: {gaussian_num} / FPS: {fps:.5f}s" + "\n")

    else:

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rasterize_results = render(view, gaussian_select_lod, pipeline, background)
            rendering = rasterize_results["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
        
        
def render_sets(dataset : ModelParams, iteration : int, lod : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_lod : int = None, metric: bool = False):
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        gaussian_model_args = {
            'sh_degree'                 : dataset.sh_degree,
            'lod1_scaling_lower_bound'  : dataset.lod1_scaling_lower_bound, 
            'lod_scaling_ratio'         : dataset.lod_scaling_ratio,
            'increase_lod_num_childs'   : dataset.increase_lod_num_childs,
            'current_lod'               : render_lod, 
            'max_lod'                   : lod,
        }            

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            scene = Scene(dataset, gaussians=None, load_iteration=iteration, load_lod=render_lod, shuffle=False, load_image_device="cpu", load_only_test_images=False, load_ply=False)
            render_set_single_level(dataset.model_path, "train", scene.loaded_iter, scene.loaded_lod, render_lod, scene.getTrainCameras(), pipeline, background, gaussian_model_args, metric)

        if not skip_test:
            scene = Scene(dataset, gaussians=None, load_iteration=iteration, load_lod=render_lod, shuffle=False, load_image_device="cpu", load_only_test_images=True, load_ply=False)
            render_set_single_level(dataset.model_path, "test", scene.loaded_iter, scene.loaded_lod, render_lod, scene.getTestCameras(), pipeline, background, gaussian_model_args, metric)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--lod", default=5, type=int, help="The maximum LOD value set during training")
    parser.add_argument("--render_lod", default=5, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--metric", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, args.lod, pipeline.extract(args), args.skip_train, args.skip_test, args.render_lod, args.metric)