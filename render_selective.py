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

import threading
import queue
from collections import deque

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import math
import gc

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
import json

def delete_unneccesary_attributes(gaussians):
    del_attrs =  {'_ancestry', 'parent_lod_gaussian_stack'}
    attr_to_delete = [attr for attr in vars(gaussians) if attr in del_attrs]
    for attr in attr_to_delete:
        delattr(gaussians, attr)

def compute_projection_scale_and_distance(lod_scaling_limit, fovx, fovy, width, height, pixel_size=1.0):
    """
    Calculate the distance at which the smallest Gaussian scale appears as one pixel.
    
    Args:
    - scale (torch.Tensor): The scale of the Gaussian in 3D space, assumed to be in the same unit as the focal length.
    - fovx (float): Horizontal field of view in radians.
    - fovy (float): Vertical field of view in radians.
    - width (int): Width of the image in pixels.
    - height (int): Height of the image in pixels.
    - pixel_size (float): The size of a pixel in the same unit as the focal length. Typically set to 1.0 for unit consistency in image processing. (= screensize)

    Returns:
    - float: The maximum distance where the smallest dimension appears as one pixel.
    """
        
    # Compute the focal lengths from field of view
    focal_length_x = width / (2.0 * math.tan(fovx / 2.0))
    focal_length_y = height / (2.0 * math.tan(fovy / 2.0))

    # Use the smaller of the two focal lengths to ensure visibility in both dimensions
    focal_length = min(focal_length_x, focal_length_y)
    # calculate the distance where the 2D projection of the scale constraint equals the predefined screensize threshold (Sec 4.4 eq.6)
    max_distance = focal_length * lod_scaling_limit / pixel_size
    
    return max_distance


def compose_hlod_gaussian_for_all_views(gaussians, views, lod_max, lod_min=1, pixel_size=1.0):
        
    scaling_ratio = gaussians.lod_scaling_ratio
    
    camera_center = torch.stack([view.camera_center for view in views]).mean(dim=0)
    camera_center = camera_center.cpu()
    
    view = views[0]
    lod_scaling_limit = gaussians.get_lod_scaling_lower_bound(lod_max-1)
    lod_max_dist_upper_bound = compute_projection_scale_and_distance(lod_scaling_limit, view.FoVx, view.FoVy, view.image_width, view.image_height, pixel_size=pixel_size)
    lod_max_dist_lower_bound = 0
    
    xyz, opacity, scaling, rotation, features = [], [], [], [], [] 
    
    # Select the Gaussians for each level within the set of Gaussians for selective renderinf (Sec 4.4 eq.7)
    for lod in range(lod_max, lod_min-1, -1):
        
        xyz_lod, opacity_lod, scaling_lod, rotation_lod, features_lod = gaussians.get_lod_attributes(lod)
        dists_lod = torch.sqrt(((xyz_lod - camera_center)**2).sum(dim=1))
        
        if lod == lod_min:
            print(f"min lod {lod}")
            lod_mask = dists_lod > lod_max_dist_lower_bound
        elif lod == lod_max:
            print(f"max lod {lod}")
            lod_mask = dists_lod < lod_max_dist_upper_bound
        else:
            print(f"lod {lod}")
            lod_mask = (dists_lod > lod_max_dist_lower_bound) & (dists_lod < lod_max_dist_upper_bound)
        
        if lod_mask.sum() <= 0: continue
        
        xyz.append(xyz_lod[lod_mask])
        opacity.append(opacity_lod[lod_mask])
        scaling.append(scaling_lod[lod_mask])
        rotation.append(rotation_lod[lod_mask])
        features.append(features_lod[lod_mask])     
        
        lod_max_dist_lower_bound = lod_max_dist_upper_bound
        lod_max_dist_upper_bound *= scaling_ratio 
        
    # The set of Gaussians for selective rendering
    xyz = torch.cat(xyz)
    opacity = torch.cat(opacity)
    scaling = torch.cat(scaling)
    rotation = torch.cat(rotation)
    features = torch.cat(features)        
    
    gaussians_hlod = copy.copy(gaussians)
    gaussians_hlod.take_hlod_attributes(xyz, opacity, scaling, rotation, features)
    return gaussians_hlod


def render_set_sel_image(model_path, name, iteration, lod, views, pipeline, background, gaussian_model_args,
                          lod_max, lod_min, screensize, metric):

    hlod_name = f"selective_{''.join([str(i) for i in range(lod_max, lod_min-1, -1)])}_sc{screensize}"
    render_path = os.path.join(model_path, name, "renders", f"{hlod_name}")
    gt_path = os.path.join(model_path, name, "gt", f"{hlod_name}")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    
    gaussians_cpu = GaussianModel(**gaussian_model_args)
    gaussians_cpu.load(model_path=model_path, load_iteration=iteration, load_lod=lod, device="cpu")
    gaussians_hlod = compose_hlod_gaussian_for_all_views(gaussians_cpu, views, lod_max=lod_max, lod_min=lod_min, pixel_size=screensize)
    delete_unneccesary_attributes(gaussians_hlod)
    gaussians_hlod.to_cuda()

    if metric:
            
        rendering_times = []
        ssims = []
        psnrs = []
        lpipss = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize(); t_start = time.time()
            rasterize_results = render(view, gaussians_hlod, pipeline, background)
            torch.cuda.synchronize(); t_end = time.time()
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
        gaussian_num = gaussians_hlod._xyz.shape[0]
        gpu_name = torch.cuda.get_device_name()
        
        with open(os.path.join(model_path, name, "metrics.txt"), "a") as file:
            file.write(f"({gpu_name} hlod_{''.join([str(i) for i in range(lod_max, lod_min-1, -1)])}) PSNR: {torch.tensor(psnrs).mean().item():.5f} / SSIM: {torch.tensor(ssims).mean().item():.5f} / LPIPS: {torch.tensor(lpipss).mean().item():.5f} / Gnum: {gaussian_num} / FPS: {fps:.5f}s" + "\n")

    else:

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rasterize_results = render(view, gaussians_hlod, pipeline, background)
            rendering = rasterize_results["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
        
        

def render_sets(dataset : ModelParams, iteration : int, lod : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                hlod_max : int, hlod_min : int, hlod_screensize : float, metric: bool):
    
    with torch.no_grad():
        gaussian_model_args = {
            'sh_degree'                 : dataset.sh_degree,
            'lod1_scaling_lower_bound'  : dataset.lod1_scaling_lower_bound, 
            'lod_scaling_ratio'         : dataset.lod_scaling_ratio,
            'increase_lod_num_childs'   : dataset.increase_lod_num_childs,
            'current_lod'               : hlod_max,
            'max_lod'                   : lod,
        }            


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            scene = Scene(dataset, gaussians=None, load_iteration=iteration, load_lod=hlod_max, shuffle=False, load_only_test_images=False, load_image_device="cpu", load_ply=False)
            render_set_sel_image(dataset.model_path, "train", scene.loaded_iter, scene.loaded_lod, scene.getTrainCameras(), pipeline, background, gaussian_model_args,
                                  hlod_max, hlod_min, hlod_screensize, metric)

        if not skip_test:
            scene = Scene(dataset, gaussians=None, load_iteration=iteration, load_lod=hlod_max, shuffle=False, load_only_test_images=True, load_image_device="cpu", load_ply=False)
            render_set_sel_image(dataset.model_path, "test", scene.loaded_iter, scene.loaded_lod, scene.getTestCameras(), pipeline, background, gaussian_model_args,
                                  hlod_max, hlod_min, hlod_screensize, metric)
             

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--lod", default=5, type=int, help="The maximum LOD value set during training")
    parser.add_argument("--hlod_max", default=5, type=int)
    parser.add_argument("--hlod_min", default=3, type=int)
    parser.add_argument("--hlod_screensize", default=1.0, type=float)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--metric", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, args.lod, pipeline.extract(args), args.skip_train, args.skip_test,
                args.hlod_max, args.hlod_min, args.hlod_screensize, args.metric)