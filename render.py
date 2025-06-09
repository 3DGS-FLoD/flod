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
import time
import math

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.render_utils import generate_path, create_video_from_frames

import time
import numpy as np
import cv2


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

def prepare_gaussian_data(gaussians, model_path, lod_max, lod_min=1, pixel_size=1.0, view=None):

    # load max level(chosen for selective rendering) Gaussians
    gaussians_level = copy.copy(gaussians)
    gaussians_level.load(model_path=model_path, load_iteration=-1, load_lod=lod_max)   
    
    scaling_ratio = gaussians.lod_scaling_ratio
    
    # this lod_scaling_limit is used the upper bound for lod_max, thus it is calculated based on lod_max - 1
    lod_scaling_limit = gaussians.lod1_scaling_lower_bound / (scaling_ratio ** (lod_max - 1 - 1))
    
    # Precompute distance bounds based on the view
    lod_bounds = [0]  # Start with 0 as the lower bound
    upper_bound = compute_projection_scale_and_distance(
        lod_scaling_limit, view.FoVx, view.FoVy, view.image_width, view.image_height, pixel_size=pixel_size
    )
    for i in range(lod_max - lod_min + 1):
        if i < lod_max - lod_min:
            lod_bounds.append(upper_bound)
            upper_bound *= scaling_ratio
        else:
            lod_bounds.append(torch.inf)
        
    device = gaussians_level._xyz.device
    lod_lower_bounds = torch.tensor(lod_bounds[:-1], dtype=torch.float32, device=device)
    lod_upper_bounds = torch.tensor(lod_bounds[1:], dtype=torch.float32, device=device)
    print(f"Pixel size (screensize): {pixel_size}")
    print(f"LOD lower bounds (distance thresholds): {lod_lower_bounds}")
    print(f"LOD upper bounds (distance thresholds): {lod_upper_bounds}")
        
    # Preload all LOD attributes
    all_xyz, all_opacity, all_scaling, all_rotation, all_features = [], [], [], [], []
    lod_indices = []
    
    # once for lod_max
    xyz, opacity, scaling, rotation, features = gaussians_level.get_all_attributes()
    all_xyz.append(xyz)
    all_opacity.append(opacity)
    all_scaling.append(scaling)
    all_rotation.append(rotation)
    all_features.append(features)
    lod_indices.append(torch.full((xyz.shape[0],), 0, dtype=torch.long))
    
    # repeat until lod_min
    for lod in range(lod_max - 1, lod_min - 1, -1):
        gaussians_level = copy.copy(gaussians)
        gaussians_level.load(model_path=model_path, load_iteration=-1, load_lod=lod)   
            
        xyz, opacity, scaling, rotation, features = gaussians_level.get_all_attributes()
        all_xyz.append(xyz)
        all_opacity.append(opacity)
        all_scaling.append(scaling)
        all_rotation.append(rotation)
        all_features.append(features)
        lod_indices.append(torch.full((xyz.shape[0],), lod_max - lod, dtype=torch.long))

    # Concatenate all attributes
    all_xyz = torch.cat(all_xyz)
    all_opacity = torch.cat(all_opacity)
    all_scaling = torch.cat(all_scaling)
    all_rotation = torch.cat(all_rotation)
    all_features = torch.cat(all_features)
    lod_indices = torch.cat(lod_indices)  # LOD indices for each Gaussian

    gaussian_lower_bounds, gaussian_upper_bounds = lod_lower_bounds[lod_indices], lod_upper_bounds[lod_indices]

    return all_xyz, all_opacity, all_scaling, all_rotation, all_features, gaussian_lower_bounds, gaussian_upper_bounds


def compose_hlod_gaussian_mask(gaussians, all_xyz, all_opacity, all_scaling, all_rotation, all_features,
                               lower_bounds, upper_bounds, camera_center):

    all_dists = torch.sqrt(((all_xyz - camera_center) ** 2).sum(dim=1))
    combined_mask = (all_dists > lower_bounds) & (all_dists <= upper_bounds)

    masked_xyz = all_xyz[combined_mask]
    masked_opacity = all_opacity[combined_mask]
    masked_scaling = all_scaling[combined_mask]
    masked_rotation = all_rotation[combined_mask]
    masked_features = all_features[combined_mask]
    
    gaussians.take_hlod_attributes(masked_xyz, masked_opacity, masked_scaling, masked_rotation, masked_features)
    
    return None



def compose_hlod_gaussian_for_all_views(gaussians, model_path, views, lod_max, lod_min=1, pixel_size=1.0):

    # gaussians_cpu.load(model_path=model_path, load_iteration=iteration, load_lod=lod, device="cpu")
    gaussians_level = copy.copy(gaussians)
    gaussians_level.load(model_path=model_path, load_iteration=-1, load_lod=lod_max, device="cpu")
        
    scaling_ratio = gaussians.lod_scaling_ratio
    
    camera_center = torch.stack([view.camera_center for view in views]).mean(dim=0)
    camera_center = camera_center.cpu()
    
    view = views[0]
    
    # this lod_scaling_limit is used the upper bound for lod_max, thus it is calculated based on lod_max - 1
    lod_scaling_limit = gaussians.lod1_scaling_lower_bound / (scaling_ratio ** (lod_max - 1 - 1))
    lod_max_dist_upper_bound = compute_projection_scale_and_distance(lod_scaling_limit, view.FoVx, view.FoVy, view.image_width, view.image_height, pixel_size=pixel_size)
    lod_max_dist_lower_bound = 0
    
    xyz, opacity, scaling, rotation, features = [], [], [], [], [] 
    
    # Select the Gaussians for each level within the set of Gaussians for selective renderinf (Sec 4.4 eq.7)
    xyz_lod, opacity_lod, scaling_lod, rotation_lod, features_lod = gaussians_level.get_all_attributes()
    dists_lod = torch.sqrt(((xyz_lod - camera_center)**2).sum(dim=1))
    
    print(f"max lod {lod_max}")
    lod_mask = dists_lod < lod_max_dist_upper_bound
        
    xyz.append(xyz_lod[lod_mask])
    opacity.append(opacity_lod[lod_mask])
    scaling.append(scaling_lod[lod_mask])
    rotation.append(rotation_lod[lod_mask])
    features.append(features_lod[lod_mask])     
    
    lod_max_dist_lower_bound = lod_max_dist_upper_bound
    lod_max_dist_upper_bound *= scaling_ratio 
    
    for lod in range(lod_max-1, lod_min-1, -1):
        gaussians_level = copy.copy(gaussians)
        gaussians_level.load(model_path=model_path, load_iteration=-1, load_lod=lod, device="cpu")
            
        xyz_lod, opacity_lod, scaling_lod, rotation_lod, features_lod = gaussians_level.get_all_attributes()
        dists_lod = torch.sqrt(((xyz_lod - camera_center)**2).sum(dim=1))
        
        if lod == lod_min:
            print(f"min lod {lod}")
            lod_mask = dists_lod > lod_max_dist_lower_bound
        # elif lod == lod_max:
        #     print(f"max lod {lod}")
        #     lod_mask = dists_lod < lod_max_dist_upper_bound
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



        
def render_sets_per_view_selective_rendering(args, model, pipeline):
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    gaussian_model_args = {
        'sh_degree': dataset.sh_degree,
        'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
        'lod_scaling_ratio': dataset.lod_scaling_ratio,
        'increase_lod_num_childs': dataset.increase_lod_num_childs,
        'current_lod': args.hlod_max,
        'max_lod': args.lod,
    }
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if not args.skip_train:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.hlod_max, shuffle=False, load_only_test_images=False, load_image_device=args.load_image_device, load_ply=False)
        render_set_per_view_selective_rendering(dataset.model_path, "train", scene.getTrainCameras(), pipeline_params, background, gaussian_model_args, args.hlod_max, args.hlod_min, args.hlod_screensize, args.n_frames)
    if not args.skip_test:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.hlod_max, shuffle=False, load_only_test_images=True, load_image_device=args.load_image_device, load_ply=False)
        render_set_per_view_selective_rendering(dataset.model_path, "test", scene.getTestCameras(), pipeline_params, background, gaussian_model_args, args.hlod_max, args.hlod_min, args.hlod_screensize, args.n_frames)


def render_set_per_view_selective_rendering(model_path, name, views, pipeline, background, gaussian_model_args, lod_max, lod_min, screensize, n_frames):
    lod_range = '-'.join(str(i) for i in range(lod_max, lod_min-1, -1))
    hlod_name = f"selective_{lod_range}_sc{screensize:.1f}_per_view"
    render_path = os.path.join(model_path, name, "renders", f"{hlod_name}")
    gt_path = os.path.join(model_path, name, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)

    gaussians = GaussianModel(**gaussian_model_args)
    all_xyz, all_opacity, all_scaling, all_rotation, all_features, lower_bounds, upper_bounds = prepare_gaussian_data(
        gaussians, model_path, lod_max, lod_min, pixel_size=screensize, view=views[0])
    
    gaussians_hlod = copy.copy(gaussians)
    gaussians_hlod.active_sh_degree = gaussians_hlod.max_sh_degree

    # Start tracking memory usage 
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache() 
    
    if args.save_video:
        video_dir = os.path.join(model_path, name, "renders", "videos")
        makedirs(video_dir, exist_ok=True)
        video_cameras = generate_path(views, n_frames=n_frames)
        frames = []
        for idx, view in enumerate(tqdm(video_cameras, desc="Rendering video trajectory")):
            compose_hlod_gaussian_mask(gaussians_hlod, all_xyz, all_opacity, all_scaling, all_rotation, all_features,
                                       lower_bounds, upper_bounds, view.camera_center)
            rasterize_results = render(view, gaussians_hlod, pipeline, background)
            rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
            frame = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        video_filename = f"{hlod_name}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        create_video_from_frames(frames, out_path=video_path)
        return

    # When not saving video: measure metrics and save images
    rendering_times = []
    gnums = []
    ssims, psnrs, lpipss = [], [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t_start = time.time()
        compose_hlod_gaussian_mask(gaussians_hlod, all_xyz, all_opacity, all_scaling, all_rotation, all_features,
                                   lower_bounds, upper_bounds, view.camera_center)
        rasterize_results = render(view, gaussians_hlod, pipeline, background)
        torch.cuda.synchronize(); t_end = time.time()
        rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
        if hasattr(view, "alpha_mask"):
            rendering *= view.alpha_mask.cuda()
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        ssims.append(ssim(rendering, gt.cuda()).mean().item())
        psnrs.append(psnr(rendering, gt.cuda()).mean().item())
        lpipss.append(lpips(rendering, gt.cuda(), net_type='vgg').mean().item())
        rendering_times.append(t_end - t_start)
        gnums.append(gaussians_hlod._xyz.shape[0])
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
    time_per_frame = sum(rendering_times[5:])/len(rendering_times[5:]) if len(rendering_times) > 5 else sum(rendering_times)/len(rendering_times)
    fps = 1/time_per_frame
    mean_gaussian_num = np.mean(gnums)
    max_gaussian_num = np.max(gnums)
    gpu_name = torch.cuda.get_device_name()
    peak_allocated = torch.cuda.max_memory_allocated()
    memory_used = peak_allocated / 1024**2  # memory usage in MB
    with open(os.path.join(model_path, name, "metrics.txt"), "a") as file:
        file.write(f"\n({gpu_name} hlod_{''.join([str(i) for i in range(lod_max, lod_min-1, -1)])}_sc{screensize}_perView) PSNR: {torch.tensor(psnrs).mean().item():.5f} / SSIM: {torch.tensor(ssims).mean().item():.5f} / LPIPS: {torch.tensor(lpipss).mean().item():.5f}\n")
        file.write(f"mean Gnum: {mean_gaussian_num} / max Gnum: {max_gaussian_num} / peak memory {memory_used}MB / FPS: {fps:.5f}s\n")


def render_sets_predetermined_selective_rendering(args, model, pipeline):
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    gaussian_model_args = {
        'sh_degree': dataset.sh_degree,
        'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
        'lod_scaling_ratio': dataset.lod_scaling_ratio,
        'increase_lod_num_childs': dataset.increase_lod_num_childs,
        'current_lod': args.hlod_max,
        'max_lod': args.lod,
    }
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if not args.skip_train:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.hlod_max, shuffle=False, load_only_test_images=False, load_image_device=args.load_image_device, load_ply=False)
        render_set_predetermined_selective_rendering(dataset.model_path, "train", scene.getTrainCameras(), pipeline_params, background, gaussian_model_args, args.hlod_max, args.hlod_min, args.hlod_screensize, args.n_frames)

    if not args.skip_test:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.hlod_max, shuffle=False, load_only_test_images=True, load_image_device=args.load_image_device, load_ply=False)
        render_set_predetermined_selective_rendering(dataset.model_path, "test", scene.getTestCameras(), pipeline_params, background, gaussian_model_args, args.hlod_max, args.hlod_min, args.hlod_screensize, args.n_frames)


def render_set_predetermined_selective_rendering(model_path, name, views, pipeline, background, gaussian_model_args, lod_max, lod_min, screensize, n_frames):

    lod_range = '-'.join(str(i) for i in range(lod_max, lod_min-1, -1))
    hlod_name = f"selective_{lod_range}_sc{screensize:.1f}_predetermined"
    render_path = os.path.join(model_path, name, "renders", f"{hlod_name}")
    gt_path = os.path.join(model_path, name, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    
    gaussians_cpu = GaussianModel(**gaussian_model_args)
    gaussians_hlod = compose_hlod_gaussian_for_all_views(gaussians_cpu, model_path, views, lod_max=lod_max, lod_min=lod_min, pixel_size=screensize)

    gaussians_hlod.to_cuda()
    gaussians_hlod.active_sh_degree = gaussians_hlod.max_sh_degree

    # Start tracking memory usage 
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache() 

    if args.save_video:
        video_dir = os.path.join(model_path, name, "renders", "videos")
        makedirs(video_dir, exist_ok=True)
        video_cameras = generate_path(views, n_frames=n_frames)
        frames = []
        for idx, view in enumerate(tqdm(video_cameras, desc="Rendering video trajectory")):
            rasterize_results = render(view, gaussians_hlod, pipeline, background)
            rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
            frame = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        video_filename = f"{hlod_name}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        create_video_from_frames(frames, out_path=video_path)
        return

    # When not saving video: measure metrics and save images
    rendering_times = []
    gnums = []
    ssims, psnrs, lpipss = [], [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t_start = time.time()
        rasterize_results = render(view, gaussians_hlod, pipeline, background)
        torch.cuda.synchronize(); t_end = time.time()
        rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
        if hasattr(view, "alpha_mask"):
            rendering *= view.alpha_mask.cuda()
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        ssims.append(ssim(rendering, gt.cuda()).mean().item())
        psnrs.append(psnr(rendering, gt.cuda()).mean().item())
        lpipss.append(lpips(rendering, gt.cuda(), net_type='vgg').mean().item())
        rendering_times.append(t_end - t_start)
        gnums.append(gaussians_hlod._xyz.shape[0])
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
    time_per_frame = sum(rendering_times[5:])/len(rendering_times[5:]) if len(rendering_times) > 5 else sum(rendering_times)/len(rendering_times)
    fps = 1/time_per_frame
    mean_gaussian_num = np.mean(gnums)
    max_gaussian_num = np.max(gnums)
    gpu_name = torch.cuda.get_device_name()
    peak_allocated = torch.cuda.max_memory_allocated()
    memory_used = peak_allocated / 1024**2  # memory usage in MB
    with open(os.path.join(model_path, name, "metrics.txt"), "a") as file:
        file.write(f"\n({gpu_name} hlod_{''.join([str(i) for i in range(lod_max, lod_min-1, -1)])}_sc{screensize}_predetermined) PSNR: {torch.tensor(psnrs).mean().item():.5f} / SSIM: {torch.tensor(ssims).mean().item():.5f} / LPIPS: {torch.tensor(lpipss).mean().item():.5f}\n")
        file.write(f"mean Gnum: {mean_gaussian_num} / max Gnum: {max_gaussian_num} / peak memory {memory_used}MB / FPS: {fps:.5f}s\n")


def render_sets_single_level(args, model, pipeline):
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    gaussian_model_args = {
        'sh_degree': dataset.sh_degree,
        'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
        'lod_scaling_ratio': dataset.lod_scaling_ratio,
        'increase_lod_num_childs': dataset.increase_lod_num_childs,
        'current_lod': args.render_lod,
        'max_lod': args.lod,
    }
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if not args.skip_train:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.render_lod, shuffle=False, load_only_test_images=False, load_image_device=args.load_image_device, load_ply=False)
        render_set_single_level(dataset.model_path, "train", scene.loaded_iter, args.render_lod, scene.getTrainCameras(), pipeline_params, background, gaussian_model_args, args.n_frames)
    if not args.skip_test:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.render_lod, shuffle=False, load_only_test_images=True, load_image_device=args.load_image_device, load_ply=False)
        render_set_single_level(dataset.model_path, "test", scene.loaded_iter, args.render_lod, scene.getTestCameras(), pipeline_params, background, gaussian_model_args, args.n_frames)


def render_set_single_level(model_path, name, iteration, render_lod, views, pipeline, background, gaussian_model_args, n_frames):
    render_path = os.path.join(model_path, name, "renders", f"level{render_lod}")
    gt_path = os.path.join(model_path, name, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    
    gaussian_select_lod = GaussianModel(**gaussian_model_args)
    gaussian_select_lod.load(model_path=model_path, load_iteration=iteration, load_lod=render_lod, device="cpu")

    gaussian_select_lod.to_cuda()
    
    # Start tracking memory usage 
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache() 

    if args.save_video:
        video_dir = os.path.join(model_path, name, "renders", "videos")
        makedirs(video_dir, exist_ok=True)
        video_cameras = generate_path(views, n_frames=n_frames)
        frames = []
        for idx, view in enumerate(tqdm(video_cameras, desc="Rendering video trajectory")):
            rasterize_results = render(view, gaussian_select_lod, pipeline, background)
            rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
            frame = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        video_filename = f"level{render_lod}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        create_video_from_frames(frames, out_path=video_path)
        return

    # When not saving video: measure metrics and save images
    rendering_times = []
    gnums = []
    ssims, psnrs, lpipss = [], [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t_start = time.time()
        rasterize_results = render(view, gaussian_select_lod, pipeline, background)
        torch.cuda.synchronize(); t_end = time.time()
        rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
        if hasattr(view, "alpha_mask"):
            rendering *= view.alpha_mask.cuda()
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        ssims.append(ssim(rendering, gt.cuda()).mean().item())
        psnrs.append(psnr(rendering, gt.cuda()).mean().item())
        lpipss.append(lpips(rendering, gt.cuda(), net_type='vgg').mean().item())
        rendering_times.append(t_end - t_start)
        gnums.append(gaussian_select_lod._xyz.shape[0])
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
    time_per_frame = sum(rendering_times[5:])/len(rendering_times[5:]) if len(rendering_times) > 5 else sum(rendering_times)/len(rendering_times)
    fps = 1/time_per_frame
    mean_gaussian_num = np.mean(gnums)
    max_gaussian_num = np.max(gnums)
    gpu_name = torch.cuda.get_device_name()
    peak_allocated = torch.cuda.max_memory_allocated()
    memory_used = peak_allocated / 1024**2  # memory usage in MB
    with open(os.path.join(model_path, name, "metrics.txt"), "a") as file:
        file.write(f"\n({gpu_name} level{render_lod}) PSNR: {torch.tensor(psnrs).mean().item():.5f} / SSIM: {torch.tensor(ssims).mean().item():.5f} / LPIPS: {torch.tensor(lpipss).mean().item():.5f}\n")
        file.write(f"mean Gnum: {mean_gaussian_num} / max Gnum: {max_gaussian_num} / peak memory {memory_used}MB / FPS: {fps:.5f}s\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument('--render_mode', choices=['selective_rendering_per_view', 'selective_rendering_predetermined', 'single_level'], required=True, help='Rendering mode to use.')
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--lod', default=5, type=int, help="The maximum LOD value set during training")
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--load_image_device', default='cuda', type=str)    # cpu if low VRAM memory
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--n_frames', default=240, type=int)

    # HLOD options
    parser.add_argument('--hlod_max', default=5, type=int)
    parser.add_argument('--hlod_min', default=3, type=int)
    parser.add_argument('--hlod_screensize', default=1.0, type=float)

    # Single LOD options
    parser.add_argument('--render_lod', default=5, type=int)

    args = get_combined_args(parser)

    print(f"Rendering {args.model_path} with mode {args.render_mode}")
    safe_state(args.quiet)
    
    if args.render_mode == 'selective_rendering_per_view':
        render_sets_per_view_selective_rendering(args, model, pipeline)

    elif args.render_mode == 'selective_rendering_predetermined':
        render_sets_predetermined_selective_rendering(args, model, pipeline)

    elif args.render_mode == 'single_level':
        render_sets_single_level(args, model, pipeline)

    else:
        raise ValueError(f"Unknown render_mode: {args.render_mode}")    