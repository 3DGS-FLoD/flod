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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, ssim_error_map
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel 
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


from utils.training_utils import expand_list_to_match_lods
from matplotlib import cm
import time
import torchvision


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):    
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussian_model_args = {
        'sh_degree'                 : dataset.sh_degree,
        'lod1_scaling_lower_bound'  : dataset.lod1_scaling_lower_bound,     # initial scale constraint (tau)
        'lod_scaling_ratio'         : dataset.lod_scaling_ratio,            # scale factor (rho)
        'increase_lod_num_childs'   : dataset.increase_lod_num_childs,
        'current_lod'               : opt.lod_min,
        'max_lod'                   : opt.lod_max,                          # L_max
        'use_voxel_sampling'        : dataset.use_voxel_sampling,
        'voxel_sampling_size'       : dataset.voxel_sampling_size
    }
    
    gaussians = GaussianModel(**gaussian_model_args)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # training hyperparameters
    lods = opt.lod_max - opt.lod_min + 1
        
    densify_grad_thresholds = expand_list_to_match_lods(opt.densify_grad_threshold, lods)
    densification_intervals = expand_list_to_match_lods(opt.densification_interval, lods)
    densify_from_iters = expand_list_to_match_lods(opt.densify_from_iter, lods)
    densify_until_iters = expand_list_to_match_lods(opt.densify_until_iter, lods)
    
    prune_opacity_thresholds = expand_list_to_match_lods(opt.prune_opacity_threshold, lods)
    pruning_intervals = expand_list_to_match_lods(opt.pruning_interval, lods)
    prune_from_iters = expand_list_to_match_lods(opt.prune_from_iter, lods)
    prune_until_iters = expand_list_to_match_lods(opt.prune_until_iter, lods)
    
    prune_overlap_thresholds = expand_list_to_match_lods(opt.prune_overlap_threshold, lods)
    pruning_overlap_intervals = expand_list_to_match_lods(opt.pruning_overlap_interval, lods)
    
    opacity_reset_intervals = expand_list_to_match_lods(opt.opacity_reset_interval, lods)

    lambda_dssims = expand_list_to_match_lods(opt.lambda_dssim, lods)        

    # level-by-level training (Sec 4.2)
    # progression to the next level occurs only after the completion of the current level’s training
    for i, current_lod in enumerate(range(opt.lod_min, opt.lod_max + 1)):
        iterations = opt.lod_iterations[i]
        first_iter = 0
        progress_bar = tqdm(range(first_iter, iterations), desc=f"Training progress LoD:{current_lod}")
        first_iter += 1
        
        # training hyperparameters for each level
        densify_grad_threshold = densify_grad_thresholds[i]
        densification_interval = densification_intervals[i]
        densify_from_iter = densify_from_iters[i]
        densify_until_iter = densify_until_iters[i]
        
        prune_opacity_threshold = prune_opacity_thresholds[i]
        pruning_interval = pruning_intervals[i]
        prune_from_iter = prune_from_iters[i]
        prune_until_iter = prune_until_iters[i]
        
        prune_overlap_threshold = prune_overlap_thresholds[i]
        pruning_overlap_interval = pruning_overlap_intervals[i]
        
        opacity_reset_interval = opacity_reset_intervals[i]
        
        lambda_dssim = lambda_dssims[i]

        # increase lod
        if current_lod > opt.lod_min:
            gaussians.increase_lod()
            gaussians.training_setup(opt)
        
        # train for current lod
        for iteration in range(first_iter, iterations + 1):        
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
                
            # time render() 
            start_time = time.time()
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            end_time = time.time()
            execution_time = end_time - start_time
            
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
            
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations) or iteration == opt.lod_iterations[i]:
                    scene.save(iteration, current_lod)

                # Densification
                if iteration < densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                    if iteration > densify_from_iter and iteration % densification_interval == 0:
                        gaussians.densify(densify_grad_threshold, scene.cameras_extent)
                        
                # Pruning
                if iteration < prune_until_iter:
                    if iteration > prune_from_iter and iteration % pruning_interval == 0:
                        size_threshold = 20 if (iteration > opacity_reset_interval and current_lod >= opt.prune_max_radii2D_lod) else None 
                        
                        #* prune overlapping gaussians with a different interval
                        if iteration % pruning_overlap_interval == 0:
                            gaussians.prune(prune_opacity_threshold, scene.cameras_extent, size_threshold, prune_overlap_threshold) # overlap pruning (Sec 4.3)
                        else:
                            gaussians.prune(prune_opacity_threshold, scene.cameras_extent, size_threshold, 0.0) # just pruning

                # Opacity Reset 
                if iteration < prune_until_iter:
                    if iteration % opacity_reset_interval == 0 or (dataset.white_background and iteration == densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")       

                
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
