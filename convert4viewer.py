import os
from os import makedirs
import numpy as np
import math
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from plyfile import PlyData, PlyElement

def compute_projection_scale_and_distance(lod_scaling_limit, fovx, fovy, width, height, pixel_size=1.0):
        
    # Compute the focal lengths from field of view
    focal_length_x = width / (2.0 * math.tan(fovx / 2.0))
    focal_length_y = height / (2.0 * math.tan(fovy / 2.0))

    # Use the smaller of the two focal lengths to ensure visibility in both dimensions
    focal_length = min(focal_length_x, focal_length_y)
    
    lod_effective_diameter = lod_scaling_limit
    max_distance = focal_length * lod_effective_diameter / pixel_size
    
    return max_distance

def compose_hlod_and_save(path, gaussians, views, lod_max, lod_min=1, pixel_size=1.0):
        
    scaling_ratio = gaussians.lod_scaling_ratio
    
    camera_center = torch.stack([view.camera_center for view in views]).mean(dim=0)
    
    view = views[0]
    lod_scaling_limit = gaussians.get_lod_scaling_lower_bound(lod_max-1)
    lod_max_dist_upper_bound = compute_projection_scale_and_distance(lod_scaling_limit, view.FoVx, view.FoVy, view.image_width, view.image_height, pixel_size=pixel_size)
    lod_max_dist_lower_bound = 0
    
    gaussians.parent_lod_gaussian_stack
    xyz, opacity, scaling, rotation, features_dc, features_rest = [], [], [], [], [], []
    
    for lod in range(lod_max, lod_min-1, -1):
        
        if lod == gaussians.current_lod:
            xyz_lod = gaussians._xyz
            opacity_lod = gaussians._opacity
            scaling_lod = gaussians.scaling_inverse_activation(gaussians.get_lod_scaling(lod))
            rotation_lod = gaussians._rotation
            features_dc_lod = gaussians._features_dc
            features_rest_lod = gaussians._features_rest
            
        else: 
            xyz_lod = gaussians.parent_lod_gaussian_stack[f'lod_{lod}']['xyz']
            opacity_lod = gaussians.parent_lod_gaussian_stack[f'lod_{lod}']['opacity']
            scaling_lod = gaussians.scaling_inverse_activation(gaussians.get_lod_scaling(lod))
            rotation_lod = gaussians.parent_lod_gaussian_stack[f'lod_{lod}']['rotation']
            features_dc_lod = gaussians.parent_lod_gaussian_stack[f'lod_{lod}']['features_dc']
            features_rest_lod = gaussians.parent_lod_gaussian_stack[f'lod_{lod}']['features_rest']
            
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
        features_dc.append(features_dc_lod[lod_mask])
        features_rest.append(features_rest_lod[lod_mask])
        
        lod_max_dist_lower_bound = lod_max_dist_upper_bound
        lod_max_dist_upper_bound *= scaling_ratio 
        
    xyz = torch.cat(xyz).detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    opacity = torch.cat(opacity).detach().cpu().numpy()
    scaling = torch.cat(scaling).detach().cpu().numpy()
    rotation = torch.cat(rotation).detach().cpu().numpy()
    features_dc = torch.cat(features_dc).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    features_rest = torch.cat(features_rest).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gaussians)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, features_dc, features_rest, opacity, scaling, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def adjust_scale(gaussians):
    gaussians._scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling)
    return gaussians

def delete_unneccesary_attributes(gaussians):
    keep_attrs =  {'_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'}
    attr_to_delete = [attr for attr in vars(gaussians) if attr not in keep_attrs]
    for attr in attr_to_delete:
        delattr(gaussians, attr)

def construct_list_of_attributes(gaussians):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(gaussians._features_dc.shape[1]*gaussians._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(gaussians._features_rest.shape[1]*gaussians._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gaussians._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gaussians._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(gaussians, path):
    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gaussians)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def convert(dataset : ModelParams, iteration : int, lod : int):

    gaussian_model_args = {
        'sh_degree'                 : dataset.sh_degree,
        'lod1_scaling_lower_bound'  : dataset.lod1_scaling_lower_bound, 
        'lod_scaling_ratio'         : dataset.lod_scaling_ratio,
        'increase_lod_num_childs'   : dataset.increase_lod_num_childs,
        'current_lod'               : lod, 
        'max_lod'                   : lod,
    }            

    point_cloud_viewer_path = os.path.join(dataset.model_path, 'point_cloud_viewer')
    makedirs(point_cloud_viewer_path, exist_ok=True)
    
    level_gaussians_path = os.path.join(point_cloud_viewer_path, 'level')
    makedirs(level_gaussians_path, exist_ok=True)
    
    for level in range(1, lod+1):
        gaussian_model_args['current_lod'] = level
        gaussians = GaussianModel(**gaussian_model_args)
        scene = Scene(dataset, gaussians=gaussians, load_iteration=iteration, load_lod=level, shuffle=False, load_image_device="cpu", load_only_test_images=True)
        gaussians = adjust_scale(gaussians)
        delete_unneccesary_attributes(gaussians)
        save_path = os.path.join(level_gaussians_path, f"level_{level}_point_cloud.ply")
        save_ply(gaussians, save_path)

    selective_subset_path = os.path.join(point_cloud_viewer_path, 'selective')
    makedirs(selective_subset_path, exist_ok=True)

    for level in range(3, lod+1):
        gaussian_model_args['current_lod'] = level
        gaussians = GaussianModel(**gaussian_model_args)
        scene = Scene(dataset, gaussians=gaussians, load_iteration=iteration, load_lod=level, shuffle=False, load_image_device="cuda")
        
        save_path = os.path.join(selective_subset_path, f"levels_{level}_{level-1}_{level-2}_point_cloud.ply")
        gaussians_hlod = compose_hlod_and_save(save_path, gaussians, scene.getTrainCameras(), lod_max=level, lod_min=level-2, pixel_size=1.0)


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--lod", default=5, type=int, help="The maximum LOD value set during training")
    args = get_combined_args(parser)

    print("Converting " + args.model_path)

    convert(model.extract(args), args.iteration, args.lod)

# python convert4viewer.py -m [model_path]