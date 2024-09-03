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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import random

from utils.system_utils import searchForMaxIterationGivenLod, searchForMaxLod, checkFileSize



class GaussianModel:

    # Adjust the scale constraint according to the current level (Sec 4.1 eq.3)
    def setup_scaling_activation(self):
        if self.current_lod < self.max_lod:
            self.scaling_lower_bound = self.lod1_scaling_lower_bound / self.lod_scaling_ratio ** (self.current_lod - 1)
        else:
            # scale constraint is 0 at maximum level
            self.scaling_lower_bound = 0.0
            
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.setup_scaling_activation()
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
            
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.shape_activation = torch.sigmoid


    def __init__(self, 
                 sh_degree                  : int,
                 lod1_scaling_lower_bound   : float,
                 lod_scaling_ratio          : float,   
                 increase_lod_num_childs    : int = 1,                 
                 current_lod                : int = 1,
                 max_lod                    : int = 5,
                 use_voxel_sampling         : bool = False,
                 voxel_sampling_size        : float = 0.2,
                 ):
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.lod1_scaling_lower_bound = lod1_scaling_lower_bound
        self.lod_scaling_ratio = lod_scaling_ratio
        self.increase_lod_num_childs = increase_lod_num_childs
        
        self.current_lod = current_lod
        self.max_lod = max_lod
        
        self._ancestry = torch.empty(0)
        
        self.parent_lod_gaussian_stack = {}
        
        self.use_voxel_sampling = use_voxel_sampling
        self.voxel_sampling_size = voxel_sampling_size
        
        self.setup_functions()
        
    def to_cpu(self):
        self._xyz = self._xyz.to("cpu")
        self._features_dc = self._features_dc.to("cpu")
        self._features_rest = self._features_rest.to("cpu")
        self._opacity = self._opacity.to("cpu")
        self._scaling = self._scaling.to("cpu")
        self._rotation = self._rotation.to("cpu")
        
    def to_cuda(self):
        self._xyz = self._xyz.to("cuda")
        self._features_dc = self._features_dc.to("cuda")
        self._features_rest = self._features_rest.to("cuda")
        self._opacity = self._opacity.to("cuda")
        self._scaling = self._scaling.to("cuda")
        self._rotation = self._rotation.to("cuda")
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
           
    @property
    def get_scaling(self):
        # 3D Gaussians’ scale at current level (Sec 4.2 eq.4) 
        # the 3D scale of the Gaussians is constrained to be larger or equal to scaling_lower_bound
        return self.scaling_activation(self._scaling) + self.scaling_lower_bound
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def take_hlod_attributes(self,
                             xyz,
                             opacity,
                             scaling,
                             rotation,
                             features
                             ):
        
        self._xyz = xyz
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        
        self._features_dc = features[:, :1]
        self._features_rest = features[:, 1:]

        self.scaling_lower_bound = 0.0
        self.scaling_activation = nn.Identity()
        self.opacity_activation = nn.Identity()
        self.rotation_activation = nn.Identity()
    
    
    def get_lod_xyz(self, lod):
        if lod != self.current_lod:
            return self.parent_lod_gaussian_stack[f'lod_{lod}']['xyz']
        else:
            return self._xyz
    
    def get_lod_scaling(self, lod):
        if lod != self.current_lod:
            scaling = self.parent_lod_gaussian_stack[f'lod_{lod}']['scaling']
            scaling_lower_bound = self.parent_lod_gaussian_stack[f'lod_{lod}']['scaling_lower_bound']
            return self.scaling_activation(scaling) + scaling_lower_bound
        else:
            return self.scaling_activation(self._scaling) + self.scaling_lower_bound
            
    def get_lod_scaling_lower_bound(self, lod):
        if lod != self.current_lod:
            return self.parent_lod_gaussian_stack[f'lod_{lod}']['scaling_lower_bound']
        else:
            return self.scaling_lower_bound  
        
    def get_lod_attributes(self, lod):
        if lod != self.current_lod:

            xyz = self.parent_lod_gaussian_stack[f'lod_{lod}']['xyz']
            opacity = self.parent_lod_gaussian_stack[f'lod_{lod}']['opacity']
            scaling = self.parent_lod_gaussian_stack[f'lod_{lod}']['scaling']
            rotation = self.parent_lod_gaussian_stack[f'lod_{lod}']['rotation']
            features_dc = self.parent_lod_gaussian_stack[f'lod_{lod}']['features_dc']
            features_rest = self.parent_lod_gaussian_stack[f'lod_{lod}']['features_rest']
            scaling_lower_bound = self.parent_lod_gaussian_stack[f'lod_{lod}']['scaling_lower_bound']
            
            opacity = self.opacity_activation(opacity)
            scaling = self.scaling_activation(scaling) + scaling_lower_bound
            rotation = self.rotation_activation(rotation)
            features = torch.cat((features_dc, features_rest), dim=1)
            
        else:

            xyz = self._xyz
            opacity = self.opacity_activation(self._opacity)
            scaling = self.scaling_activation(self._scaling) + self.scaling_lower_bound
            rotation = self.rotation_activation(self._rotation)
            features = torch.cat((self._features_dc, self._features_rest), dim=1)
            
        return xyz, opacity, scaling, rotation, features

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def voxelize_sample(self, data=None, voxel_size=0.01):
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        
        self.spatial_lr_scale = spatial_lr_scale
        
        if self.use_voxel_sampling:
            voxel_points = np.round(np.asarray(pcd.points)/self.voxel_sampling_size)
            unique_voxel_points = np.unique(voxel_points, axis=0)
            sampled_indices = [random.choice(np.where((voxel_points == voxel_point).all(axis=1))[0]) for voxel_point in unique_voxel_points]
            fused_point_cloud = torch.tensor(np.asarray(pcd.points[sampled_indices])).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors[sampled_indices])).float().cuda())
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[sampled_indices])).float().cuda()), 0.0000001)

        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        dist = torch.clamp_min(torch.sqrt(dist2) - self.lod1_scaling_lower_bound, 0.0000001) # adjust to scaling lower bound
        scales = torch.log(dist)[...,None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
       
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
            
        if self._ancestry.nelement() != 0:
            for i in range(self._ancestry.shape[1]):
                l.append('ancestry_lod_{}'.format(i))
        return l
    
    def load(self, model_path, load_lod=None, load_iteration=None, device="cuda"):
        
        self.model_path = model_path
    
        if load_lod != -1:
            self.loaded_lod = load_lod
        else: # search for max lod at either -1 or None for load_lod value
            self.loaded_lod = searchForMaxLod(os.path.join(self.model_path, "point_cloud"))
                            
        if load_iteration == -1:
            self.loaded_iter = searchForMaxIterationGivenLod(os.path.join(self.model_path, "point_cloud"), self.loaded_lod)
        else:
            self.loaded_iter = load_iteration

        self.load_ply(os.path.join(self.model_path,
                                    "point_cloud",
                                    f"lod_{self.loaded_lod}_iteration_{self.loaded_iter}",
                                    "point_cloud.ply"), device=device)
        
        self.load_parent_lod_gaussian_stack(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           f"lod_{self.loaded_lod}_iteration_{self.loaded_iter}",
                                                           "ancestry.pt"), device=device)
        

        self.current_lod = self.loaded_lod
        self.setup_scaling_activation()
    
    def save_parent_lod_gaussian_stack(self, path):
        torch.save(self.parent_lod_gaussian_stack, path)
        
    def load_parent_lod_gaussian_stack(self, path, device="cuda"):    
        self.parent_lod_gaussian_stack = torch.load(path, map_location=device)
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        ancestry = self._ancestry.float().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        if self._ancestry.nelement() != 0:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, ancestry), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def load_ply(self, path, device="cuda"):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        anc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ancestry_lod_")]
        if len(anc_names) > 0:
            anc_names = sorted(anc_names, key = lambda x: int(x.split('_')[-1]))
            ancestry = np.zeros((xyz.shape[0], len(anc_names)))
            for idx, attr_name in enumerate(anc_names):
                ancestry[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._ancestry = torch.tensor(ancestry, dtype=torch.long, device=device)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))            

        self.active_sh_degree = self.max_sh_degree
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state  

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._ancestry = self._ancestry[valid_points_mask] if self._ancestry.nelement() != 0 else self._ancestry

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ancestry):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._ancestry = torch.cat([self._ancestry, new_ancestry], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(torch.clamp_min(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N) - self.scaling_lower_bound, 0.0000001))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_ancestry = self._ancestry[selected_pts_mask].repeat(N,1) if self._ancestry.nelement() != 0 else self._ancestry

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_ancestry)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
                
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_ancestry = self._ancestry[selected_pts_mask] if self._ancestry.nelement() != 0 else self._ancestry
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_ancestry)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        

    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)     
           
    def prune(self, min_opacity, extent, max_screen_size, prune_overlap_threshold):
                
        prune_mask = (self.get_opacity < min_opacity).squeeze()
                
        if prune_overlap_threshold: # overlap pruning (Sec 4.3)
            # eliminate Gaussians whose average distance to its 3 nearest neighbors falls below a predefined distance threshold
            dist = torch.sqrt(distCUDA2(self._xyz))
            prune_overlap_mask = (dist < prune_overlap_threshold)
            prune_mask = torch.logical_or(prune_mask, prune_overlap_mask)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        
        torch.cuda.empty_cache()

    def capture_parent_gaussian(self):
        parent_gaussian = {
            'xyz'                           : self._xyz,
            'features_dc'                   : self._features_dc,
            'features_rest'                 : self._features_rest,
            'scaling'                       : self._scaling,
            'rotation'                      : self._rotation,
            'opacity'                       : self._opacity,    
            'ancestry'                      : self._ancestry,
            'max_radii2D'                   : self.max_radii2D,
            'scaling_lower_bound'           : self.scaling_lower_bound,
        }
        
        return parent_gaussian

    def restore_parent_gaussian(self, lod):
        if lod != self.current_lod: 
            parent_gaussian = self.parent_lod_gaussian_stack[f'lod_{lod}']
            
            self._xyz = parent_gaussian['xyz']
            self._features_dc = parent_gaussian['features_dc']
            self._features_rest = parent_gaussian['features_rest']
            self._scaling = parent_gaussian['scaling']
            self._rotation = parent_gaussian['rotation']
            self._opacity = parent_gaussian['opacity']
            self._ancestry = parent_gaussian['ancestry']
            self.max_radii2D = parent_gaussian['max_radii2D']
            self.scaling_lower_bound = parent_gaussian['scaling_lower_bound']
    
    def stack_parent_gaussian(self):
        self.parent_lod_gaussian_stack[f'lod_{self.current_lod}'] = self.capture_parent_gaussian()
    
    #* choose one hierarchy of gaussians based on parent gaussian at selected lod
    def select_one_family(self, lod, family_index):
        if self._ancestry.nelement() == 0:
            select_family_index = torch.full((self._xyz.shape[0],), False, dtype=torch.bool)
            select_family_index[family_index] = True
        else: 
            select_family_index = self._ancestry[:,lod-1] == family_index
               
        self._xyz = self._xyz[select_family_index]
        self._features_dc = self._features_dc[select_family_index]
        self._features_rest = self._features_rest[select_family_index]
        self._scaling = self._scaling[select_family_index]
        self._rotation = self._rotation[select_family_index]
        self._opacity = self._opacity[select_family_index]

    def increase_lod(self):
        #* stack parent lod gaussians
        # Upon transitioning to the next level, clones of the Gaussians from the current level are created and saved as the final Gaussians of current level
        self.stack_parent_gaussian()
        
        #* present values for splitting
        N = self.increase_lod_num_childs 
        parent_scaling = self.get_scaling
        self.current_lod += 1
        self.setup_scaling_activation() 
        
        #* create child gaussian attributes
        # Adjust the scale constraint (Sec 4.2 eq.5) to prevent abrupt changes in scale
        child_scaling = self.scaling_inverse_activation(parent_scaling.repeat(N,1) - self.scaling_lower_bound) 
        child_rotation = self._rotation.repeat(N,1) 
        child_features_dc = self._features_dc.repeat(N,1,1)
        child_features_rest = self._features_rest.repeat(N,1,1)
        child_opacity = self._opacity.repeat(N,1)
        child_xyz = self._xyz.repeat(N,1)
        
        # parent gaussian labels for saving parent-child relationship
        child_parent_index = torch.arange(0, self.get_xyz.shape[0], device="cuda")[:, None].repeat(N,1)
       
        #* set to training parameters
        self._xyz = nn.Parameter(child_xyz.detach().requires_grad_(True))
        self._features_dc = nn.Parameter(child_features_dc.detach().requires_grad_(True))
        self._features_rest = nn.Parameter(child_features_rest.detach().requires_grad_(True))
        self._opacity = nn.Parameter(child_opacity.detach().requires_grad_(True))
        self._scaling = nn.Parameter(child_scaling.detach().requires_grad_(True))
        self._rotation = nn.Parameter(child_rotation.detach().requires_grad_(True))
                               
        if self._ancestry.nelement() == 0:
            self._ancestry = child_parent_index
        else:            
            parent_greatparent_index = self._ancestry.repeat(N, 1)            
            self._ancestry = torch.cat([parent_greatparent_index, child_parent_index], dim=1)
   
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")      
        
        torch.cuda.empty_cache()
