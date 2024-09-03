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
import random
import json
from utils.system_utils import searchForMaxIterationGivenLod, searchForMaxLod, checkFileSize
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import matplotlib.pyplot as plt
from matplotlib import cm

import torch
COLORMAP = cm.get_cmap('tab10')

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, 
                 load_iteration=None, load_lod=None,
                 shuffle=True, resolution_scales=[1.0], load_only_test_images=False, load_image_device="cuda", load_by_image_names=None, load_ply=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.loaded_lod = None
        self.gaussians = gaussians

        print(f"Memory allocated at scene init: {torch.cuda.memory_allocated() / 1024**2} MB")
        
        if load_iteration:
            if load_lod != -1:
                self.loaded_lod = load_lod
            else: # search for max lod at either -1 or None for load_lod value
                self.loaded_lod = searchForMaxLod(os.path.join(self.model_path, "point_cloud"))
                                
            print("Loading trained model at Lod {}, iteration {}".format(self.loaded_lod, self.loaded_iter))
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIterationGivenLod(os.path.join(self.model_path, "point_cloud"), self.loaded_lod)
            else:
                self.loaded_iter = load_iteration
                            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, load_ply=load_ply)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Cannot find points.pcd file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, load_ply=load_ply)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras) 
            random.shuffle(scene_info.test_cameras) 

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        if load_by_image_names is not None:
            filter_train_cameras = [camera for camera in scene_info.train_cameras if camera.image_name in load_by_image_names]        

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(filter_train_cameras, resolution_scale, args, load_image_device)
                
        else:
            if load_only_test_images:
                for resolution_scale in resolution_scales:
                    print("Loading Test Cameras")
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, load_image_device)

            else:
                for resolution_scale in resolution_scales:
                    print("Loading Training Cameras")
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, load_image_device)    
                    print("Loading Test Cameras")
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, load_image_device)

        print(f"Memory allocated at scene loaded cameras: {torch.cuda.memory_allocated() / 1024**2} MB")

        # an option to not load gaussians (only load dataset)
        if self.gaussians != None:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            f"lod_{self.loaded_lod}_iteration_{self.loaded_iter}",
                                                            "point_cloud.ply"))
                self.gaussians.load_parent_lod_gaussian_stack(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            f"lod_{self.loaded_lod}_iteration_{self.loaded_iter}",
                                                            "ancestry.pt"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

            print(f"Memory allocated at scene load gaussians: {torch.cuda.memory_allocated() / 1024**2} MB")


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def save(self, iteration, lod):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/lod_{}_iteration_{}".format(lod, iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_parent_lod_gaussian_stack(os.path.join(point_cloud_path, "ancestry.pt"))
