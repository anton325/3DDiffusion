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
import pickle
import random
import json
from gecco_torch.utils.system_utils import searchForMaxIteration
from gecco_torch.scene.dataset_readers import sceneLoadTypeCallbacks
from gecco_torch.scene.gaussian_model import GaussianModelTemplate
# from scene.gaussian_model_original import GaussianModel
from gecco_torch.arguments import ModelParams
from gecco_torch.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModelTemplate

    def __init__(self, args : ModelParams, gaussians : GaussianModelTemplate, load_iteration=None, shuffle=True, resolution_scales=[1.0], **kwargs):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.circle_cameras = {}
        self.spiral_cameras = {}
        self.val_cameras = {}

        # print("load dataset")
        # print(f"Recognizing dataset {args.source_path}")
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender/synthetic nerf data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,**kwargs)
        elif os.path.isdir(os.path.join(args.source_path, "rgb")) and os.path.isdir(os.path.join(args.source_path, "pose")):
            print("Found rgb and pose directory, assuming SRN data set!")
            scene_info = sceneLoadTypeCallbacks['SRN'](args.source_path, args.white_background, args.eval)
        elif os.path.isdir(os.path.join(args.source_path, "train")) and os.path.isdir(os.path.join(args.source_path, "val")) and os.path.isdir(os.path.join(args.source_path, "test")):
            print("Found train and val and test directory, assuming shapenet rendered dataset!")
            scene_info = sceneLoadTypeCallbacks['ShapenetRender'](args.source_path, args.white_background, args.eval,**kwargs)
        elif os.path.exists(os.path.join(args.source_path, "cameras.npz")):
            print("Found cameras.npz file, assuming NMR data set!")
            scene_info = sceneLoadTypeCallbacks['NMR'](args.source_path, args.white_background, args.eval)
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
            if scene_info.circle_cameras:
                camlist.extend(scene_info.circle_cameras)
            if scene_info.val_cameras:
                camlist.extend(scene_info.val_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file: # all cameras
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # print(resolution_scales)
            # print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            # print("Loading Circle Cameras")
            self.circle_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.circle_cameras, resolution_scale, args)
            # print("Loading Spiral Cameras")
            self.spiral_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.spiral_cameras, resolution_scale, args)
            # print("Loading Eval Cameras")
            self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.pcd_init_with_depth_map, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        point_cloud_path_pkl = os.path.join(self.model_path, "point_cloud/iteration_{}.pkl".format(iteration))
        
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # with open(point_cloud_path_pkl,"wb") as f:
        #     pickle.dump(self.gaussians,f)

    def convert_gt_depth_map(self,**kwargs):
        for scale in self.train_cameras:
            [cam.blender_to_world_units(**kwargs) for cam in self.train_cameras[scale]]

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getCircleCameras(self, scale=1.0):
        return self.circle_cameras[scale]

    def getSpiralCameras(self, scale=1.0):
        return self.spiral_cameras[scale]

    def getValCameras(self, scale=1.0):
        return self.val_cameras[scale]