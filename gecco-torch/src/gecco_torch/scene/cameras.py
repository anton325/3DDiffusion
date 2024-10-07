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
from torch import nn
import numpy as np
from gecco_torch.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_depth_map, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        try:
            self.gt_depth_map = torch.from_numpy(gt_depth_map)
        except:
            self.gt_depth_map = torch.from_numpy(np.array(gt_depth_map))
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def blender_to_world_units(self,**kwargs):
        def conversion(x):
            max_val = -kwargs['depth_information']['offset']+1/kwargs['depth_information']['size']
            if x == 255:
                # return zero instead of max_val, assumption of Gsplat renderer is that everything thats far away is depth 0
                return 0
            elif x < -kwargs['depth_information']['offset']:
                return 0
            else:
                # inverse the linear function 255/1.5 (x-1)
                return x/(255*kwargs['depth_information']['size']) - (kwargs['depth_information']['offset'])
        
        # max distance in world koordinaten
        max_val = -kwargs['depth_information']['offset']+1/kwargs['depth_information']['size']
        # numpy_array = self.gt_depth_map.cpu().numpy()

        # # Open a file in write mode
        # with open(f'gt_depth_map_{1}.txt', 'w') as f:
        #     # Use `np.savetxt` to print the NumPy array to the file
        #     np.savetxt(f, numpy_array, fmt='%s')
        # print(max_val)
        # print(160/(255*kwargs['depth_information']['size']))
        # print(160/(255*kwargs['depth_information']['size']) - (kwargs['depth_information']['offset']))
        self.gt_depth_map = torch.clamp(self.gt_depth_map/(255*kwargs['depth_information']['size']) - (kwargs['depth_information']['offset']),min=-kwargs['depth_information']['offset'],max=max_val)
        self.gt_depth_map[self.gt_depth_map <= -kwargs['depth_information']['offset']] = 0
        self.gt_depth_map[self.gt_depth_map >= max_val] = 0
        # numpy_array = self.gt_depth_map.cpu().numpy()
        # with open(f'gt_depth_map_{2}.txt', 'w') as f:
        #     # Use `np.savetxt` to print the NumPy array to the file
        #     np.savetxt(f, numpy_array, fmt='%s')
        # self.gt_depth_map = [[conversion(x) for x in row] for row in self.gt_depth_map]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

