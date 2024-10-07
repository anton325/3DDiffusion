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
from scene.gaussian_model_template import GaussianModelTemplate

IDENTIFIER = "gaussian_sphere"

class GaussianSphereModel(GaussianModelTemplate):
    def __init__(self, sh_degree : int):
        super(GaussianSphereModel, self).__init__(sh_degree)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling).repeat(1,3)

    def create_from_pcd(self, pcd : BasicPointCloud,pcd_init_with_depth_map:bool, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        # only one scale per point
        scales = torch.log(torch.sqrt(dist2))[...,None]
        # initialization: We estimate the initial covariance matrix as an isotropic Gaussian with axes equal to the mean of the distance to the closest three points.
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if pcd_init_with_depth_map:
            # scales *= 1.05
            opacities = inverse_sigmoid(0.9*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # wrapper to torch them know that these variables are to be optimized during training
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # contiguous -> one contiguous block in memory without gaps
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = rots
        #self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args,**kwargs):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},

            # no longer optimize rotation
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def load_ply(self, path):
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
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3 # -3 because the first 3 are not in f_rest but in f_dc
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # still works, now there is only one scale element
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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))

        # no longer rotation optimization
        self._rotation = torch.tensor(rots,dtype=torch.float,device='cuda') # nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def prune_points(self, mask):
        """
        mask sind die punkte, die wir loswerden wollen,
        ~mask sind die punkte, die wir behalten
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        
        # rotation no longer in optimizer
        # if len(self._rotation) <= len(self._scaling):
        #     self._rotation = self._rotation[0:len(self._scaling)] #optimizable_tensors["rotation"]
        # else:
        #     self._rotation = self._rotation[0].repeat(len(self._scaling),1)
        new_rotation = torch.from_numpy(np.zeros((self._scaling.shape[0],4))).float().cuda()
        # print("new rotation ",new_rotation.shape)
        new_rotation[:,:] = self._rotation[0,:]
        # print("new rotation ",new_rotation.shape)
        self._rotation = new_rotation
        # print("new self rotation ",self._rotation.shape)
        # self._rotation = self._rotation[0].repeat(self._scaling.shape[0],1)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling):
        # the new parameters of network (left over after pruning) have to come from
        # the optimizer, so that theyre still nn.Parameters and thus included in optimization
        # print(f"densification postfix start shape rotation gaussians {self._rotation.shape}")
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,}
        # "rotation" : new_rotation}

        # füge die neuen tensors ein in die alten (cat(alt,neu))
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]

        # no longer rotation from optimizer
        # self._rotation = optimizable_tensors["rotation"]
        # print(f"densification postfix mid shape rotation gaussians {self._rotation.shape}")
        # self._rotation = self._rotation[0:len(self._scaling)]
        # print(f"densification postfix end shape rotation gaussians {self._rotation.shape}")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        over-reconstruction -> split, ein gaussian ist zu groß geworden. ihnen ist aufgefallen,
        dass over reconstruction gaussians einen hohen view-space gradienten haben, weil durch das
        zu groß sein die region over-reconstructed ist, aber halt nicht gut reconstructed ist
        """
        # number of points
        n_init_points = self.get_xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # finde over-reconstruction indem wir schauen, ob die scalings groß sind (wir suchen nach großen scalings)
        # weil große scalings -> große gaussians
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print("densify and split number of true masks {} out of {}".format(sum(selected_pts_mask),selected_pts_mask.shape))
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # print(f"densify and split shape rotation gaussians {self._rotation.shape}")
        # if len(self._rotation) <= len(selected_pts_mask):
        #     selected_rotations = self._rotation[0:(sum(selected_pts_mask))] #optimizable_tensors["rotation"]
        # else:
        #     selected_rotations = self._rotation[0].repeat((sum(selected_pts_mask)),1)
            # selected_rotations = torch.zeros((sum(selected_pts_mask),4),device="cuda")
            # selected_rotations[:,:] = self._rotation[0,:]

        # selected_rotations = torch.from_numpy(np.zeros((sum(selected_pts_mask),4))).float().cuda()
        selected_rotations = self._rotation[0].repeat((sum(selected_pts_mask),1))

        rots = build_rotation(selected_rotations).repeat(N,1,1)
        # print(f"densify and split shape rotation gaussians {rots.shape}")
        # print(f"densify and split samples shape gaussians {samples.shape}")
        # print(f"densify and split samples shape gaussians { samples.unsqueeze(-1).shape}")
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # print(f"densify and split scaling shape gaussians {new_scaling.shape}")
        # make new scaling fit (get scaling repeats 3 shapes, we only want one)
        
        new_scaling = new_scaling[:,0].unsqueeze(1)
        # new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling)
        # print(f"After densifcation split shape rotation gaussians {self._rotation.shape}")
        # print(f"After densifcation split shape scales gaussians {self._scaling.shape}")
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        # print(f"After densifcation split after prune shape rotation gaussians {self._rotation.shape}")
        # print(f"After densifcation split after prune shape scales gaussians {self._scaling.shape}")
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        under-reconstruction -> clone, under populated area. ihnen ist aufgefallen, dass wenn es relativ leer ist
        die gaussians die da sind einen hohen view-space gradienten haben, weil durch das
        zu wenig da sein die region nicht so gut reconstructed ist
        """
        # Extract points that satisfy the gradient condition -> großer gradient -> under reconstructed
        # wegen under population oder over reconstruction
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # finde under-reconstruction indem wir schauen, ob die scalings groß sind (wir suchen nach kleinen scalings)
        # weil große scalings stehen für große gaussians und die finden wir an sich gut, die wollen wir in dem Bereich lassen
        # Wir wollen die kleinen gaussians clonen, um diesen bereich besser abzubilden
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print("densify and clone number of true masks {} out of {}".format(sum(selected_pts_mask),selected_pts_mask.shape))
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # print(f"shpe scaling {self._scaling.shape}")
        new_scaling = self._scaling[selected_pts_mask]
        # print(f"shpe new scaling {new_scaling.shape}")
        # print("densify mask shape")
        # print("old rotation shape ",self._rotation.shape)
        # print(selected_pts_mask.shape)
        # print(sum(selected_pts_mask))
        # print("new rotation shape ",new_rotation.shape)

        # new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling)
        # print("after densification in clone rotation shape ",self._rotation.shape)