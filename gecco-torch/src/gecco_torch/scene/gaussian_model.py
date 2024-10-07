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
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from gecco_torch.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_cosine_scheduler
# from gecco_torch.utils.lie_utils import quaternion_to_lie_liecasadi, lie_to_quaternion_liecasadi
from torch import nn
import time
from gecco_torch.utils.system_utils import mkdir_p
from plyfile import PlyData
from gecco_torch.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from gecco_torch.utils.graphics_utils import BasicPointCloud
from gecco_torch.utils.general_utils import strip_symmetric, build_scaling_rotation
from gecco_torch.scene.gaussian_model_template import GaussianModelTemplate

IDENTIFIER = "gaussian_blobs"

class GaussianModel(GaussianModelTemplate):
    def __init__(self, sh_degree : int):
        super(GaussianModel, self).__init__(sh_degree)

    @property
    def get_scaling(self):
        scaling = torch.clamp(self._scaling, min=-9,max=7)
        return self.scaling_activation(scaling)
    
    def create_from_values(self, xyz, features_dc, opacity, scaling, rotation):
        self._xyz = xyz
        # self _features soll (num points, 1, 3) haben 
        if features_dc is not None:
            self._features_dc = features_dc.reshape(-1,1,3) # .transpose(1, 2)
        else:
            self._features_dc = None
        
        self._opacity = opacity.reshape(-1,1)
        self._scaling = scaling
        self._rotation = rotation
        self._features_rest = torch.zeros((xyz.shape[0],15,3),device=xyz.device) # (num points, 15, 3)
    
    

    def create_from_pcd(self, pcd : BasicPointCloud, pcd_init_with_depth_map:bool, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        # initialization: We estimate the initial covariance matrix as an isotropic Gaussian with axes equal to the mean of the distance to the closest three points.
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1 # quaternion hat 3 imaginäre, 1 realen part, den auf 1 stellen???

        if pcd_init_with_depth_map:
            # scales *= 1.05
            opacities = inverse_sigmoid(0.9*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # wrapper to torch them know that these variables are to be optimized during training
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # contiguous -> one contiguous block in memory without gaps
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # no longer optimize rotation
        self._rotation = nn.Parameter(rots.requires_grad_(True))
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
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if kwargs['scheduler'] == "exp":
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        elif kwargs['scheduler'] == "cos":
            self.xyz_scheduler_args = get_cosine_scheduler(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                           lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                           max_steps=training_args.position_lr_max_steps)
        else:
            raise NotImplementedError(f"Scheduler {kwargs['scheduler']} not implemented")
        # for i in range(30000):
        #     print(self.xyz_scheduler_args(i))


    def load_ply(self, path):
        # print("load ply")
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
        # features_extra = np.zeros_like(features_extra)

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
        # rots = [quaternion_to_lie(rot) for rot in rots]
        def process_rot(rot):
            # Apply quaternion to lie conversion
            lie_rot = quaternion_to_lie_liecasadi(rot)
            # Convert back from lie to quaternion
            quaternion_rot = lie_to_quaternion_liecasadi(lie_rot)
            return quaternion_rot
        # t = time.time()
        # rots = [quaternion_to_lie_liecasadi(rot) for rot in rots]
        # print(f"To lie space took {time.time()-t} seconds for {len(rots)} many rotations") # 8.7s for 71000 rotations -> 0.5s per 4000 rotations
        # t = time.time()
        # rots = [lie_to_quaternion_liecasadi(rot) for rot in rots] 
        # print("back took {} seconds".format(time.time()-t)) # 18.4s for 71000 rotations -> 1s per 4000 rotations
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     # Schedule the processing of each rotation for parallel execution
        #     futures = [executor.submit(process_rot, rot) for rot in rots]

        #     # Collect the results as they are completed
        #     rots_transformed = [future.result() for future in as_completed(futures)]
        # # rots = [lie_to_quaternion(rot) for rot in rots]
        # rots = rots_transformed

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def theoretical_densify_and_clone(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        print(f"selected_pts_mask {sum(selected_pts_mask)}")
        # print(f"densify and clone shape grads {grads.shape}") # (number_points,1)
        # print(f"densify and clone shape selected_pts_mask {selected_pts_mask.shape}")
        # print(f"densify and clone number true selected_pts_mask {sum(selected_pts_mask)}")
        # finde under-reconstruction indem wir schauen, ob die scalings groß sind (wir suchen nach kleinen scalings)
        # weil große scalings stehen für große gaussians und die finden wir an sich gut, die wollen wir in dem Bereich lassen
        # Wir wollen die kleinen gaussians clonen, um diesen bereich besser abzubilden
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        print(f"density and clone würde {sum(selected_pts_mask)} neue Punkt erstellen")
        return sum(selected_pts_mask)
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        under-reconstruction -> clone, under populated area. ihnen ist aufgefallen, dass wenn es relativ leer ist
        die gaussians die das sind einen hohen view-space gradienten haben, weil durch das
        zu wenig da sein die region nicht so gut reconstructed ist
        """
        # Extract points that satisfy the gradient condition -> großer gradient -> under reconstructed
        # wegen under population oder over reconstruction
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # print(f"densify and clone shape grads {grads.shape}") # (number_points,1)
        # print(f"densify and clone shape selected_pts_mask {selected_pts_mask.shape}")
        # print(f"densify and clone number true selected_pts_mask {sum(selected_pts_mask)}")
        
        # finde under-reconstruction indem wir schauen, ob die scalings groß sind (wir suchen nach kleinen scalings)
        # weil große scalings stehen für große gaussians und die finden wir an sich gut, die wollen wir in dem Bereich lassen
        # Wir wollen die kleinen gaussians clonen, um diesen bereich besser abzubilden
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print("densify and clone number of true masks {} out of {}".format(sum(selected_pts_mask),selected_pts_mask.shape))
        # print(f"densify and clone number true selected_pts_mask p2 {sum(selected_pts_mask)}")
        new_xyz = self._xyz[selected_pts_mask]
        # print(f"densify and clone shape new_xyz {new_xyz.shape}")
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        # print(f"after densification in clone xyz shape {self._xyz.shape}")


    def theoretical_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # number of points
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        print(f"Densify and split würde {sum(selected_pts_mask)} hinzufügen")
        # print(f"Densify and split würde {sum(selected_pts_mask)} hinzufügen und {sum(prune_filter)} eliminieren (Summe {sum(selected_pts_mask)-sum(prune_filter)})")
        # die hinzugefügt werden, werden doppelt hinzugefügt und die alten werden gelöscht, also müsste man theoretisch 2*sum(selected_pts_mask)-sum(prune_filter) rechnen
        # aber sum(selected_pts_mask) == sum(prune_filter) weil wir sozusagen die alten punkte jetzt doppelt haben
        return sum(selected_pts_mask)#-sum(prune_filter)

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
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        # print(f"Densify and split number new points {new_xyz.shape}")
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # print(f"Densify and split prune mask true {sum(prune_filter)}")
        self.prune_points(prune_filter)


    def model_opacities_logarithmically(self,start_opacity, num_points=100):
        x = np.linspace(0, 190, num_points)
        y = []
        for x_in in x:
            y.append(0.175 * np.log(x_in + 1) + start_opacity)
        return y
    
    def model_opacities_sigmoid(self,start,end,start_opacity, num_points=100):
        x = torch.linspace(start, end, num_points)
        y = []
        # 0.95\cdot\frac{1}{1+e^{-\left(x-10\right)}}+0.005
        # for x_in in x:
        #     y.append(0.9949999999 * 1 / (1 + np.exp(-(3*x_in - 10))) + start_opacity)
        # return y
        return (0.99499999 * 1 / (1 + torch.exp(-(3*x - 10))) + start_opacity).cuda()
    
    def prune_until_number_of_points_fast(self,min_opacity,extent,max_screen_size,path,iteration,prune_down_to_number):
        # print(min_opacity) # original min opacity 0.005
        print("prune fast")
        current_number_of_points = self.get_xyz.shape[0]
        pruned_away = 0
        # with open(pathlib.Path(path,f"opacities_{iteration}.pkl"),"wb") as f:
        #     pickle.dump(self.get_opacity.cpu().numpy(),f)

        if current_number_of_points - pruned_away > prune_down_to_number:
            # es gibt bedarf hartes prunen zu betreiben

            # threshold opacity
            threshold = 0.99
            percentage_of_opacities_over_threshold = sum(self.get_opacity.squeeze() > threshold)/len(self.get_opacity)
            print(f"percentage over threshold {percentage_of_opacities_over_threshold}")
            if percentage_of_opacities_over_threshold > 0.98:
                print("nearly all opacities ones, prune certain percantage randomly")
                length = len(self.get_opacity)
                percentage_to_delete_randomly = 1 - prune_down_to_number/length
                num_ones = int(length * percentage_to_delete_randomly)
                prune_mask = torch.zeros(length, dtype=torch.bool,device=self.get_opacity.device)
                prune_mask[:num_ones] = 1
                prune_mask = prune_mask[torch.randperm(length)]

            else:
                print(f"Not all opacities 1, prune down to {prune_down_to_number}")

                num_prune_opacities_to_try = 200
                sigmoid_schedule_opacities = self.model_opacities_sigmoid(5,15,min_opacity, num_points=num_prune_opacities_to_try)

                opacities = self.get_opacity.squeeze()
                opacities = (opacities-opacities.min())/(opacities.max()-opacities.min())
                print(opacities.shape)
                print(sigmoid_schedule_opacities.shape)
                opacities_bigger = opacities>sigmoid_schedule_opacities[:,None]
                cummulated = opacities_bigger.sum(1)
                index_opacities = num_prune_opacities_to_try - sum(cummulated<prune_down_to_number)
                
                min_opacity = sigmoid_schedule_opacities[index_opacities]

                prune_mask = (self.get_opacity < min_opacity).squeeze()
                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            print(prune_mask.dtype)
            print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
            self.prune_points(prune_mask)
            print("End of densify and prune shape of xyz {}".format(self.get_xyz.shape))
            torch.cuda.empty_cache()
        else:
            print(f"mit {min_opacity} würden {pruned_away} Punkte wegfallen (Insgesamt anzahl punkte {current_number_of_points-pruned_away}, Ziel {prune_down_to_number})")
            print("Kein prune down hard bedarf")

    # closed form solution
    def prune_until_number_of_points(self,min_opacity,extent,max_screen_size,path,iteration,prune_down_to_number):
        # print(min_opacity) # original min opacity 0.005
        current_number_of_points = self.get_xyz.shape[0]
        pruned_away = 0
        # with open(pathlib.Path(path,f"opacities_{iteration}.pkl"),"wb") as f:
        #     pickle.dump(self.get_opacity.cpu().numpy(),f)

        if current_number_of_points - pruned_away > prune_down_to_number:
            # es gibt bedarf hartes prunen zu betreiben

            threshold = 0.98
            # print(self.get_opacity.squeeze())
            percentage_of_opacities_over_threshold = sum(self.get_opacity.squeeze() > threshold)/len(self.get_opacity)
            # print(f"percentage over thrershold {percentage_of_opacities_over_threshold}")
            if percentage_of_opacities_over_threshold > 0.99:
                # print("nearly all opacities ones, prune certain percantage randomly")
                self.prune_randomly(prune_down_to_number)

            else:
                # print(f"Not all opacities 1, prune down to {prune_down_to_number}")

                num_prune_opacities_to_try = 500
                sigmoid_schedule_opacities = self.model_opacities_sigmoid(0,10,min_opacity, num_points=num_prune_opacities_to_try)

                opacities = self.get_opacity.squeeze()
                opacities = (opacities-opacities.min())/(opacities.max()-opacities.min())
                # print(opacities.shape)
                # print(sigmoid_schedule_opacities.shape)
                opacities_bigger = opacities>sigmoid_schedule_opacities[:,None]
                cummulated = opacities_bigger.sum(1)
                index_opacities = num_prune_opacities_to_try - sum(cummulated<prune_down_to_number)
                
                min_opacity = sigmoid_schedule_opacities[index_opacities]
                prune_mask = (self.get_opacity < min_opacity).squeeze()
                # print(f"prune away {prune_mask.sum()}")
                if current_number_of_points-prune_mask.sum() < 1000:
                    index_opacities = index_opacities - 1
                    min_opacity = sigmoid_schedule_opacities[index_opacities]
                    prune_mask = (self.get_opacity < min_opacity).squeeze()
                    # print(f"too much, prune away {prune_mask.sum()}")

                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                # print(prune_mask.dtype)
                # print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
                self.prune_points(prune_mask)
                # print("End of densify and prune shape of xyz {}".format(self.get_xyz.shape))
                torch.cuda.empty_cache()
                current_number_of_points = self.get_xyz.shape[0]
                if current_number_of_points > prune_down_to_number:
                    # print(f"there are still too many, prune the rest randomly")
                    self.prune_randomly(prune_down_to_number)
        else:
            # print(f"mit {min_opacity} würden {pruned_away} Punkte wegfallen (Insgesamt anzahl punkte {current_number_of_points-pruned_away}, Ziel {prune_down_to_number})")
            # print("Kein prune down hard bedarf")
            pass

    def prune_randomly(self,prune_down_to_number):
        # print("prune randomly")
        
        length = len(self.get_opacity)
        # print(length)
        # print(prune_down_to_number)
        percentage_to_delete_randomly = 1 - prune_down_to_number/length
        # print(percentage_to_delete_randomly)
        num_ones = int(length * percentage_to_delete_randomly)
        prune_mask = torch.zeros(length, dtype=torch.bool,device=self.get_opacity.device)
        prune_mask[:num_ones] = 1
        prune_mask = prune_mask[torch.randperm(length)]
        # print(prune_mask.dtype)
        # print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
        self.prune_points(prune_mask)
        # print("End of prune randomly shape of xyz {}".format(self.get_xyz.shape))
        torch.cuda.empty_cache()

    # feature: binary search
    def prune_until_number_of_points_binary_search(self,min_opacity,extent,max_screen_size,path,iteration,prune_down_to_number):
        # print(min_opacity) # original min opacity 0.005
        current_number_of_points = self.get_xyz.shape[0]
        pruned_away = 0
        # with open(pathlib.Path(path,f"opacities_{iteration}.pkl"),"wb") as f:
        #     pickle.dump(self.get_opacity.cpu().numpy(),f)

        if current_number_of_points - pruned_away > prune_down_to_number:
            # es gibt bedarf hartes prunen zu betreiben
            # exponential schedule
            # increase = 1.01
            # min_opacity = min_opacity * 1/increase # hier abziehen weil in weil addieren wir immer VORHER drauf
            chosen_opacity_index = 0

            threshold = 0.9999
            percentage_of_opacities_over_threshold = sum(self.get_opacity.squeeze() > threshold)/len(self.get_opacity)

            if percentage_of_opacities_over_threshold > 0.9999:
                print("nearly opacities ones, prune certain percantage randomly")
                length = len(self.get_opacity)
                percentage_to_delete_randomly = 1 - prune_down_to_number/length
                num_ones = int(length * percentage_to_delete_randomly)
                prune_mask = torch.zeros(length, dtype=torch.bool,device=self.get_opacity.device)
                prune_mask[:num_ones] = 1
                prune_mask = prune_mask[torch.randperm(length)]

            else:
                print("Not all opacities 1")

                log_schedule_opacities = self.model_opacities_sigmoid(min_opacity, num_points=200)

                # schedule_opacities = 
                L = 0
                R = len(log_schedule_opacities)-1
                chosen_opacity_index = int(np.floor((L+R)/2))
                while L<=R:
                    print("Beginning binary search")
                    print(f"L {L}")
                    print(f"R {R}")
                    print(f"chosen_opacity_index {chosen_opacity_index}")
                    min_opacity = log_schedule_opacities[chosen_opacity_index]
                    prune_mask = (self.get_opacity < min_opacity).squeeze()
                    if max_screen_size:
                        big_points_vs = self.max_radii2D > max_screen_size
                        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                    pruned_away = sum(prune_mask)
                    print(f"mit {min_opacity} würden {pruned_away} Punkte wegfallen (Insgesamt anzahl punkte {current_number_of_points-pruned_away}, Ziel {prune_down_to_number})")
                    
                    if current_number_of_points - pruned_away > prune_down_to_number:
                        L = chosen_opacity_index + 1
                    elif current_number_of_points - pruned_away <= prune_down_to_number:
                        R = chosen_opacity_index - 1
                    if L > R:
                        break
                    chosen_opacity_index = int(np.floor((L+R)/2))


                # if current_number_of_points - pruned_away < 10:
                #     print("Würde zu viele Punkte wegprunen, deshalb einen Schritt (zwei, weil wir ja grad noch einen draufgerechnet haben) zurück")
                #     min_opacity = log_schedule_opacities[chosen_opacity_index-2]

                prune_mask = (self.get_opacity < min_opacity).squeeze()
                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            print(prune_mask.dtype)
            print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
            self.prune_points(prune_mask)
            print("End of densify and prune shape of xyz {}".format(self.get_xyz.shape))
            torch.cuda.empty_cache()
        else:
            print(f"mit {min_opacity} würden {pruned_away} Punkte wegfallen (Insgesamt anzahl punkte {current_number_of_points-pruned_away}, Ziel {prune_down_to_number})")
            print("Kein prune down hard bedarf")
            
    def prune_until_number_of_points_orig(self,min_opacity,extent,max_screen_size,path,iteration,prune_down_to_number):
        # print(min_opacity) # original min opacity 0.005
        current_number_of_points = self.get_xyz.shape[0]
        pruned_away = 0
        # with open(pathlib.Path(path,f"opacities_{iteration}.pkl"),"wb") as f:
        #     pickle.dump(self.get_opacity.cpu().numpy(),f)

        if current_number_of_points - pruned_away > prune_down_to_number:
            # es gibt bedarf hartes prunen zu betreiben
            # exponential schedule
            # increase = 1.01
            # min_opacity = min_opacity * 1/increase # hier abziehen weil in weil addieren wir immer VORHER drauf
            chosen_opacity_index = 0

            if all(self.get_opacity.squeeze() == 1):
                print("all opacities ones, prune certain percantage randomly")
                length = len(self.get_opacity)
                percentage_to_delete_randomly = 1 - prune_down_to_number/length
                num_ones = int(length * percentage_to_delete_randomly)
                prune_mask = torch.zeros(length, dtype=torch.bool,device=self.get_opacity.device)
                prune_mask[:num_ones] = 1
                prune_mask = prune_mask[torch.randperm(length)]

            else:
                print("Not all opacities 1")
                log_schedule_opacities = self.model_opacities_sigmoid(min_opacity, num_points=200)
                while chosen_opacity_index < len(log_schedule_opacities):
                    min_opacity = log_schedule_opacities[chosen_opacity_index]
                    prune_mask = (self.get_opacity < min_opacity).squeeze()
                    if max_screen_size:
                        big_points_vs = self.max_radii2D > max_screen_size
                        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                    pruned_away = sum(prune_mask)
                    # print(f"mit {min_opacity} würden {pruned_away} Punkte wegfallen (Insgesamt anzahl punkte {current_number_of_points-pruned_away}, Ziel {prune_down_to_number})")
                    chosen_opacity_index+=1
                    if current_number_of_points - pruned_away < prune_down_to_number:
                        break
                if current_number_of_points - pruned_away < 10:
                    # print("Würde zu viele Punkte wegprunen, deshalb einen Schritt (zwei, weil wir ja grad noch einen draufgerechnet haben) zurück")
                    min_opacity = log_schedule_opacities[chosen_opacity_index-2]

                prune_mask = (self.get_opacity < min_opacity).squeeze()
                if max_screen_size:
                    big_points_vs = self.max_radii2D > max_screen_size
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            print(prune_mask.dtype)
            # print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
            self.prune_points(prune_mask)
            # print("End of densify and prune shape of xyz {}".format(self.get_xyz.shape))
            torch.cuda.empty_cache()
        # else:
        #     print(f"mit {min_opacity} würden {pruned_away} Punkte wegfallen (Insgesamt anzahl punkte {current_number_of_points-pruned_away}, Ziel {prune_down_to_number})")
        #     print("Kein prune down hard bedarf")
            


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # the new parameters of network (left over after pruning) have to come from
        # the optimizer, so that theyre still nn.Parameters and thus included in optimization

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # füge die neuen tensors ein in die alten (cat(alt,neu))
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
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

if __name__ == "__main__":
    print("hi")