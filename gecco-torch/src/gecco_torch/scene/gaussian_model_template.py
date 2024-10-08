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
from gecco_torch.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from gecco_torch.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from gecco_torch.utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from gecco_torch.utils.graphics_utils import BasicPointCloud
from gecco_torch.utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModelTemplate:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, t = torch.float32):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation, t)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = sh_degree # 0 früher wars auf 0, aber wir optimieren das model ja nicht insofern setzen wir den aktiv auf 3
        self.max_sh_degree = sh_degree # standard is 3
        self._xyz = torch.empty(0) # the means of all the gaussians
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
        self.setup_functions()

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

    # def restore(self, model_args, training_args):
    #     (self.active_sh_degree,
    #     self._xyz,
    #     self._features_dc,
    #     self._features_rest,
    #     self._scaling,
    #     self._rotation,
    #     self._opacity,
    #     self.max_radii2D,
    #     xyz_gradient_accum,
    #     denom,
    #     opt_dict,
    #     self.spatial_lr_scale) = model_args
    #     self.training_setup(training_args)
    #     self.xyz_gradient_accum = xyz_gradient_accum
    #     self.denom = denom
    #     self.optimizer.load_state_dict(opt_dict)
    
    @property
    def get_xyz_lr(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                return param_group['lr']
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        # return all the means of the gaussians
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1, t = torch.float32):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, t)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # print(f"update learning rate in {iteration} to {lr}")
                return lr
            
    def construct_list_of_attributes(self, save_cov = False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        if not save_cov:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
        if save_cov:
            for i in range(6):
                l.append('cov_{}'.format(i))
        return l

    def save_ply(self, path, cov = None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # print(self._features_rest.shape)
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        if cov is None:
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
        else:
            cov = cov.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_cov = cov is not None)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # for i in [xyz, normals, f_dc, f_rest, opacities, scale, rotation]:
        #     print(i.shape)
        if cov is None:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, cov), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


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


    def model_grads_linear(self,startgrad,num_points=100):
        return torch.linspace(startgrad, 1, num_points).cuda()
        # return torch.linspace(startgrad, 0.005, num_points).cuda()
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,**kwargs):
        # densify nennen sie das löschen von gaussians mit niedrigem alpha (sehr durchsichtige)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # print(max_grad) # original max grad 0.0002
        
        # standartmäßig auf falsch gesetzt
        if kwargs['enforce_hard_limit']:
            # print(f"Aktuelle anzahl: {self.get_xyz.shape[0]}")
            upper_limit = kwargs['n_hard_upper_limit']
            # if kwargs['prune_down_scheduled'] and kwargs['last_value_prune_down_scheduled'] != None:
            #     upper_limit = kwargs['last_value_prune_down_scheduled'] * kwargs['prune_down_scheduled_hard_upper_limit_factor']
            total_num_gaussians = self.get_xyz.shape[0]
            # if total_num_gaussians > upper_limit:
            # how many new gaussians do we allow:
            allowed_additional_gaussians = upper_limit - total_num_gaussians
            num_grads = 200
            min_grad = 0.0002 # 0.0002
            grads_normalized = grads.squeeze()
            min_grad = (min_grad-grads_normalized.min())/(grads_normalized.max()-grads_normalized.min())
            linear_schedule_grads = self.model_grads_linear(min_grad, num_points=num_grads)
            grads_normalized = (grads_normalized-grads_normalized.min())/(grads_normalized.max()-grads_normalized.min())
            # print(grads_normalized.shape)
            # print(linear_schedule_grads.shape)
            grads_bigger = grads_normalized>linear_schedule_grads[:,None]
            cummulated = grads_bigger.sum(1)
            # import matplotlib.pyplot as plt
            # fig,ax = plt.subplots()
            # ax.plot(cummulated.cpu())
            # plt.savefig("grads.png")
            # print(cummulated)

            index_grads = num_grads - (sum(cummulated<allowed_additional_gaussians))
            if index_grads == num_grads:
                index_grads = num_grads-1
            # index_grads = num_grads - sum(cummulated<upper_limit)
            
            max_grad = linear_schedule_grads[index_grads]
            max_grad = max_grad*(grads.squeeze().max()-grads.squeeze().min())+grads.squeeze().min()
            # num_added_gaussians = self.theoretical_densify_and_clone(grads, max_grad, extent)
            # num_added_gaussians += self.theoretical_densify_and_split(grads, max_grad, extent)
            # total_num_gaussians = self.get_xyz.shape[0] + num_added_gaussians
            # print(f"At {max_grad} num added guassians : {num_added_gaussians}, would total to {total_num_gaussians} enforcing upper limit of {upper_limit} \n")
            # print(f"At {max_grad} num added guassians : {num_added_gaussians}, would total to {total_num_gaussians} enforcing upper limit of {upper_limit} \n")
            
            # if total_num_gaussians <= upper_limit:
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # also immer der else teil (das ist der normale teil)
        # else:
        #     self.densify_and_clone(grads, max_grad, extent)
        #     self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
        self.prune_points(prune_mask)
        # print("End of densify and prune shape of xyz {}".format(self.get_xyz.shape))
        torch.cuda.empty_cache()

    def densify_and_prune_original(self, max_grad, min_opacity, extent, max_screen_size,**kwargs):
        # densify nennen sie das löschen von gaussians mit niedrigem alpha (sehr durchsichtige)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # wie initialisiert man hier keine rotation mit einem quaternion = quaternion (0,0,0,1)
        # quaternion hat 3 imaginäre, 1 realen part, den auf 1 stellen

        # print(max_grad) # original max grad 0.0002
        
        # standartmäßig auf falsch gesetzt
        if kwargs['enforce_hard_limit']:
            # print(f"Aktuelle anzahl: {self.get_xyz.shape[0]}")
            upper_limit = kwargs['n_hard_upper_limit']
            # if kwargs['prune_down_scheduled'] and kwargs['last_value_prune_down_scheduled'] != None:
            #     upper_limit = kwargs['last_value_prune_down_scheduled'] * kwargs['prune_down_scheduled_hard_upper_limit_factor']
            total_num_gaussians = 10000000
            max_attempts = 10000
            i_attempts = 0
            if total_num_gaussians > upper_limit:
                descent = 0.0001
                max_grad -= descent # erstmal abziehen weil wir am anfang der while schleife drauf addieren damit wir in der ersten iteration bei original max grad sind

            while total_num_gaussians > upper_limit and i_attempts < max_attempts:
                i_attempts += 1
                num_added_gaussians = 0
                max_grad += descent
                num_added_gaussians += self.theoretical_densify_and_clone(grads, max_grad, extent)
                num_added_gaussians += self.theoretical_densify_and_split(grads, max_grad, extent)
                total_num_gaussians = self.get_xyz.shape[0] + num_added_gaussians
                # print(f"At {max_grad} num added guassians : {num_added_gaussians}, would total to {total_num_gaussians} enforcing upper limit of {upper_limit} \n")
            
            if total_num_gaussians <= upper_limit:
                self.densify_and_clone(grads, max_grad, extent)
                self.densify_and_split(grads, max_grad, extent)


        # also immer der else teil (das ist der normale teil)
        else:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # print(f"Prune points {prune_mask.sum()} and shape of mask is {prune_mask.shape}")
        self.prune_points(prune_mask)
        # print("End of densify and prune shape of xyz {}".format(self.get_xyz.shape))
        torch.cuda.empty_cache()
    


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # print("cat tensors to optimizer group name ",group["name"])
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            # print(f"extension tensor shape {extension_tensor.shape}")
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