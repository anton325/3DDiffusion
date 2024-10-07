from torch import nn, optim
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import lietorch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_plotting import visualize_so3_density, visualize_so3_probabilities
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.utils.build_cov_matrix_torch import build_covariance_from_activated_scaling_rotation, build_covariance_from_scaling_rotation_xyzw, strip_lowerdiag
from gecco_torch.utils.riemannian_helper_functions import L_to_cov_x6, find_cholesky_L, group_operation_add, group_operation_inverse_L, exp_map_at_L, log_map_at_L, is_in_L_plus, geodesic_step
from gecco_torch.additional_metrics.metrics_so3 import c2st_gaussian, best_fit_geodesic_distance, geodesic_distance
import math
import json
from torchvision.utils import save_image
from gecco_torch.gaussian_renderer import render
from typing import NamedTuple


class LogUniformSchedule(nn.Module):
    """
    LogUniform noise schedule which seems to work better in our (GECCO) context.

    alle schedules returnen einfach nur für jedes n ein sigma, 
    sie werden gecalled mit schedule(samples) und samples hat shape (batchsize, num_points, 3)
    und dann gibt er für jedes element im batch ein sigma
    """

    def __init__(self, max: float, min: float = 0.002, low_discrepancy: bool = True):
        super().__init__()

        self.sigma_min = min
        self.sigma_max = max
        self.log_sigma_min = math.log(min)
        self.log_sigma_max = math.log(max)
        self.low_discrepancy = low_discrepancy

    def return_schedule(self,n):
        u = torch.linspace(0,1,n).cuda()
        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma

class SimpleMLP(nn.Module):
    def __init__(self,input_size=7): # input size: 6 von der rotationsmatrix, 1 von variance
        super(SimpleMLP, self).__init__()
        neurons = 512
        self.mlp = nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
        )
        self.layer_L = nn.Linear(neurons, 6)

    def forward(self, x, s):
        concat = torch.cat([x, s], dim=-1)
        out = self.mlp(concat)

        L = self.layer_L(out)

        # exponentieren, damit die diagonale safe positiv ist
        L[:,:3] = torch.exp(L[:,:3])
        # print(torch.isnan(L).sum())
        return L
    
@torch.no_grad()
def heun_sampler_L(model, gt_L, noise_schedule):
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_L.shape[0]).vec().to(gt_L.device)
    noisy_scale = torch.randn(gt_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t = noisy_start_L.cuda()

    beginning_dist = best_fit_geodesic_distance(gt_L, noisy_start_L)
    print(f"Beginning distance: {beginning_dist}") # 65_000

    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]
        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0
        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)

        denoised_L = model(x_t, noise_level)
        
        inverse_L_i = group_operation_inverse_L(x_t)

        group_op_res = group_operation_add(inverse_L_i, denoised_L)

        d_i = log_map_at_L(group_op_res, x_t) / t_i

        scaled_d_i = exp_map_at_L(d_i * (t_iminus - t_i), x_t)
        x_next = group_operation_add(x_t, scaled_d_i)

        if t_iminus != 0:
            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            denoised_L_2nd = model(x_next, noise_level)

            inverse_L_i_next = group_operation_inverse_L(x_next)

            group_op_res_2nd = group_operation_add(inverse_L_i_next, denoised_L_2nd)

            d_i_strich = log_map_at_L(group_op_res_2nd, x_next) / t_iminus

            d_i_strichs = (t_iminus - t_i) * group_operation_add(d_i / 2, d_i_strich / 2)

            x_next = group_operation_add(x_t, exp_map_at_L(d_i_strichs, x_next))
            
            step_dist = best_fit_geodesic_distance(gt_L, x_next)
            print(f"iteration {i} distance: {step_dist}") # 65_000
        x_t = x_next
    return x_t

@torch.no_grad()
def heun_sampler_L_vanilla(model, gt_L, noise_schedule):
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_L.shape[0]).vec().to(gt_L.device)
    noisy_scale = torch.randn(gt_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)

    noise_schedule = torch.flip(noise_schedule,[0])
    L_i = noisy_start_L.cuda()

    beginning_dist = best_fit_geodesic_distance(gt_L, noisy_start_L)
    print(f"Beginning distance: {beginning_dist}") # 65_000

    for i in range(len(noise_schedule)):
        t_cur = noise_schedule[i]
        t_next = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0

        noise_level = t_cur * torch.ones([L_i.shape[0],1],device=L_i.device)

        denoised_L = model(L_i, noise_level)
        
        inverse_L_i = group_operation_inverse_L(L_i)

        group_op_res = group_operation_add(inverse_L_i, denoised_L)

        d_i = log_map_at_L(group_op_res, L_i) / t_cur

        scaled_d_i = exp_map_at_L(d_i * (t_next - t_cur), L_i)
        L_next = group_operation_add(L_i, scaled_d_i)

        step_dist = best_fit_geodesic_distance(gt_L, L_next)
        print(f"iteration {i} distance: {step_dist}") # 65_000

        if t_next != 0:
            noise_level = t_next * torch.ones([L_i.shape[0],1],device=L_i.device)
            denoised_L_2nd = model(L_next, noise_level)

            inverse_L_i_next = group_operation_inverse_L(L_next)

            group_op_res_2nd = group_operation_add(inverse_L_i_next, denoised_L_2nd)

            d_i_strich = log_map_at_L(group_op_res_2nd, L_i) / t_next

            d_i_strichs = (t_next - t_cur) * group_operation_add(d_i / 2, d_i_strich / 2)

            L_next = group_operation_add(L_i, exp_map_at_L(d_i_strichs, L_i))

            step_dist = best_fit_geodesic_distance(gt_L, L_next)
            print(f"iteration {i} distance: {step_dist}") # 65_000
        L_i = L_next
    return L_i

@torch.no_grad()
def heun_sampler_L_logdirection(model, gt_L, noise_schedule):
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_L.shape[0]).vec().to(gt_L.device)
    noisy_scale = torch.randn(gt_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)

    noise_schedule = torch.flip(noise_schedule,[0])
    L_i = noisy_start_L.cuda()

    beginning_dist = best_fit_geodesic_distance(gt_L, noisy_start_L)
    print(f"Beginning distance: {beginning_dist}") # 65_000

    for i in range(len(noise_schedule)):
        t_cur = noise_schedule[i]
        t_next = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0

        noise_level = t_cur * torch.ones([L_i.shape[0],1],device=L_i.device)

        denoised_L = model(L_i, noise_level)
        
        step_dist = best_fit_geodesic_distance(gt_L, denoised_L)
        print(f"iteration {i} denoised distance: {step_dist}") # 65_000

        direction = log_map_at_L(denoised_L, L_i) / t_cur
        L_next = exp_map_at_L(-(t_next - t_cur) * direction, L_i)
        step_dist = best_fit_geodesic_distance(gt_L, L_next)
        print(f"iteration {i} step distance: {step_dist}") # 65_000


        if t_next != 0:
            noise_level = t_next * torch.ones([L_i.shape[0],1],device=L_i.device)
            denoised_L_2nd = model(L_next, noise_level)

            direction_2 = log_map_at_L(denoised_L_2nd, L_next) / t_next

            L_next = exp_map_at_L(-(t_next - t_cur) * (direction + direction_2) / 2, L_i)

            step_dist = best_fit_geodesic_distance(gt_L, L_next)
            print(f"iteration {i} 2nd order distance: {step_dist}") # 65_000
        L_i = L_next
    return L_i

@torch.no_grad()
def heun_sampler_L_geodesic_step(model, gt_L, noise_schedule):
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_L.shape[0]).vec().to(gt_L.device)
    noisy_scale = torch.randn(gt_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)

    beginning_dist = best_fit_geodesic_distance(gt_L, noisy_start_L)
    print(f"Beginning distance: {beginning_dist}") # 65_000

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t = noisy_start_L.cuda()

    original = x_t.clone()

    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]
        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        denoised_L = model(x_t, noise_level)

        # print(f"Distanz original zu aktuellem Fortschritt: \n{geodesic_distance(x_t,original)}")

        inverse_L_i = group_operation_inverse_L(x_t)

        group_op_res = group_operation_add(inverse_L_i, denoised_L)

        d_i = log_map_at_L(group_op_res, x_t) / t_i

        x_next = geodesic_step(x_t, d_i, -(t_iminus - t_i))

            
        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_next,original)}")
        # print(f"Distanz Fortschritt mit Schritt zu in dieser Runde vorhergesagtem x_0: \n {geodesic_distance(x_next,denoised_L)}")
        # print(" ")
        

        if i != len(noise_schedule) - 1:
            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            denoised_L_2nd = model(x_next, noise_level)

            inverse_L_i_next = group_operation_inverse_L(x_next)

            group_op_res_2nd = group_operation_add(inverse_L_i_next, denoised_L_2nd)

            d_i_strich = log_map_at_L(group_op_res_2nd, x_next) / t_iminus

            d_i_strichs = d_i + d_i_strich

            x_next_strich = geodesic_step(x_t, d_i_strichs, -(t_iminus - t_i))

            # print(f"Distanz original zu Fortschritt Strich: \n{geodesic_distance(x_next_strich,original)}")
            # print(f"Distanz Schritt zu Strich vorhergesagt: \n {geodesic_distance(x_next,denoised_L_2nd)}")
            # print(f"Distanz Schritt Strich zu Strich vorhergesagt:\n{geodesic_distance(x_next_strich,denoised_L_2nd)}")

            step_dist = best_fit_geodesic_distance(gt_L, x_next_strich)
            print(f"iteration {i} distance: {step_dist}") # 65_000
            x_next = x_next_strich

        x_t = x_next
    return x_t


path = Path("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/utils/out/")
path.mkdir(exist_ok=True, parents=False)

gm = GaussianModel(3)
gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
rotations_plane = gm.get_rotation.detach()[:500]
scaling_plane = gm.get_scaling.detach()[:500]

model = SimpleMLP()
src = Path("/home/giese/Documents/gecco/so3models/so3ddpm","gaus_rot_cholesky_small_2024-06-04_15:23:29._vexp", "model")
state_dict_path = src / "gaus_rot_cholesky_small_vexp_model-160000.pckl"
# state_dict_path = Path("/home/giese/Documents/gecco/so3models/so3ddpm","gaus_rot_cholesky_2024-05-29_15:26:14._vexp","model","gaus_rot_cholesky_vexp_model-20000.pckl")
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)
model = model.cuda()

gt_cov_matrices = build_covariance_from_activated_scaling_rotation(scaling_plane, rotations_plane)
gt_L = find_cholesky_L(gt_cov_matrices)

schedule = LogUniformSchedule(165)
noise_schedule = schedule.return_schedule(128)
# noise_schedule = torch.cat([torch.zeros(1).cuda(),noise_schedule])
print(noise_schedule)

sampled_Ls = heun_sampler_L_logdirection(model, gt_L, noise_schedule)
# sampled_Ls = heun_sampler_L(model, gt_L, noise_schedule)
# sampled_Ls = heun_sampler_L_geodesic_step(model, gt_L, noise_schedule)

sum_distance, row_ind, col_ind = best_fit_geodesic_distance(gt_L, sampled_Ls, return_indices=True)
print(f"sum_distance: {sum_distance}")
closest_sampled_L = sampled_Ls[row_ind]
closest_sampled_covs = L_to_cov_x6(closest_sampled_L)

with open("/home/giese/Documents/gaussian-splatting/circle_cams.json","r") as f:
    circle_cams = json.load(f)  

class Camera(NamedTuple):
    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    tanfovx: float
    tanfovy: float
    imsize: int

img_path = Path(src ,'vis' )
img_path.mkdir(parents=True, exist_ok=True)
for i in circle_cams.keys():
    cam = circle_cams[i]
    camera = Camera(
        world_view_transform = torch.tensor(cam["world_view_transform"]).cuda(),#.unsqueeze(0),
        projection_matrix = torch.tensor(cam['projection_matrix']).cuda(),#.unsqueeze(0),
        tanfovy = 0.45714,
        tanfovx = 0.45714,
        imsize=400,
    )
    bg = torch.tensor([1.0,1.0,1.0], device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
    # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
    kwargs = {
        'use_cov3D' : True,
        'covs' : closest_sampled_covs #closest_sampled_covs
    }
    with torch.no_grad():
        render_dict = render(gm,bg,camera = camera, **kwargs)
        img = render_dict['render']
        save_image(img, img_path / f'_render_{i}.png')
b = 3