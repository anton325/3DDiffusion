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
import math

FORCE_SMALL_SCALE = False

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
    def __init__(self,input_size=10): # input size: 9 von der rotationsmatrix, 1 von variance
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.layer_mu = nn.Linear(256, 6)
        self.layer_scale = nn.Linear(6, 1)

    def forward(self, x, s):

        # konvertiere quaternion zu rotation matrix
        rot_mat = lietorch.SO3(x).matrix()[:,:3,:3]
        rot_mat = rot_mat.reshape(-1,9)

        concat = torch.cat([rot_mat, s], dim=-1)
        out = self.mlp(concat)

        mu = self.layer_mu(out)
        R1 = mu[:,0:3] / torch.norm(mu[:,0:3], dim=-1, keepdim=True)
        R3 = torch.cross(R1, mu[:, 3:], dim=-1)
        R3 = R3 / torch.norm(R3, dim=-1, keepdim=True)
        R2 = torch.cross(R3, R1, dim = -1)

        rotation_matrix = torch.stack([R1,R2,R3],dim=-1)

        quat = lietorch.SO3(rotation_matrix,from_rotation_matrix=True)

        scale = self.layer_scale(mu)
        scale = nn.functional.softplus(scale) + 0.0001
        return quat, scale
    
@torch.no_grad()
def original_sample(model,gt_rotations,noise_schedule):
    # Starting sampling from the trained model
    print("original sampling...")
    X0 = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec()

    def fn_sample(x, delta_mu, s):
        rotated_mu = (delta_mu * lietorch.SO3(x)).vec()
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(rotated_mu, s)
        samples = (lietorch.SO3(rotated_mu) * lietorch.SO3.exp(axis_angles)).vec()
        return samples

    
    x_t = X0.cuda()
    with torch.no_grad():
        for sn in torch.flip(noise_schedule,[0]):
            noise_level = sn*torch.ones([x_t.shape[0],1],device=x_t.device)
            mu, s = model(x_t, noise_level)

            x_t = fn_sample(x_t, mu, s)

    # Remove nans if we accidentally sampled any
    x_t = x_t[~torch.isnan(x_t.sum(axis=-1))]
    return x_t
    
@torch.no_grad()
def sample_heun(model,gt_rotations,noise_schedule):
    print("heun sampling ...")
    X0 = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec()
    
    noise_schedule = torch.flip(noise_schedule,[0])

    x_t = X0.cuda()

    def fn_sample(x, s):
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(x, s)
        samples = (lietorch.SO3(x) * lietorch.SO3.exp(axis_angles))
        return samples

    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        mu, s = model(x_t, noise_level)
        mu = fn_sample(mu.vec(), s)
    
        scaled_rotation = lietorch.SO3.exp(lietorch.SO3(x_t).log()/t_i)
        scaled_mu = lietorch.SO3.exp(mu.log()/t_i).inv()
        dif_rotation = scaled_rotation * scaled_mu

        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0
        x_next = lietorch.SO3(x_t) * lietorch.SO3.exp((t_iminus-t_i) * dif_rotation.log())

        if t_iminus != 0:
            scaled_rotation_2nd = lietorch.SO3.exp(x_next.log()/t_iminus)
            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            mu, s = model(x_next.vec(), noise_level)
            mu = fn_sample(mu.vec(), s)
            scaled_mu_2nd = lietorch.SO3.exp(mu.log()/t_iminus).inv()
            dif_rotation_2nd = scaled_rotation_2nd * scaled_mu_2nd
            x_next = lietorch.SO3(x_t) * lietorch.SO3.exp((t_iminus-t_i) * 
                                            (lietorch.SO3.exp(dif_rotation_2nd.log()/2) * lietorch.SO3.exp(dif_rotation.log()/2)).log()
                                            )
        x_t = x_next.vec()

    return x_t

@torch.no_grad()
def trivial_sample(model,gt_rotations,noise_schedule):
    print("trivial_sampling ...")
    X0 = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec()
    
    noise_schedule = torch.flip(noise_schedule,[0])

    x_t = X0.cuda()

    def fn_sample(x, s):
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(x, s)
        samples = (lietorch.SO3(x) * lietorch.SO3.exp(axis_angles))
        return samples

    for i in range(len(noise_schedule)):
        if i > 0:
            continue
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        mu, s = model(x_t, noise_level)
        mu = fn_sample(mu.vec(), s)
    
        x_t = mu.vec()

    return x_t

# @torch.no_grad()
# def sample_heun(model,gt_rotations,noise_schedule):
#     print("heun sampling mit scaling...")
#     X0 = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec()
    
#     noise_schedule = torch.flip(noise_schedule,[0])

#     x_t = X0.cuda()

#     def fn_sample(x, s):
#         axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
#                                                                     s_single,
#                                                                     force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
#                                 randomness="different")(x, s)
#         samples = (lietorch.SO3(x) * lietorch.SO3.exp(axis_angles))
#         return samples

#     for i in range(len(noise_schedule)):
#         t_i = noise_schedule[i]

#         noise_level = t_i*torch.ones([x_t.shape[0],1],device=x_t.device)
#         mu, s = model(x_t, noise_level)
#         sample_mean = mu * lietorch.SO3(x_t)
#         mu = fn_sample(sample_mean.vec(), s)
    
#         scaled_rotation = lietorch.SO3.exp(lietorch.SO3(x_t).log()/t_i)
#         scaled_mu = lietorch.SO3.exp(mu.log()/t_i).inv()
#         dif_rotation = scaled_rotation * scaled_mu

#         t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0
#         x_next = lietorch.SO3(x_t) * lietorch.SO3.exp((t_iminus-t_i) * dif_rotation.log())

#         if t_iminus != 0:
#             scaled_rotation_2nd = lietorch.SO3.exp(x_next.log()/t_i)
#             mu, s = model(x_next.vec(), noise_level)
#             sample_mu = mu * x_next
#             mu = fn_sample(sample_mu.vec(), s)
#             scaled_mu_2nd = lietorch.SO3.exp(mu.log()/t_i).inv()
#             dif_rotation_2nd = scaled_rotation_2nd * scaled_mu_2nd
#             x_next = lietorch.SO3(x_t) * lietorch.SO3.exp((t_iminus-t_i) * 
#                                             (lietorch.SO3.exp(dif_rotation_2nd.log()/2) * lietorch.SO3.exp(dif_rotation.log()/2)).log()
#                                             )
#         x_t = x_next.vec()

#     return x_t



path = Path("/home/giese/Documents/gecco/gecco-torch/src/gecco_torch/utils/out/")
path.mkdir(exist_ok=True, parents=False)

gm = GaussianModel(3)
gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
rotations_plane = gm.get_rotation.detach() 

fig,ax = plt.subplots()
visualize_so3_probabilities(jax.vmap(lambda x: jaxlie.SO3(x).as_matrix())(rotations_plane.cpu().numpy()),0.001)
plt.savefig(path / "gt_prob_dist.png")

fig,ax = plt.subplots()
visualize_so3_density(jax.vmap(lambda x: jaxlie.SO3(x).as_matrix())(rotations_plane.cpu().numpy()),100)
plt.savefig(path / "gt_density_dist.png")

model = SimpleMLP()

state_dict_path = Path("/home/giese/Documents/gecco/so3models/so3ddpm","gaus_rot_2024-05-27_11:09:40._vexp_all_same_noise","model","gaus_rot_vexp_model-390000.pckl")
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)
model = model.cuda()

gt_rotations = rotations_plane[:,[1,2,3,0]] # in xyzw

schedule = LogUniformSchedule(165)
noise_schedule = schedule.return_schedule(256)
# noise_schedule = torch.cat([torch.zeros(1).cuda(),noise_schedule])
print(noise_schedule)

# x_t = sample_heun(model,gt_rotations,noise_schedule)
x_t = trivial_sample(model,gt_rotations,noise_schedule)
# x_t = original_sample(model,gt_rotations,noise_schedule)

fig,ax = plt.subplots()
visualize_so3_probabilities(jax.vmap(lambda x: jaxlie.SO3(x).as_matrix())(x_t[:,[3,0,1,2]].cpu().numpy()),0.001)
plt.savefig(path / "sampled_prob_dist.png")

fig,ax = plt.subplots()
visualize_so3_density(jax.vmap(lambda x: jaxlie.SO3(x).as_matrix())(x_t[:,[3,0,1,2]].cpu().numpy()),100)
plt.savefig(path / "sampled_density_dist.png")