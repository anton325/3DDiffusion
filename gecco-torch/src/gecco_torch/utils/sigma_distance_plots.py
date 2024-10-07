from gecco_torch.additional_metrics.metrics_so3 import geodesic_distance
import torch
from gecco_torch.utils.build_cov_matrix_torch import build_covariance_from_activated_scaling_rotation
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.riemannian_helper_functions import find_cholesky_L
import lietorch
import matplotlib.pyplot as plt


def sigma_distance_plot(self, example):
    t_steps = self.model.t_steps(128, 165, 0.002, 7)
    t_steps_ascending = torch.flip(t_steps, [0])

    gt_rest = example.data[0,:,:7]
    gt_rest_diffusionspace = self.reparam.data_to_diffusion(gt_rest, example.ctx)

    gt_scaling = example.data[0,:,7:10]
    gt_rotation = example.data[0,:,10:14]
    gt_cov = build_covariance_from_activated_scaling_rotation(gt_scaling, gt_rotation)
    gt_L = find_cholesky_L(gt_cov)

    geo_distances = []
    euclid_distances = []

    for t in t_steps_ascending[1:]:
        noisy_rest = gt_rest_diffusionspace + torch.randn_like(gt_rest) * t

        noisy_scaling = gt_scaling +  torch.randn_like(gt_scaling) * t

        rotation_xyzw = gt_rotation[:,[1,2,3,0]] # von wxyz zu xyzw
        noise_for_rotation = t.repeat_interleave(rotation_xyzw.shape[0])

        # Sampling from current temperature
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                                    s_single,
                                                                    force_small_scale=False).sample_one_vmap(),
                                randomness="different")(rotation_xyzw, noise_for_rotation)
        noisy_rotation_xyzw = (lietorch.SO3(rotation_xyzw) * lietorch.SO3.exp(axis_angles.type(rotation_xyzw.dtype))).vec()

        noisy_rotation = noisy_rotation_xyzw[:,[3,0,1,2]] # von xyzw zu wxyz

        
        noisy_cov = build_covariance_from_activated_scaling_rotation(noisy_scaling, noisy_rotation)
        noisy_L = find_cholesky_L(noisy_cov)

        sum_euclid_distance_rest = sum(torch.norm(noisy_rest - gt_rest_diffusionspace, dim = -1))
        sum_geo_dis = sum(geodesic_distance(noisy_L, gt_L))
        euclid_distances.append(sum_euclid_distance_rest.item())
        geo_distances.append(sum_geo_dis.item())
        print(f"For sigma {t} euclid distance: {sum_euclid_distance_rest}, geodesic distance: {sum_geo_dis}")

    fig, ax = plt.subplots()
    ax.plot(t_steps_ascending[1:].cpu().numpy(), euclid_distances, label="Euclidean distance")
    ax.plot(t_steps_ascending[1:].cpu().numpy(), geo_distances, label="Geodesic distance")
    ax.legend()
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Distance")
    ax.set_title(f"Euclidean ({int(euclid_distances[-1])} and Geodesic ({int(geo_distances[-1])}) distance between ground truth and noisy samples")
    plt.savefig("cholesky_sigma_distance_plot.png")