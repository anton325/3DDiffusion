from gecco_torch.scene.gaussian_model import GaussianModel
import numpy as np
import torch
from scipy.stats import multivariate_normal
from gecco_torch.utils.riemannian_helper_functions import L_to_cov_x6, find_cholesky_L, upper_triangle_to_cov_3x3
from gecco_torch.utils.isotropic_gaussian_no_vmap import IsotropicGaussianSO3
import lietorch


gm = GaussianModel(3)
gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")

rotation = gm.get_rotation.type(torch.float64)
scaling = gm.get_scaling.type(torch.float64)

cov = gm.get_covariance(t = torch.float32)
cholesky = find_cholesky_L(cov)

scaling_distribution = multivariate_normal(mean=np.zeros(3), cov=np.diag(165 * np.ones(3)))
# Calculate the probability density of each observation
prob_densities = scaling_distribution.pdf(scaling.detach().cpu().numpy().reshape(-1,3))
# Calculate the joint likelihood
joint_likelihood = np.sum(prob_densities)
print(f"prob scalings under N(0,165): {joint_likelihood}")

unit_rotations = torch.zeros(cov.shape[0], 4,device=cov.device)
unit_rotations[:,0] = 1
rotation_distribution = IsotropicGaussianSO3(unit_rotations, 165 * torch.ones(cov.shape[0],device=cov.device))
print(f"prob rotations under unit rotations: {rotation_distribution.prob(rotation[:,[1,2,3,0]].type(torch.float32)).sum()}")

# do eigenvalue decomposition
eigenvalues, eigenvectors = torch.linalg.eigh(upper_triangle_to_cov_3x3(L_to_cov_x6(cholesky)))
scaling_decomposition = torch.sqrt(eigenvalues)
rotation_decomposition = lietorch.SO3(eigenvectors, from_rotation_matrix=True)
print(f"prob eigenvec rotations under unit rotations: {rotation_distribution.prob(rotation_decomposition.vec().type(torch.float32)).sum()}")

prob_densities_decomposition = scaling_distribution.pdf(scaling_decomposition.detach().cpu().numpy().reshape(-1,3))
# Calculate the joint likelihood

# print(f"prob eigenvec rotations under unit rotations: {rotation_distribution.prob(lietorch.SO3([],from_uniform_sampled=3999).vec().cuda()).sum()}")
joint_likelihood_decomposed = np.sum(prob_densities_decomposition)
print(f"prob scalings decomposition under N(0,165): {joint_likelihood_decomposed}")
pass