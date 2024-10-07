import torch
from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.utils.build_cov_matrix_torch import build_covariance_from_activated_scaling_rotation, strip_lowerdiag, build_covariance_from_scaling_rotation_xyzw
import json
from typing import NamedTuple
from torchvision.utils import save_image
from gecco_torch.gaussian_renderer import render
from pathlib import Path
import lietorch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.additional_metrics.metrics_so3 import geodesic_distance

"""
convention: 
3x3 = [[L[0]    0     0],
       [L[3]  L[1]    0],
       [L[4]  L[5]  L[2]]]
"""

# def eigenvalue_decomposition(L):

def L_to_scale_rotation(L):
    """
    in shape (4000 * batch_size, 6)
    out shape (4000 * batch_size, 3), (4000 * batch_size, 4)
    """
    cov_3x3 = upper_triangle_to_cov_3x3(L_to_cov_x6(L))
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_3x3)
    scaling = torch.sqrt(torch.clip(eigenvalues, min = 0))
    rotations = lietorch.SO3(eigenvectors, from_rotation_matrix=True).vec()
    return scaling, rotations

def geodesic_step(L1, direction, stepsize):
    lower_triangle = L1[:,3:] + stepsize * direction[:,3:]
    # print(f"stepsize geodesic step: {stepsize}")
    diag = L1[:,:3] * torch.exp(stepsize * direction[:,:3] * group_operation_inverse_L(L1)[:,:3])
    result = torch.cat([diag,lower_triangle],dim=-1)
    return result

def is_in_L_plus(L):
    """
    L: (batch, 6)
    """
    return torch.all(L[:,:3] > 0)

def log_map_at_L(K, L):
    lower_triangle_dif = add_lower_triangle(K-L, torch.zeros_like(K))

    diag_K = to_diag(K)
    diag_L = to_diag(L)
    inverse_diag_L = 1/diag_L
    inverse_diag_L[torch.isinf(inverse_diag_L)] = 0
    mult = diag_K * inverse_diag_L
    log_mult = torch.log(mult)
    log_mult[torch.isinf(log_mult)] = 0
    log_mult_L = diag_L * log_mult

    result = lower_triangle_dif + log_mult_L 
    return result

def exp_map_at_L(X, L):
    lower_triangle_sum = add_lower_triangle(X, L)

    diag_X = to_diag(X)
    diag_L = to_diag(L)
    
    inv_L = 1/diag_L
    inv_L[torch.isinf(inv_L)] = 0
    mult = diag_L * torch.exp(diag_X * inv_L)

    result = lower_triangle_sum + mult
    return result


def to_diag(X):
    """
    input shape: (batch, 6)
    return shape (batch, 6)
    """
    return torch.cat([X[:,:3],torch.zeros_like(X[:,:3])],dim=-1)

def to_lower_triangle(L):
    """
    input shape: (batch, 6)
    return shape (batch, 6)
    """
    return torch.cat([torch.zeros_like(L[:,:3]),L[:,3:],],dim=-1)

def L_to_cov_x6(L):
    """
    L: (batch, 6)
    """
    L_3x3 = lower_triangle_to_3x3(L)
    """
    die beiden methoden sind equal (getestet für shape (batchsize,3,3))
    """
    cov = torch.bmm(L_3x3,L_3x3.transpose(-2, -1))
    # cov = L_3x3 @ L_3x3.transpose(-2, -1)
    cov = strip_lowerdiag(cov)
    return cov

def lower_triangle_to_3x3(lower_traingle):
    """
    input of shape (batch size, 6)
    """
    lower_triangle_3x3 = torch.zeros((lower_traingle.shape[0], 3, 3), device=lower_traingle.device, dtype = lower_traingle.dtype)
    lower_triangle_3x3[:, 0, 0] = lower_traingle[:, 0]
    lower_triangle_3x3[:, 1, 1] = lower_traingle[:, 1]
    lower_triangle_3x3[:, 2, 2] = lower_traingle[:, 2]
    lower_triangle_3x3[:, 1, 0] = lower_traingle[:, 3]  
    lower_triangle_3x3[:, 2, 0] = lower_traingle[:, 4]
    lower_triangle_3x3[:, 2, 1] = lower_traingle[:, 5]
    return lower_triangle_3x3


def upper_triangle_to_cov_3x3(cov):
    """
    input of shape (batch size, 6)
    """
    cov_3x3 = torch.zeros((cov.shape[0], 3, 3), device=cov.device, dtype=cov.dtype)
    cov_3x3[:, 0, 0] = cov[:, 0]
    cov_3x3[:, 0, 1] = cov[:, 1]
    cov_3x3[:, 0, 2] = cov[:, 2]
    cov_3x3[:, 1, 0] = cov[:, 1]  # Symmetric element
    cov_3x3[:, 1, 1] = cov[:, 3]
    cov_3x3[:, 1, 2] = cov[:, 4]
    cov_3x3[:, 2, 0] = cov[:, 2]  # Symmetric element
    cov_3x3[:, 2, 1] = cov[:, 4]  # Symmetric element
    cov_3x3[:, 2, 2] = cov[:, 5]
    return cov_3x3


def group_operation_add(L1,L2):
    """
    L1, L2: (batch, 6)
    """
    lower_triangs = add_lower_triangle(L1,L2)
    prod = multiply_diag_L(L1,L2)
    s = lower_triangs + prod
    return s

def add_lower_triangle(L1,L2):
    """
    L1, L2: (batch, 6)
    """
    L = torch.zeros_like(L1)
    L[:,3] = L1[:,3] + L2[:,3]
    L[:,4] = L1[:,4] + L2[:,4]
    L[:,5] = L1[:,5] + L2[:,5]
    return L

def multiply_diag_L(L1,L2):
    """
    L1, L2: (batch, 6)
    """
    L = torch.zeros_like(L1)
    L[:,0] = L1[:,0] * L2[:,0]
    L[:,1] = L1[:,1] * L2[:,1]
    L[:,2] = L1[:,2] * L2[:,2]
    return L

def group_operation_inverse_L(L):
    """
    beobachtung: im log space ist das das gleiche wie einmal mit -1 durchmultiplizieren
    """
    diag_L = to_diag(L)
    lower_triangle_L = to_lower_triangle(L)
    inverse_diag_L = 1/diag_L
    inverse_diag_L[torch.isinf(inverse_diag_L)] = 0
    result = inverse_diag_L - lower_triangle_L

    return result


def get_noisy_log_L(noise_level):
    unit_rotation = torch.tensor([0.,0.,0.,1.]).reshape(1,4).cuda().repeat(noise_level.shape[0],1) # xyzw
    axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,s_single).sample_one_vmap(),randomness="different")(unit_rotation, noise_level)
    noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
    noisy_scale = torch.randn(noise_level.shape[0],3).cuda() * noise_level
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_L = find_cholesky_L(noisy_cov)
    return noisy_L
    
def find_cholesky_L(cov_matrices):
    """
    rein als cov matrix.
    Dann führen wir die Cholesky Zerlegung durch. Allerdings logaritmieren wir die Einträge auf der Diagonalen.
    Dadurch können wir in der letzten layer vom Netzwerk diese Einträge exponentiieren, damit die Diagonalelemente >= 0
    Output: |L[0]    0      0 |
            |L[1]   L[2]    0 |
            |L[3]   L[4]  L[5]|
    """
    cov_matrices_3x3 = upper_triangle_to_cov_3x3(cov_matrices)

    cov_matrices_3x3 = cov_matrices_3x3.type(torch.float64)
    eigenvalues, _ = torch.linalg.eigh(cov_matrices_3x3)
    non_pos_def = eigenvalues.min(dim=-1).values <= 0  # Check the smallest eigenvalue

    # Correct non-positive-definite matrices
    """
    Manche haben nicht alle Eigenvalues >= 0, das ist nicht erlaubt. Wir nehmen an, dass es numerische Ungenauigkeiten sind, und ersetzen diese einfach durch 0
    und berechnen die Covariance-Matrix neu
    """
    if non_pos_def.sum() > 0:
        corrected_matrices = cov_matrices_3x3.clone()
        for i in torch.where(non_pos_def)[0]:
            eigvals, eigvecs = torch.linalg.eigh(corrected_matrices[i])
            eigvals[eigvals < 0] = 1e-7
            corrected_matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.transpose(-2, -1)
            corrected_matrices[i] = corrected_matrix
    else:
        corrected_matrices = cov_matrices_3x3

    L = torch.linalg.cholesky(corrected_matrices)

    Ls = torch.zeros((cov_matrices.shape[0],6), dtype=cov_matrices.dtype).to(cov_matrices.device)
    # Prepare output tensor
    Ls[..., 0] = L[..., 0, 0]
    Ls[..., 1] = L[..., 1, 1]
    Ls[..., 2] = L[..., 2, 2]
    Ls[..., 3] = L[..., 1, 0]
    Ls[..., 4] = L[..., 2, 0]
    Ls[..., 5] = L[..., 2, 1]

    return Ls

def test_add_lower_triang(L1,L2):
    s = add_lower_triangle(L1,L2)
    assert torch.allclose(s,torch.tensor([[ 0,  0,  0,  L1[:,3] + L2[:,3],  L1[:,4] + L2[:,4], L1[:,5] + L2[:,5]]], device='cuda:0')), "must be close"

def test_multiply_diag_L(L1,L2):
    s = multiply_diag_L(L1,L2)
    assert torch.allclose(s,torch.tensor([[L1[:,0] * L2[:,0],L1[:,1] * L2[:,1],L1[:,2] * L2[:,2], 0, 0, 0]], device='cuda:0')), "must be close"

def test_group_operation_inverse_L(L1):
    s = group_operation_inverse_L(L1)
    assert torch.allclose(s,torch.tensor([[1/L1[:,0], 1/L1[:,1], 1/L1[:,2],  - L1[:,3],- L1[:,4], - L1[:,5]]], device='cuda:0')), "must be close"

def test_group_operation_add(L1,L2):
    s = group_operation_add(L1,L2)
    assert torch.allclose(s,torch.tensor([[L1[:,0] * L2[:,0],
                                           L1[:,1] * L2[:,1],
                                           L1[:,2] * L2[:,2],
                                           L1[:,3] + L2[:,3],
                                           L1[:,4] + L2[:,4], 
                                           L1[:,5] + L2[:,5]]], device='cuda:0')), "must be close"
    
def test_log_map_for_L(K, L):
    s = log_map_at_L(K, L)
    assert torch.allclose(s, torch.tensor([[L[:,0] * torch.log(1/L[:,0] * K[:,0]),
                                            L[:,1] * torch.log(1/L[:,1] * K[:,1]),
                                            L[:,2] * torch.log(1/L[:,2] * K[:,2]),
                                            K[:,3] - L[:,3], K[:,4] - L[:,4], K[:,5] - L[:,5]]], device='cuda:0')), "must be close"
    
def test_exp_map_for_L(X, L):
    s = exp_map_at_L(X, L)
    assert torch.allclose(s, torch.tensor([[L[:,0] * torch.exp(X[:,0] * (1/L[:,0])),
                                            L[:,1] * torch.exp(X[:,1] * (1/L[:,1])),
                                            L[:,2] * torch.exp(X[:,2] * (1/L[:,2])),
                                            X[:,3] + L[:,3], X[:,4] + L[:,4], X[:,5] + L[:,5]]], device='cuda:0')), "must be close"
def test_log_exp_map(K,L):
    s = log_map_at_L(K,L)
    s = exp_map_at_L(s,L)
    assert torch.allclose(s,K), "must be close"

def test_exp_log_map():
    X = torch.randn(4,6).cuda()
    L = torch.randn(4,6).cuda()
    L[:,:3] = torch.exp(L[:,:3])
    K = exp_map_at_L(X,L)
    X_again = log_map_at_L(K,L)
    assert torch.allclose(X,X_again), "must be close"

def renders(gm,covs):
    with open("/home/giese/Documents/gaussian-splatting/circle_cams.json","r") as f:
        circle_cams = json.load(f)

    class Camera(NamedTuple):
        world_view_transform: torch.Tensor
        projection_matrix: torch.Tensor
        tanfovx: float
        tanfovy: float
        imsize: int

    img_path = Path("out_riemannian")
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
        bg = torch.tensor([1.0,1.0,1.0], device=torch.device('cuda'))
        kwargs = {
            'use_cov3D' : True,
            'covs' : covs 
        }
        with torch.no_grad():
            render_dict = render(gm,bg,camera = camera, **kwargs)
            img = render_dict['render']
            save_image(img, img_path / f'_render_{i}.png')

def get_noisy_scale(gt_scale,noise_level):
    noise_level = torch.tensor([noise_level]).repeat(gt_scale.shape[0],1).cuda()
    noisy_scale = torch.randn(noise_level.shape[0],3).cuda() * noise_level
    noisy_scale = gt_scale + noisy_scale
    return noisy_scale

def get_noisy_rotation(gt_rotations, noise_level):
    noise_level = torch.tensor([noise_level]).repeat(gt_rotations.shape[0],1).cuda()
    rotations = gt_rotations[:,[1,2,3,0]]
    axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,s_single).sample_one_vmap(),randomness="different")(rotations, noise_level)
    noisy_rotation = (lietorch.SO3(rotations) * lietorch.SO3.exp(axis_angles)).vec()
    return noisy_rotation[:,[3,0,1,2]]

if __name__ == "__main__":
    # gm = GaussianModel(3)
    # gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
    # rotations_plane = gm.get_rotation.detach() 
    # scale_plane = gm.get_scaling.detach()
    # gt_cov_matrices = build_covariance_from_scaling_rotation(scale_plane, rotations_plane)
    # # renders(gm,gt_cov_matrices)
    # gt_L = find_cholesky_L(gt_cov_matrices)
    # test_L = gt_L[0].unsqueeze(0)
    # test_K = gt_L[1].unsqueeze(0)
    # test_add_lower_triang(test_L,test_L)
    # test_multiply_diag_L(test_L,test_L)
    # test_group_operation_inverse_L(test_L)
    # test_group_operation_add(test_L,test_L)
    # test_log_map_for_L(test_K, test_L)
    # test_exp_map_for_L(test_K, test_L)
    # test_log_exp_map(test_K,test_L)
    # # test_exp_log_map()
    # covs = L_to_cov_x6(gt_L)
    # renders(gm,covs)

    # start = test_L
    # end = test_K
    # print(f"Distance between start and end {geodesic_distance(start,end)}")
    # direction = group_operation_add(group_operation_inverse_L(start),end)
    # walk_step = geodesic_step(start,direction,-0.001)
    # print(f"geodesic step: Distance between walked step and end {geodesic_distance(walk_step,end)}")


    # direction_2 = exp_map_at_L(log_map_at_L(group_operation_add(group_operation_inverse_L(start),end),start)/100,start)

    # walk_step = geodesic_step(start,direction_2,-0.9)
    # print(f"geodesic step 2: Distance between walked step and end {geodesic_distance(walk_step,end)}")

    # direction_3 = log_map_at_L(group_operation_add(group_operation_inverse_L(start),end),start)
    # walk_add_step = group_operation_add(start,exp_map_at_L(-0.9*direction_2,start))
    # print(f"group operator step Distance between walked step and end {geodesic_distance(walk_add_step,end)}")

    # t = 165
    # t2 = 150

    # direction = log_map_at_L(group_operation_add(group_operation_inverse_L(start),end),start)/t2
    # walk_step = geodesic_step(start, direction, t2-t)
    # print(f"geodesic step: Distance between walked step and end {geodesic_distance(walk_step,end)}")

    # direction = log_map_at_L(group_operation_add(group_operation_inverse_L(start),end),start)/t2
    # walk_step = group_operation_add(start,exp_map_at_L((t2-t)*direction,start))
    # print(f"group operation add step: Distance between walked step and end {geodesic_distance(walk_step,end)}")

    # """
    # 2nd order
    # """
    # t = 165
    # t2 = 150
    # t3 = 140

    # direction = log_map_at_L(group_operation_add(group_operation_inverse_L(start),end),start)/t
    # walk_step = geodesic_step(start, direction, t2-t)
    # print(f"geodesic step: Distance between walked step and end {geodesic_distance(walk_step,end)}")
    # mid_point = walk_step

    # direction_2nd = log_map_at_L(group_operation_add(group_operation_inverse_L(mid_point),end),mid_point)/t2
    # walk_step = geodesic_step(mid_point, direction_2nd, t3-t2)
    # print(f"mit 2nd geodesic step: Distance between walked step and end {geodesic_distance(walk_step,end)}")



    # direction_combination = direction + direction_2nd
    # # direction_combination = group_operation_add(direction,direction_2nd)

    # walk_step = geodesic_step(start, direction_combination, (t2-t))
    # print(f"geodesic step 2nd order step: Distance between 2nd walked step and end {geodesic_distance(walk_step,end)}")


    # noise = 0.0015
    # noisy_scales = get_noisy_scale(scale_plane,noise)
    # noisy_rotations = get_noisy_rotation(rotations_plane,noise)
    # noisy_cov_matrices = build_covariance_from_scaling_rotation(noisy_scales, noisy_rotations)
    # noisy_L = find_cholesky_L(noisy_cov_matrices)
    # print(f"Distance between start and noisy end {geodesic_distance(gt_L,noisy_L).mean()}")
    # renders(gm,noisy_cov_matrices)

    # a = 3


    # noisy_cov = torch.tensor([[ 1030., -1015., -2428.,  1635.,  3188.,  7180.]], device='cuda:0')
    # find_chol = find_cholesky_L(noisy_cov)
    # print(find_chol)
    noisy_scaling2= torch.tensor([[ 323.9162,  0.7178,  -30.3929]], device='cuda:0')
    noisy_rotation2 = torch.tensor([[ 0.466, 0.236, 0.8431, -0.128]], device='cuda:0')
    noisy_cov2 = build_covariance_from_activated_scaling_rotation(noisy_scaling2, noisy_rotation2)
    print(noisy_cov2)
