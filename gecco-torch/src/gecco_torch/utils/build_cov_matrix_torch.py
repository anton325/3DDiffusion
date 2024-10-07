import torch
from gecco_torch.scene.gaussian_model import GaussianModel


def build_covariance_from_scaling_rotation_xyzw(scaling, rotation):
    """
    diese funktion wurde original für rotations in wxyz konvention geschrieben. Ich möchte sie jetzt aber für xyzw rotations nutzen.
    Deswegen wird als erstes die convention angepasst
    """
    rotation = rotation[:, [3, 0, 1, 2]]
    return build_covariance_from_activated_scaling_rotation(scaling, rotation)

def build_covariance_from_activated_scaling_rotation(scaling, rotation):
    """
    diese funktion wurde original für rotations in wxyz konvention geschrieben
    scalings müssen schon aktiviert reinkommen
    """
    # activated_scaling = torch.exp(scaling)
    activated_scaling = scaling
    L = build_scaling_rotation(activated_scaling, rotation)
    # print(L)
    # torch actual_covariance = L @ L.transpose(1, 2)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_lowerdiag(actual_covariance)
    # noisy_scaling2= torch.tensor([[ 323.9162,  0.7178,  -30.3929]], device='cuda:0')
    # noisy_rotation2 = torch.tensor([[ 0.466, 0.236, 0.8431, -0.128]], device='cuda:0')
    # d = build_scaling_rotation(noisy_scaling2, noisy_rotation2)
    # print(d)
    # print(d@d.transpose(1,2))
    return symm

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), device = L.device, dtype=L.dtype)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty

def reverse_strip_lowerdiag(batched_6):
    L = torch.zeros((batched_6.shape[0], 3, 3), device = batched_6.device, dtype=batched_6.dtype)
    L[:, 0, 0] = batched_6[:, 0]
    L[:, 0, 1] = batched_6[:, 1]
    L[:, 0, 2] = batched_6[:, 2]
    L[:, 1, 1] = batched_6[:, 3]
    L[:, 1, 2] = batched_6[:, 4]
    L[:, 2, 2] = batched_6[:, 5]
    L[:, 1, 0] = L[:, 0, 1]
    L[:, 2, 0] = L[:, 0, 2]
    L[:, 2, 1] = L[:, 1, 2]
    return L

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.shape[0], 3, 3), device = r.device)

    r = q[:, 0] # real part
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
    
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation_batched(scaling, rotation):
    """
    Diese scaling + rotation -> covariance Funktion ist geklaut von gaussian splatting. Somit ist sie für wxyz quaternionen
    geschrieben worden. Wir nutzen aber xyzw quaternionen. Deswegen wird hier die convention angepasst.
    """
    rotation = rotation[:,:,[3,0,1,2]] # von xyzw zu wxyz
    L = build_scaling_rotation_batched(scaling, rotation)
    # torch actual_covariance = L @ L.transpose(1, 2)
    actual_covariance = L @ L.transpose(2, 3)
    symm = strip_lowerdiag_batched(actual_covariance)
    return symm

def strip_lowerdiag_batched(L):
    uncertainty = torch.zeros((L.shape[0], L.shape[1], 6), device = L.device)

    uncertainty[:,:, 0] = L[:,:, 0, 0]
    uncertainty[:,:, 1] = L[:,:, 0, 1]
    uncertainty[:,:, 2] = L[:,:, 0, 2]
    uncertainty[:,:, 3] = L[:,:, 1, 1]
    uncertainty[:,:, 4] = L[:,:, 1, 2]
    uncertainty[:,:, 5] = L[:,:, 2, 2]

    return uncertainty

def build_rotation_batched(r):
    norm = torch.sqrt(r[:,:,0]*r[:,:,0] + r[:,:,1]*r[:,:,1] + r[:,:,2]*r[:,:,2] + r[:,:,3]*r[:,:,3])

    q = r / norm[:,:, None]

    R = torch.zeros((q.shape[0], q.shape[1], 3, 3), device = r.device)

    r = q[:,:, 0] # real part
    x = q[:,:, 1]
    y = q[:,:, 2]
    z = q[:,:, 3]

    R[:,:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:,:, 0, 1] = 2 * (x*y - r*z)
    R[:,:, 0, 2] = 2 * (x*z + r*y)
    R[:,:, 1, 0] = 2 * (x*y + r*z)
    R[:,:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:,:, 1, 2] = 2 * (y*z - r*x)
    R[:,:, 2, 0] = 2 * (x*z - r*y)
    R[:,:, 2, 1] = 2 * (y*z + r*x)
    R[:,:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
    
def build_scaling_rotation_batched(s, r):
    L = torch.zeros((s.shape[0],s.shape[1], 3, 3), device=s.device)
    R = build_rotation_batched(r)

    L[:,:,0,0] = s[:,:,0]
    L[:,:,1,1] = s[:,:,1]
    L[:,:,2,2] = s[:,:,2]

    L = R @ L
    return L


if __name__ == "__main__":
    gc = GaussianModel(3)
    gc.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
    scalings = gc.get_scaling
    rotations = gc.get_rotation
    cov_gt = build_covariance_from_scaling_rotation_xyzw(scalings, rotations)

    scalings = scalings.unsqueeze(0)
    rotations = rotations.unsqueeze(0)
    cov_batched = build_covariance_from_scaling_rotation_batched(scalings, rotations).squeeze(0)
    print(torch.allclose(cov_gt, cov_batched))

    rotations = torch.tensor([[0.8694,0.4830,0.0386,0.0966]])
    scales = torch.tensor([[-5,-5,-5]])

    cov = build_covariance_from_activated_scaling_rotation(scales, rotations)

    from gecco_torch.utils.riemannian_helper_functions import L_to_cov_x6, find_cholesky_L