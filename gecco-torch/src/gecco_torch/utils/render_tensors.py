import numpy as np
import torch
# import jaxlie
from typing import List
# import jax.numpy as jnp
import matplotlib.pyplot as plt
import lietorch

from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.structs import GaussianContext3d, Camera
from gecco_torch.gaussian_renderer import render
from gecco_torch.structs import Mode

from gecco_torch.utils.lie_utils import batched_lietorch_tangential_to_quaternion
# from gecco_torch.utils.rotation_utils import rotation_matrix_to_quaternion_pytorch3d
from gecco_torch.utils.sh_utils import RGB2SH
from gecco_torch.utils.riemannian_helper_functions import L_to_cov_x6, find_cholesky_L
from gecco_torch.utils.build_cov_matrix_torch import strip_lowerdiag, build_covariance_from_activated_scaling_rotation
from gecco_torch.utils.isotropic_plotting import visualize_so3_probabilities


def get_render_fn(mode: List[Mode]):
    if Mode.normal in mode:
        render_fn = render_fn_options['normal']
    elif Mode.isotropic_rgb in mode:
        render_fn = render_fn_options['isotropic_rgb']
    elif Mode.lie_rotations_wrong in mode:
        render_fn = render_fn_options['lie_wrong']
    elif Mode.only_xyz in mode:
        render_fn = render_fn_options['fake']
    elif Mode.rotation_matrix_mode in mode:
        render_fn = render_fn_options['rotation_matrix']
    elif Mode.log_L in mode:
        render_fn = render_fn_options['log_L']
    elif Mode.only_xyz_cov in mode:
        render_fn = render_fn_options['only_xyz_cov']
    elif Mode.fill_xyz in mode:
        render_fn = render_fn_options['fake']
    elif Mode.xyz_sh in mode:
        render_fn = render_fn_options['fake']
    elif Mode.xyz_scaling in mode:
        render_fn = render_fn_options['fake']
    elif Mode.xyz_rotation in mode:
        render_fn = render_fn_options['fake']
    elif Mode.xyz_opacity in mode:
        render_fn = render_fn_options['fake']
    elif Mode.xyz_scaling_rotation in mode:
        render_fn = render_fn_options['fake']
    elif Mode.normal_gt_rotations in mode:
        render_fn = render_fn_options['render_gt_rotations']

    elif Mode.gt_rotations_canonical in mode:
        render_fn = render_fn_options['render_gt_rotations_canonical']
    elif Mode.cov_matrix_3x3 in mode:
        render_fn = render_fn_options['cov_matrix_3x3']
    elif Mode.procrustes in mode:
        render_fn = render_fn_options['rot_matrix_3x3']
    elif Mode.so3_diffusion in mode:
        render_fn = render_fn_options['render_so3']
    elif Mode.so3_x0 in mode:
        render_fn = render_fn_options['render_so3']
    elif Mode.no_rotation in mode:
        render_fn = render_fn_options['fake']
    elif Mode.cholesky in mode:
        render_fn = render_fn_options['render_cholesky']
    elif Mode.activated_scales in mode:
        render_fn = render_fn_options['render_activated_scales']
    elif Mode.normal_opac in mode:
        render_fn = render_fn_options['render_normal_opac']


    else:
        raise NotImplementedError(f"render function for mode {mode} not implemented")
    return render_fn

def ctx_to_cuda(ctx):
    ctx = GaussianContext3d(
        image=ctx.image.to(torch.device('cuda')),
        K=ctx.K.to(torch.device('cuda')),
        c2w=ctx.c2w.to(torch.device('cuda')),
        w2c=ctx.w2c.to(torch.device('cuda')),
        camera=Camera(
            world_view_transform = ctx.camera.world_view_transform.to(torch.device('cuda')),
            projection_matrix = ctx.camera.projection_matrix.to(torch.device('cuda')),
            tanfovx = ctx.camera.tanfovx,
            tanfovy = ctx.camera.tanfovy,
            imsize = ctx.camera.imsize,
        ),
        splatting_cameras=None, # TODO: splatting_cameras ordentlich übertragen, die funktion wird beim rendern beim Trainieren nicht gebraucht, deswegen reicht das None
        mask_points=ctx.mask_points,
        insinfo=ctx.insinfo
    )
    return ctx

# def plot_rotations_wxyz(rotations, i):
#     """
#     rotations müssen in wxyz format sein
#     """
#     torch.save(rotations, f"images/rotations_{i}.pt")
#     visualize_so3_probabilities(jnp.array([jaxlie.SO3(x.detach().cpu().numpy()).as_matrix() for x in rotations]), 0.001)
#     plt.savefig(f"images/rotations_{i}.png")

def RGB2SH(rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

def make_camera(context, i, splatting_cam_choice = None):
    """
    len(context.splatting_cameras) = Anzahl der zusätzlichen Splatting Cameras
    len(context.splatting_cameras[0]) = 2, weil jetzt die Tuple bestehend aus Camera und Bild Tensor kommen
    len(context.splatting_cameras[0][0]) = 5, weil die Camera 5 Parameter hat
    len(context.splatting_cameras[0][0][0]) = batchsize
    context.splatting_cameras[0][0][0][0] = (4,4) jetzt kommen die einzelnen parameter

    context.splatting_cameras[0][1].shape = (batchsize, 3, 400, 400)
    """
    if splatting_cam_choice is not None:
        cam = Camera(
                world_view_transform = context.splatting_cameras[splatting_cam_choice][0][0][i],
                projection_matrix = context.splatting_cameras[splatting_cam_choice][0][1][i],
                tanfovx = context.splatting_cameras[splatting_cam_choice][0][2][i],
                tanfovy = context.splatting_cameras[splatting_cam_choice][0][3][i],
                imsize = context.splatting_cameras[splatting_cam_choice][0][4][i],
            )
    else:
        cam = Camera(
            world_view_transform = context.camera.world_view_transform[i],
            projection_matrix = context.camera.projection_matrix[i],
            tanfovx = context.camera.tanfovx[i],
            tanfovy = context.camera.tanfovy[i],
            imsize = context.camera.imsize[i],
        )
    return cam

def correct_data_format(data):
    if type(data) == type(np.array([1.0,2.0])):
        data = torch.from_numpy(data)
    if data.type != torch.float32:
        data = data.type(torch.float32)
    if data.device == torch.device('cpu'):
        data = data.cuda()
    return data

def mult_quats_torch(q1,q2):
    a0, a1, a2, a3 = q1[0],q1[1],q1[2],q1[3]
    b0, b1, b2, b3 = q2[0],q2[1],q2[2],q2[3]
    return torch.tensor([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0
    ])

def mult_quats_torch_vectorized(q1,q2):
    a0, a1, a2, a3 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    b0, b1, b2, b3 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    # Compute the product of quaternions element-wise
    w = a0*b0 - a1*b1 - a2*b2 - a3*b3
    x = a0*b1 + a1*b0 + a2*b3 - a3*b2
    y = a0*b2 - a1*b3 + a2*b0 + a3*b1
    z = a0*b3 + a1*b2 - a2*b1 + a3*b0
    
    # Stack the components together to form the output quaternions
    result = torch.stack((w, x, y, z), dim=1)
    return result


def quaternion_to_rotation_matrix(q):
    r, x, y, z = q[0],q[1],q[2],q[3]
    rotation_matrix = torch.tensor([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)],
        [2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)],
        [2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)]
    ])
    return rotation_matrix

def rotation_matrix_to_quaternion(R):
    T = torch.trace(R)
    if T > 0:
        w = 0.5 * torch.sqrt(1 + T)
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        x = 0.5 * torch.sqrt(1 + 2 * R[0, 0] - T)
        w = (R[2, 1] - R[1, 2]) / (4 * x)
        y = (R[0, 1] + R[1, 0]) / (4 * x)
        z = (R[0, 2] + R[2, 0]) / (4 * x)
    elif R[1, 1] > R[2, 2]:
        y = 0.5 * torch.sqrt(1 + 2 * R[1, 1] - T)
        w = (R[0, 2] - R[2, 0]) / (4 * y)
        x = (R[0, 1] + R[1, 0]) / (4 * y)
        z = (R[1, 2] + R[2, 1]) / (4 * y)
    else:
        z = 0.5 * torch.sqrt(1 + 2 * R[2, 2] - T)
        w = (R[1, 0] - R[0, 1]) / (4 * z)
        x = (R[0, 2] + R[2, 0]) / (4 * z)
        y = (R[1, 2] + R[2, 1]) / (4 * z)
    return torch.tensor([w, x, y, z])

def render_no_modifications(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], splatting_cam_choice = None, return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 14 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9, 10, 11, 12: rotation (als quaternion)
    13: opacity

    data shape (batch size, 4000, 14)

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    rotations = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rotation = data[i,:,9:13]
        if Mode.in_camera_space in mode:
            # die gaussian pc muss noch aus dem jeweiligen camera space in den world space zum splatten transformiert werden
            xyz = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], xyz) + context.c2w[i][:3, -1] # adding translation part
            c2w_rotation = context.c2w[i][:3,:3]
            c2w_quat = rotation_matrix_to_quaternion(c2w_rotation)
            c2w_quat = c2w_quat.unsqueeze(0).repeat(xyz.shape[0], 1).to(xyz.device).float()
            # print(f"{i} c2w_quat {c2w_quat}")
            # print(c2w_rotation)
            # print(f"rotation 0 {data[i,0,9:13]}")
            # for j in range(data.shape[1]):
            #     # R_quaternion = quaternion_to_rotation_matrix(data[i,j,9:13]).cuda()
            #     # new_rotation = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], R_quaternion)
            #     # data[i,j,9:13] = rotation_matrix_to_quaternion(new_rotation).cuda()
            #     rotation[j,:] = mult_quats_torch(c2w_quat,rotation[j,:]).cuda().float()
            rotation = mult_quats_torch_vectorized(c2w_quat,rotation).cuda().float()
            # doing the same for the rotatinons
        features_dc = data[i,:,3:6] 
        
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            features_dc = RGB2SH(features_dc)

        if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
            rotations.append(rotation.unsqueeze(0)) # append in wxyz
            
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = data[i,:,13],scaling = data[i,:,6:9],rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context, i, splatting_cam_choice)
        # print("load")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        try:
            render_dict = render(pc,bg,camera = cam)
        except:
            print("Fehler in render no modifications, saved gaussian pc to output/export_render_tensor")
            # pc.save_ply(f"/home/giese/Documents/gaussian-splatting/output/export_render_tensor2/point_cloud/iteration_30000/point_cloud_{step}_{i}.ply")
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
        out['rotations'] = torch.concat(rotations)
    return out

def render_normal_opac(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], splatting_cam_choice = None, return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 13 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9, 10, 11, 12: rotation (als quaternion)

    data shape (batch size, 4000, 14)

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    rotations = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rotation = data[i,:,9:13]
        if Mode.in_camera_space in mode:
            # die gaussian pc muss noch aus dem jeweiligen camera space in den world space zum splatten transformiert werden
            xyz = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], xyz) + context.c2w[i][:3, -1] # adding translation part
            c2w_rotation = context.c2w[i][:3,:3]
            c2w_quat = rotation_matrix_to_quaternion(c2w_rotation)
            c2w_quat = c2w_quat.unsqueeze(0).repeat(xyz.shape[0], 1).to(xyz.device).float()
            # print(f"{i} c2w_quat {c2w_quat}")
            # print(c2w_rotation)
            # print(f"rotation 0 {data[i,0,9:13]}")
            # for j in range(data.shape[1]):
            #     # R_quaternion = quaternion_to_rotation_matrix(data[i,j,9:13]).cuda()
            #     # new_rotation = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], R_quaternion)
            #     # data[i,j,9:13] = rotation_matrix_to_quaternion(new_rotation).cuda()
            #     rotation[j,:] = mult_quats_torch(c2w_quat,rotation[j,:]).cuda().float()
            rotation = mult_quats_torch_vectorized(c2w_quat,rotation).cuda().float()
            # doing the same for the rotatinons
        features_dc = data[i,:,3:6] 
        
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            features_dc = RGB2SH(features_dc)

        if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
            rotations.append(rotation.unsqueeze(0)) # append in wxyz
            
        opacity = 10 * torch.ones(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = opacity, scaling = data[i,:,6:9],rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context, i, splatting_cam_choice)
        # print("load")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        try:
            render_dict = render(pc,bg,camera = cam)
        except:
            print("Fehler in render no modifications, saved gaussian pc to output/export_render_tensor")
            # pc.save_ply(f"/home/giese/Documents/gaussian-splatting/output/export_render_tensor2/point_cloud/iteration_30000/point_cloud_{step}_{i}.ply")
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
        out['rotations'] = torch.concat(rotations)
    return out


def render_activated_scales(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], splatting_cam_choice = None, return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 14 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling -> activated
    9, 10, 11, 12: rotation (als quaternion)
    13: opacity

    data shape (batch size, 4000, 14)

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    rotations = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rotation = data[i,:,9:13]
        features_dc = data[i,:,3:6] 
        scalings = data[i,:,6:9]
        scalings_clipped = torch.clip(scalings,min = 1e-15)
        scalings_log = torch.log(scalings_clipped)
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            features_dc = RGB2SH(features_dc)
        
        if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
            rotations.append(rotation.unsqueeze(0))

        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = data[i,:,13], scaling = scalings_log, rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i, splatting_cam_choice)
        # print("load")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        try:
            render_dict = render(pc,bg,camera = cam)
        except:
            print("Fehler in render activated scales, saved gaussian pc to output/export_render_tensor")
            # pc.save_ply(f"/home/giese/Documents/gaussian-splatting/output/export_render_tensor2/point_cloud/iteration_30000/point_cloud_{step}_{i}.ply")
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
        out['rotations'] = torch.concat(rotations)
    return out


def render_cholesky(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], splatting_cam_choice = None, return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 13 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features (rgb)
    6: opacity
    7, 8, 9, 10, 11, 12: L aus cholesky decomposition der covariance matrix, cov = L L.T
    [x,y,z,a,b,c] -> 
    [[x, 0, 0],
     [a, y, 0],
     [b, c, z]]

    data shape (batch size, 4000, 13)

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """

    """
    im ground truth fall müssen wir anders vorgehen: das merkt man daran, dass die shape (batch size, 4000, 14) ist
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    Ls = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]

        if data.shape[2] == 13:
            use_cov3d = True
            Ls_vals = data[i,:,7:13]
            cov = L_to_cov_x6(Ls_vals)
            # wenn im cov ein NA (wie auch immer) reingerutscht ist, ersetze durch eine valide cholesky
            # cov[0,0] = torch.nan
            if torch.isnan(cov).sum() > 0:
                print(f"Render cholesky: detected NAN in cov, number of nans: {torch.isnan(cov).sum()}, replacing with valid cholesky")
                mask = torch.isnan(cov).sum(-1) > 0 # shape (4000), nur scalar values, nämlich die summen entlang der 6
                cov[mask] = torch.tensor([ 2.5000e+01,  4.7684e-07, -4.7684e-07,  2.5000e+01,  0.0000e+00,2.5000e+01], device=cov.device, dtype=cov.dtype)
            scaling = None
            rotation = None
            if Mode.cholesky_distance in mode and kwargs.get('cholesky_distance',False):
                Ls.append(Ls_vals.unsqueeze(0))
        else:
            # raise Exception("render cholesky shape not 13 not implemented")
            # das ist der gt fall
            use_cov3d = False
            scaling = torch.log(data[i,:,7:10]) # wir laden sie im aktivierten Zustand, zum rendern müssen wir sie gelogged ins model einbringen
            rotation = data[i,:,10:14]
            cov = None
            if Mode.cholesky_distance in mode and kwargs.get('cholesky_distance',False):
                with torch.autocast(device_type="cuda", enabled=False):
                    gt_cov = build_covariance_from_activated_scaling_rotation(torch.exp(scaling), rotation)
                gt_L = find_cholesky_L(gt_cov)
                Ls.append(gt_L.unsqueeze(0))
            

        features_dc = data[i,:,3:6] 
        
        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False
            # features_dc = torch.clamp(features_dc,0,1)

        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = data[i,:,6], scaling = scaling, rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        # pc2 = GaussianModel(3)
        # pc2.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        # pc._scaling = pc2._scaling
        # pc._rotation = pc2._rotation
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i, splatting_cam_choice)
        render_kwargs = {
            'use_cov3D' : use_cov3d,
            'covs' : cov,
            'override_color':override_color,
            'rgbs' : rgbs,
        }
        
        render_dict = render(pc,bg,camera = cam,**render_kwargs)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.cholesky_distance in mode and kwargs.get('cholesky_distance',False):
        out['Ls'] = torch.concat(Ls)
    return out
    
def render_so3(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], return_depth: bool = False,**kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 15 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9: opacity
    10, 11, 12, 13: rotation (als quaternion)
    14: scale -> die brauchen wir aber nicht, die wird nur fürs sampling ausgegeben -> also wir brauchen die fürs sampling, aber hier nicht

    Macht eigentlich keinen Sinn, weil so3 nur einen Schritt denoised und nicht zu x0 geht. -> also brauchen wir einen eigenen sampler dafür
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    rotations = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rotation = data[i,:,10:14]
        # rotations sind im xyzw format, aber gaussian splatting will die im wxyz format
        rotation = rotation[:,[3,0,1,2]]
        features_dc = data[i,:,3:6] 
        
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            features_dc = RGB2SH(features_dc)
        
        if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
            rotations.append(rotation.unsqueeze(0)) # append in wxyz
            
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = data[i,:,9], scaling = data[i,:,6:9],rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)
        # print("load")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        try:
            render_dict = render(pc,bg,camera = cam)
        except Exception as e:
            print(f"Fehler in render rotation matrix, saved gaussian pc to output/export_render_tensor, fehler {e}")
            # pc.save_ply(f"/home/giese/Documents/gaussian-splatting/output/export_render_tensor2/point_cloud/iteration_30000/point_cloud_{step}_{i}.ply")
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
        out['rotations'] = torch.concat(rotations)
    return out
    
def render_gt_rotations(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 14 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9, 10, 11, 12: rotation (als quaternion)
    13: opacity

    data shape (batch size, 4000, 14)

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        # load gt rotation
        gt_gaussian = GaussianModel(3)
        gt_gaussian.load_ply(f"/globalwork/giese/gaussians/{context.insinfo.category[i]}/{context.insinfo.instance[i]}/point_cloud/iteration_10000/point_cloud.ply")
        rotation = gt_gaussian._rotation[:xyz.shape[0],:]
        if rotation.shape[0] < xyz.shape[0]:
            rotation = torch.cat([rotation,torch.zeros((xyz.shape[0]-rotation.shape[0],4),device=rotation.device)],dim=0)
        rotation = rotation.detach()
        
        features_dc = data[i,:,3:6] 

        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = data[i,:,9],scaling = data[i,:,6:9],rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)

        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False

        kwargs = {
                'override_color':override_color,
                'rgbs' : rgbs,
            }
        
        # print("load")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        try:
            render_dict = render(pc,bg,camera = cam,**kwargs)
        except:
            print("Fehler in render rotation matrix, saved gaussian pc to output/export_render_tensor")
            # pc.save_ply(f"/home/giese/Documents/gaussian-splatting/output/export_render_tensor2/point_cloud/iteration_30000/point_cloud_{step}_{i}.ply")
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    if return_depth:
        return torch.concat(rendered_images), torch.concat(depth)
    else:
        return torch.concat(rendered_images)
    
def render_gt_rotations_canonical(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 14 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9: opacity
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        # load gt rotation
        gt_gaussian = GaussianModel(3)
        gt_gaussian.load_ply(f"/globalwork/giese/gaussians/{context.insinfo.category[i]}/{context.insinfo.instance[i]}/point_cloud/iteration_10000/point_cloud.ply")
        rotation = gt_gaussian._rotation[:xyz.shape[0],:]
        if rotation.shape[0] < xyz.shape[0]:
            rotation = torch.cat([rotation,torch.zeros((xyz.shape[0]-rotation.shape[0],4),device=rotation.device)],dim=0)
        rotation = rotation.detach()
        
        # die scalings wurden nach größe sortiert, jetzt müssen wir sie noch zurücksortieren
        scalings_gt = gt_gaussian._scaling[:xyz.shape[0],:]
        if scalings_gt.shape[0] < xyz.shape[0]:
            scalings_gt = torch.cat([scalings_gt,-10*torch.ones((xyz.shape[0]-scalings_gt.shape[0],3),device=rotation.device)],dim=0)

        _, sorted_indices = torch.sort(scalings_gt,dim=1,descending=True)

        scaling_sort_back = torch.empty_like(xyz)

        # # generate inverse indices and place elements back where they belong

        # columns_range = torch.arange(data.shape[1])
        # current_data = data[i:,6:9].clone()

        #     # Use advanced indexing to perform the assignment in one operation
        # scaling_sort_back[columns_range, sorted_indices] = current_data[columns_range]

        for j in range(data.shape[1]):
            scaling_sort_back[j][sorted_indices[j]] = data[i,:,6:9][j].clone()

        features_dc = data[i,:,3:6]

        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = data[i,:,9],scaling = scaling_sort_back ,rotation = rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)

        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False

        kwargs = {
                'override_color':override_color,
                'rgbs' : rgbs,
            }
        
        # print("load")
        # pc.load_ply("/home/giese/Documents/gaussian-splatting/temp/pc.ply")
        try:
            render_dict = render(pc,bg,camera = cam,**kwargs)
        except:
            print("Fehler in render rotation matrix, saved gaussian pc to output/export_render_tensor")
            # pc.save_ply(f"/home/giese/Documents/gaussian-splatting/output/export_render_tensor2/point_cloud/iteration_30000/point_cloud_{step}_{i}.ply")
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    if return_depth:
        return torch.concat(rendered_images), torch.concat(depth)
    else:
        return torch.concat(rendered_images)
    
def render_rotation_matrix(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], return_depth: bool = False,**kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 19 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9, 10, 11, 12, 13, 14, 15, 16, 17, : rotation (als rotation matrix)
    18: opacity

    data shape (batch size, 4000, 18)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rotation = data[i,:,9:18]

        # with torch.autocast(device_type="cuda", enabled=False):
            # rotation = rotation_matrix_to_quaternion_torch_batched(rotation.view(rotation.shape[0],3,3))
        # rotation = rotation_matrix_to_quaternion_pytorch3d(rotation.view(rotation.shape[0],3,3))
        rotation_objects = lietorch.SO3(rotation, from_rotation_matrix=True)
        rotation = rotation_objects.vec()[:,[3,0,1,2]] # lietorch in xyxz format, gaussian splatting will wxyz format

        features_dc = data[i,:,3:6] 
        
        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False
        
        if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
            plot_rotations_wxyz(rotation, i)

        # rotation = torch.randn((4000,4)).to(features_dc.device)
        # vis_points_opac = 100*torch.ones((context.mask_points[i],1),device=data.device,dtype=torch.float32) # data[i,:,18]
        # invis_points_opac = -100*torch.ones((xyz.shape[0]-context.mask_points[i],1),device=data.device,dtype=torch.float32) # data[i,:,18]
        # opacity = torch.cat([vis_points_opac,invis_points_opac],dim=0)
        # opacity = 100*torch.ones((context.mask_points[i],1),device=data.device,dtype=torch.float32) # data[i,:,18]
        opacity = data[i,:,18]
        scaling = data[i,:,6:9] # clamp findet im model drin statt
        # scaling = torch.clamp(scaling,-7,7)
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = opacity,scaling = scaling,rotation = rotation)
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)
        try:
            kwargs = {
                'override_color':override_color,
                'rgbs' : rgbs,
            }
            render_dict = render(pc,bg,camera = cam,**kwargs)
        except:
            # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
            print("Fehler in render rotation matrix, saved gaussian pc to output/export_render_tensor")

        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    if return_depth:
        return torch.concat(rendered_images), torch.concat(depth)
    else:
        return torch.concat(rendered_images)
    
def render_log_L(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], splatting_cam_choice = None, return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 13 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6: opacity
    7, 8, 9, 10, 11, 12: cholesky L

    data shape (batch size, 4000, 13)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    Ls = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        use_cov3d = True
        Ls_vals = data[i,:,7:13]
        # un-log
        # Ls_vals[:,0:3] = torch.exp(torch.clip(Ls_vals[:,0:3], max = 7))
        Ls_vals = torch.cat([torch.exp(torch.clip(Ls_vals[:,0:3], max = 7)),Ls_vals[:,3:6]],dim=1)
        Ls.append(Ls_vals.unsqueeze(0))
        cov = L_to_cov_x6(Ls_vals)
        if cov.dtype != xyz.dtype:
            cov = cov.type(xyz.dtype)

        features_dc = data[i,:,3:6] 
        
        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False

        render_kwargs = {
            'use_cov3D' : use_cov3d,
            'covs' : cov,
            'override_color':override_color,
            'rgbs' : rgbs,
        }

        opacity = data[i,:,6]
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = opacity, scaling = None, rotation = None)
        # pc.save_ply(path = "/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply", cov = cov)
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i, splatting_cam_choice)
        render_dict = render(pc, bg, camera = cam, **render_kwargs)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.cholesky_distance in mode and kwargs.get('cholesky_distance',False):
        out['Ls'] = torch.concat(Ls)
    return out
    
def render_cov_matrix_3x3(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 19 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6: opacity
    7, 8, 9, 10, 11, 12, 13, 14, 15: 3x3 cov matrix

    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    rotations = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        cov3D_precomp_3x3 = data[i,:,7:].reshape(-1,3,3)
        cov3D_precomp = strip_lowerdiag(cov3D_precomp_3x3)

        features_dc = data[i,:,3:6] 
        
        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False
            # features_dc = torch.clamp(features_dc,0,1)
        # rotation = torch.randn((4000,4)).to(features_dc.device)
        # opac = data[i,:,12]
        # opac = 100*torch.ones((xyz.shape[0],1),device=data.device,dtype=torch.float32)
        opacity = data[i,:,6]
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = opacity, scaling = None, rotation = None)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)
        kwargs = {
            'use_cov3D' : True,
            'covs' : cov3D_precomp,
            'override_color' : override_color,
            'rgbs' : rgbs,
        }
        render_dict = render(pc, bg, camera = cam,**kwargs)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
        out['rotations'] = torch.concat(rotations)
    return out


def render_rot_matrix_3x3(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], splatting_cam_choice = None, return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 19 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6: opacity
    7,8,9: scaling
    10, 11, 12, 13, 14, 15, 16, 17, 18: 3x3 cov matrix

    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    rotations = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rot_precomp_3x3 = data[i,:,10:].reshape(-1,3,3)
        rot_quat_xyzw = lietorch.SO3(rot_precomp_3x3,from_rotation_matrix=True).vec()
        rot_quat = rot_quat_xyzw[:,[3,0,1,2]] # von xyzw zu wxyz

        if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
            rotations.append(rot_quat.unsqueeze(0)) # append in wxyz
            
        scaling = data[i,:,7:10]
        features_dc = data[i,:,3:6] 
        
        rgbs = None
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            # features_dc = RGB2SH(features_dc)
            rgbs = features_dc
            features_dc = None
            override_color = True
        else:
            override_color = False
            # features_dc = torch.clamp(features_dc,0,1)
        # rotation = torch.randn((4000,4)).to(features_dc.device)
        # opac = data[i,:,12]
        # opac = 100*torch.ones((xyz.shape[0],1),device=data.device,dtype=torch.float32)
        opacity = data[i,:,6]
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = opacity, scaling = scaling, rotation = rot_quat)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i, splatting_cam_choice)
        kwargs_render = {
            'override_color' : override_color,
            'rgbs' : rgbs,
        }
        render_dict = render(pc, bg, camera = cam,**kwargs_render)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    out = {
        'render' : torch.concat(rendered_images)
    }
    if return_depth:
        out['depth'] = torch.concat(depth)
    if Mode.rotational_distance in mode and kwargs.get('plot_rotations',False):
        out['rotations'] = torch.concat(rotations)
    return out
    
def render_cov_matrix_no_color(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode], return_depth: bool = False, **kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 19 dimensionen:
    0,1,2 : xyz
    6, 7, 8, 9, 10, 11: covariance (als lower triangle matrix):
    [x,y,z,a,b,c] -> 
    [[x, 0, 0],
     [a, y, 0],
     [b, c, z]]
    12: opacity

    data shape (batch size, 4000, 13)

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    depth = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        Ls_vals = data[i,:,6:12]
        Ls = torch.zeros((xyz.shape[0],3,3),device=data.device,dtype=torch.float32)
        Ls[:,0,0] = Ls_vals[:,0]
        Ls[:,1,1] = Ls_vals[:,1]
        Ls[:,2,2] = Ls_vals[:,2]
        Ls[:,1,0] = Ls_vals[:,3]
        Ls[:,2,0] = Ls_vals[:,4]
        Ls[:,2,1] = Ls_vals[:,5]

        # diagonal elements
        diagonal = Ls.diagonal(dim1=-2, dim2=-1)
        # exp
        diagonal_exp = torch.exp(diagonal)
        # insert diagonal values
        Ls.diagonal(dim1=-2, dim2=-1).copy_(diagonal_exp)


        cov3D_precomp_3x3 = torch.bmm(Ls, Ls.transpose(1, 2)).type(torch.float32)
        cov3D_precomp = torch.zeros((xyz.shape[0],6),device=data.device,dtype=torch.float32)
        cov3D_precomp[:,0] = cov3D_precomp_3x3[:,0,0]
        cov3D_precomp[:,1] = cov3D_precomp_3x3[:,0,1]
        cov3D_precomp[:,2] = cov3D_precomp_3x3[:,0,2]
        cov3D_precomp[:,3] = cov3D_precomp_3x3[:,1,1]
        cov3D_precomp[:,4] = cov3D_precomp_3x3[:,1,2]
        cov3D_precomp[:,5] = cov3D_precomp_3x3[:,2,2]
        
        # clamp_value = 49 # 7^2
        # eigenvalues, eigenvectors = torch.linalg.eigh(covs)
        # clamped_eigenvalues = torch.clamp(eigenvalues, 0, clamp_value)
        # # Reconstruct the scaling matrix S, taking square root of absolute values of eigenvalues and retaining the sign
        # signs = torch.sign(clamped_eigenvalues)
        # S_clamped = torch.diag_embed(signs * torch.sqrt(torch.abs(clamped_eigenvalues)))

        # # Reconstruct the covariance matrices
        # covs = eigenvectors @ S_clamped @ S_clamped.transpose(-2, -1) @ eigenvectors.transpose(-2, -1) 
        # covs = covs.type(torch.float32)

        if Mode.in_camera_space in mode:
            # die gaussian pc muss noch aus dem jeweiligen camera space in den world space zum splatten transformiert werden
            xyz = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], xyz) + context.c2w[i][:3, -1] # adding translation part
            c2w_rotation = context.c2w[i][:3,:3]
            c2w_quat = rotation_matrix_to_quaternion(c2w_rotation)
            c2w_quat = c2w_quat.unsqueeze(0).repeat(xyz.shape[0], 1).to(xyz.device).float()
            # print(f"{i} c2w_quat {c2w_quat}")
            # print(c2w_rotation)
            # print(f"rotation 0 {data[i,0,9:13]}")
            # for j in range(data.shape[1]):
            #     # R_quaternion = quaternion_to_rotation_matrix(data[i,j,9:13]).cuda()
            #     # new_rotation = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], R_quaternion)
            #     # data[i,j,9:13] = rotation_matrix_to_quaternion(new_rotation).cuda()
            #     rotation[j,:] = mult_quats_torch(c2w_quat,rotation[j,:]).cuda().float()
            rotation = mult_quats_torch_vectorized(c2w_quat,rotation).cuda().float()
            # doing the same for the rotatinons
        features_dc = RGB2SH(0.5*torch.ones((cov3D_precomp.shape[0],1,3),device=cov3D_precomp.device,dtype=torch.float32))
        
        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            features_dc = RGB2SH(features_dc)
        # rotation = torch.randn((4000,4)).to(features_dc.device)
        # opac = data[i,:,12]
        opac = -100*torch.ones((xyz.shape[0],1),device=data.device,dtype=torch.float32)
        pc.create_from_values(xyz = xyz, features_dc = features_dc, opacity = opac,scaling = None,rotation = None)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)
        kwargs = {
            'use_cov3D' : True,
            'covs' : cov3D_precomp,
        }
        render_dict = render(pc,bg,camera = cam,**kwargs)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    if return_depth:
        return torch.concat(rendered_images), torch.concat(depth)
    else:
        return torch.concat(rendered_images)

def render_lie_wrong(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode],return_depth: bool = False,**kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 13 dimensionen:
    0,1,2 : xyz
    3,4,5: spherical harmonics features
    6,7,8: scaling
    9, 10, 11,: rotation (im lie space, im tangential raum)
    12: opacity

    no modification nimmt an, dass die rotation als quaternions gegeben sind

    gecco vanilla betreibt diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    depth = []
    rendered_images = []
    rotations = batched_lietorch_tangential_to_quaternion(data[:,:,9:12]).float()
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        rotation = rotations[i]
        if Mode.in_camera_space in mode:
            # die gaussian pc muss noch aus dem jeweiligen camera space in den world space zum splatten transformiert werden
            xyz = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], xyz) + context.c2w[i][:3, -1] # adding translation part
            c2w_rotation = context.c2w[i][:3,:3]
            c2w_quat = rotation_matrix_to_quaternion(c2w_rotation)
            c2w_quat = c2w_quat.unsqueeze(0).repeat(xyz.shape[0], 1).to(xyz.device).float()
            # print(f"{i} c2w_quat {c2w_quat}")
            # print(c2w_rotation)
            # print(f"rotation 0 {data[i,0,9:13]}")
            # for j in range(data.shape[1]):
            #     # R_quaternion = quaternion_to_rotation_matrix(data[i,j,9:13]).cuda()
            #     # new_rotation = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], R_quaternion)
            #     # data[i,j,9:13] = rotation_matrix_to_quaternion(new_rotation).cuda()
            #     rotation[j,:] = mult_quats_torch(c2w_quat,rotation[j,:]).cuda().float()
            rotation = mult_quats_torch_vectorized(c2w_quat,rotation).cuda().float()
            # doing the same for the rotatinons
        features_dc = data[i,:,3:6]

        if Mode.rgb in mode:
            # diffusion im rgb space, muss zurück zu spherical harmonics für splatting
            features_dc = RGB2SH(features_dc)

        pc.create_from_values(xyz = xyz, features_dc=features_dc, opacity = data[i,:,12],scaling=data[i,:,6:9],rotation=rotation)
        # pc.save_ply("/home/giese/Documents/gaussian-splatting/output/export_render_tensor/point_cloud/iteration_30000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)
        render_dict = render(pc,bg,camera = cam)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
        
        if return_depth:
            depth.append(render_dict['depth'].unsqueeze(0))

    if return_depth:
        return torch.concat(rendered_images), torch.concat(depth)
    else:
        return torch.concat(rendered_images)

def render_isotropic_rgb(data: torch.Tensor, context:GaussianContext3d, mode,**kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 8 dimensionen:
    0,1,2 : xyz
    3,4,5: rgb
    6: scaling (isotropic)
    7: opacity


    auch hier diffusion im camera space. (also muss zum gaussian rendern die pointcloud in den world space transformiert werden)
    """
    # print("render isotropic reg")
    def RGB2SH(rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
    
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)

    rendered_images = []
    for i in range(data.shape[0]):
        pc = GaussianModel(3)
        xyz = data[i,:,0:3]
        # xyz shape 4000.3
        if Mode.in_camera_space in mode:
            # die gaussian pc muss noch aus dem jeweiligen camera space in den world space zum splatten transformiert werden
            xyz = torch.einsum("ab,nb->na", context.c2w[i][:3, :3], xyz) + context.c2w[i][:3, -1] # adding translation part

        # data[i,:,6] hat shape 4000, deswegen reshape auf (4000,1) und dann 3 mal auf der letzten dimension (1) wiederholen
        scalings = torch.repeat_interleave(data[i,:,6].reshape(-1,1), repeats=3, dim=1)


        features_dc = RGB2SH(data[i,:,3:6])

        # standard identity rotation

        rotations = torch.concat([torch.ones((data.shape[1],1),device=data.device),
                                  torch.zeros((data.shape[1],3),device=data.device)
                                  ],
                                  dim=1)
        # print("create from tensor")
        pc.create_from_values(xyz = xyz, features_dc=features_dc, opacity = data[i,:,7],scaling=scalings,rotation=rotations)
        # try:
        #     pc.save_ply("/home/giese/Documents/gaussian-splatting/output/done30000/point_cloud/iteration_30000/point_cloud.ply")
        #     pc.load_ply("/home/giese/Documents/gaussian-splatting/output/done30000/point_cloud/iteration_30000/point_cloud.ply")
        # except:
        #     pc.save_ply("/home/wc101072/tmp_pc/point_cloud.ply")
        #     pc.load_ply("/home/wc101072/tmp_pc/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        cam = make_camera(context,i)
        # print("render")
        render_dict = render(pc,bg,camera = cam)
        rendered_image = render_dict['render']
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann
    return torch.concat(rendered_images)


def render_fake_xyz(data: torch.Tensor, context:GaussianContext3d, mode: List[Mode],return_depth: bool = False,**kwargs) -> torch.Tensor:
    """
    der tensor besteht aus 3 dimensionen:
    0,1,2 : xyz
    Wir wollen nur auf xyz trainineren, also gar kein splatting, wir returnen einfach zeros (batchsize,3, 400,400)

    """
    # print("render isotropic reg")
    
    data = correct_data_format(data)
    if context.image.device == torch.device('cpu'):
        context = ctx_to_cuda(context)


    """
    Macht mit render depth keinen sinn mehr
    
    """
    rendered_images = []
    depth = []
    for i in range(data.shape[0]):
        rendered_image = torch.zeros((3,400,400),device=data.device)
        rendered_images.append(rendered_image.unsqueeze(0)) # unsqueeze damit man die torch concat kann

        rendered_depth = torch.zeros((400,400),device=data.device)
        depth.append(rendered_depth.unsqueeze(0)) # unsqueeze damit man die torch concat kann
    if return_depth:
        return torch.concat(rendered_images), torch.concat(depth)
    else:
        return torch.concat(rendered_images)

render_fn_options = {
    'normal' : render_no_modifications,
    'isotropic_rgb' : render_isotropic_rgb,
    'lie_wrong' : render_lie_wrong,
    'fake' : render_fake_xyz,
    'rotation_matrix' : render_rotation_matrix,
    'log_L' : render_log_L,
    'cov_matrix_3x3' : render_cov_matrix_3x3,
    'rot_matrix_3x3' : render_rot_matrix_3x3,
    'only_xyz_cov' : render_cov_matrix_no_color,
    'render_gt_rotations' : render_gt_rotations,
    'render_gt_rotations_canonical' : render_gt_rotations_canonical,
    'render_so3' : render_so3,
    'render_cholesky' : render_cholesky,
    'render_activated_scales' : render_activated_scales,
    'render_normal_opac' : render_normal_opac,
}

if __name__ == "__main__":
    pass