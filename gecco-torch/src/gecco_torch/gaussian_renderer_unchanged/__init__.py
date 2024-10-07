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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from gsplat.gsplat_cam_renderer import GaussianRasterizer2 as GaussianDepthRasterizer
from gecco_torch.gsplat.gsplat_cam_renderer import GaussianRasterizer as GaussianDepthRasterizer
from gecco_torch.gsplat.gsplat_cam_renderer import GaussianRasterizationSettings as GaussianDepthRasterizationSettings
from gecco_torch.scene.gaussian_model_template import GaussianModelTemplate

def render(viewpoint_camera, pc : GaussianModelTemplate, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,**kwargs):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # print("render")
    # print(viewpoint_camera.camera_center)
    # print(viewpoint_camera.world_view_transform)
    # print(viewpoint_camera.full_proj_transform)
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # print(torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda").shape) # shape (1000,3) -> (number of points in gaussian,3) 
    # (in the first couple hundreds iterations its always the number of init points)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 # +0 is like a no-op
    # print(screenspace_points.shape)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    if kwargs['rasterizer'] == 'depth':
        raster_settings = GaussianDepthRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,# white or black tensor of shape 3
            scale_modifier=scaling_modifier,# float 1
            sh_degree=pc.active_sh_degree,# immer 3 ab irgendeiner iteration
            prefiltered=False,
            debug=pipe.debug # False
        )
        rasterizer = GaussianDepthRasterizer(raster_settings=raster_settings)
        
    elif kwargs['rasterizer'] == 'normal':
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color, 
            scale_modifier=scaling_modifier, 
            campos=viewpoint_camera.camera_center,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform, # in benutzung mit der depth restiratzation2 api: projection_matrix full_projection_matrix
            sh_degree=pc.active_sh_degree,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    else:
        raise NotImplementedError(f"Rasterizer {kwargs['rasterizer']} not implemented")
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # print(pipe.compute_cov3D_python) # false
    if pipe.compute_cov3D_python or kwargs.get("optimize_with_cov",False):
        cov3D_precomp = pc.get_covariance()
    else:
        # for shapenet here
        if not kwargs.get("use_cov",False):
            scales = pc.get_scaling
            rotations = pc.get_rotation

    # cov_matrices = torch.zeros((cov3D_precomp.shape[0], 3, 3),device=pc.get_xyz.device,dtype=torch.float32)

    # # Assign the elements directly using advanced indexing
    # cov_matrices[:, 0, 0] = cov3D_precomp[:, 0]
    # cov_matrices[:, 0, 1] = cov3D_precomp[:, 1]
    # cov_matrices[:, 0, 2] = cov3D_precomp[:, 2]
    # cov_matrices[:, 1, 0] = cov3D_precomp[:, 1]  # Symmetric element
    # cov_matrices[:, 1, 1] = cov3D_precomp[:, 3]
    # cov_matrices[:, 1, 2] = cov3D_precomp[:, 4]
    # cov_matrices[:, 2, 0] = cov3D_precomp[:, 2]  # Symmetric element
    # cov_matrices[:, 2, 1] = cov3D_precomp[:, 4]  # Symmetric element
    # cov_matrices[:, 2, 2] = cov3D_precomp[:, 5]
    # Ls = torch.zeros((pc.get_xyz.shape[0],6))

    # for i,matrix in enumerate(cov_matrices):
    #     try:
    #         L = torch.linalg.cholesky(matrix + 0*torch.ones_like(cov_matrices[i]))
    #         # die diagonal werte log, weil wir die später exp
    #         Ls[i,0] = torch.log(L[0,0])
    #         Ls[i,1] = torch.log(L[1,1])
    #         Ls[i,2] = torch.log(L[2,2])
    #         Ls[i,3] = L[1,0]
    #         Ls[i,4] = L[2,0]
    #         Ls[i,5] = L[2,1]
    #     except:
    #         try:
    #             L = torch.linalg.cholesky(matrix + 1e-7*torch.ones_like(cov_matrices[i]))
    #             # die diagonal werte log, weil wir die später exp
    #             Ls[i,0] = torch.log(L[0,0])
    #             Ls[i,1] = torch.log(L[1,1])
    #             Ls[i,2] = torch.log(L[2,2])
    #             Ls[i,3] = L[1,0]
    #             Ls[i,4] = L[2,0]
    #             Ls[i,5] = L[2,1]
    #         except:
    #             # die diagonal werte log, weil wir die später exp
    #             Ls[i,:] = Ls[i-1,:].clone()


    # Ls_vals = Ls.clone()
    # Ls = torch.zeros((pc.get_xyz.shape[0],3,3),device=pc.get_xyz.device,dtype=torch.float32)
    # Ls[:,0,0] = Ls_vals[:,0]
    # Ls[:,1,1] = Ls_vals[:,1]
    # Ls[:,2,2] = Ls_vals[:,2]
    # Ls[:,1,0] = Ls_vals[:,3]
    # Ls[:,2,0] = Ls_vals[:,4]
    # Ls[:,2,1] = Ls_vals[:,5]

    # # diagonal elements
    # diagonal = Ls.diagonal(dim1=-2, dim2=-1)
    # # exp
    # diagonal_exp = torch.exp(diagonal)
    # # insert diagonal values
    # Ls.diagonal(dim1=-2, dim2=-1).copy_(diagonal_exp)


    # cov3D_precomp_3x3 = torch.bmm(Ls, Ls.transpose(1, 2)).type(torch.float32)
    # cov3D_precomp = torch.zeros((cov3D_precomp.shape[0],6),device=pc.get_xyz.device,dtype=torch.float32)
    # cov3D_precomp[:,0] = cov3D_precomp_3x3[:,0,0]
    # cov3D_precomp[:,1] = cov3D_precomp_3x3[:,0,1]
    # cov3D_precomp[:,2] = cov3D_precomp_3x3[:,0,2]
    # cov3D_precomp[:,3] = cov3D_precomp_3x3[:,1,1]
    # cov3D_precomp[:,4] = cov3D_precomp_3x3[:,1,2]
    # cov3D_precomp[:,5] = cov3D_precomp_3x3[:,2,2]

    if kwargs.get("use_cov",False):
        scales = None
        rotations = None
        cov3D_precomp = kwargs.get("cov3D_precomp")

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # print(pipe.convert_SHs_python) # false
        if pipe.convert_SHs_python:
            features = pc.get_features
            # features[:,1:,:] = 0
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # for shapenet no convert sh
            shs = pc.get_features
    else:
        colors_precomp = override_color

    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5

    def RGB2SH(rgb):
        return (rgb - 0.5) / C0
    
    # colors_precomp = SH2RGB(shs[:,0,:])
    # shs = None
    # shs = RGB2SH(0.5*torch.ones((pc.get_xyz.shape[0],16,3),device=pc.get_xyz.device,dtype=torch.float32))

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # print(means2D) # all zeros
    if kwargs['rasterizer'] == 'depth':
        # print(f"viewmatrix world view transform shape: {viewpoint_camera.world_view_transform.shape}")
        # print(f"projmatrix projection matrix shape: {viewpoint_camera.projection_matrix.shape}")
        # print(f"world view: {viewpoint_camera.world_view_transform}")
        # print(f"projection matrix: {viewpoint_camera.projection_matrix}")
        # print("tanv fovx: ",tanfovx)
        rendered_image, depth, radii = rasterizer(
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.projection_matrix,
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        # print(radii)
        return_dict = {"render": rendered_image,
                "depth": depth.squeeze(0),
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
        
    elif kwargs['rasterizer'] == 'normal':
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(radii)
    return return_dict