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
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from gsplat.gsplat_cam_renderer import GaussianRasterizer2 as GaussianDepthRasterizer
from gecco_torch.gsplat.gsplat_cam_renderer import GaussianRasterizer as GaussianDepthRasterizer
from gecco_torch.gsplat.gsplat_cam_renderer import GaussianRasterizationSettings as GaussianDepthRasterizationSettings
from gecco_torch.scene.gaussian_model_template import GaussianModelTemplate

def render(pc : GaussianModelTemplate, bg_color : torch.Tensor, camera,**kwargs):
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


    override_color = kwargs.get('override_color',False)
    scaling_modifier = 1.0
    raster_settings = GaussianDepthRasterizationSettings(
        image_height=camera.imsize,
        image_width=camera.imsize,
        tanfovx=camera.tanfovx,
        tanfovy=camera.tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        sh_degree=pc.active_sh_degree,
        prefiltered=False,
        debug= False #pipe.debug
    )
    rasterizer = GaussianDepthRasterizer(raster_settings=raster_settings)
    

    means3D = pc.get_xyz
    if torch.isnan(means3D).any():
        print("ERROR RENDER: means3D has nan values")
    means2D = screenspace_points
    opacity = pc.get_opacity
    if torch.isnan(opacity).any():
        print("ERROR RENDER: opacity has nan values")

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if kwargs.get('use_cov3D',False):
        cov3D_precomp = kwargs['covs']
        if torch.isnan(cov3D_precomp).any():
            print("ERROR RENDER: cov3D has nan values")
    else:
        scales = pc.get_scaling
        if torch.isnan(scales).any():
            print("ERROR RENDER: scales has nan values")
        rotations = pc.get_rotation
        if torch.isnan(rotations).any():
            print("ERROR RENDER: rotations has nan values")

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if not override_color:
        shs = pc.get_features
        if torch.isnan(shs).any():
            print("ERROR RENDER: shs has nan values")

    else:
        colors_precomp = kwargs['rgbs']

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # print(means2D) # all zeros
    if cov3D_precomp is not None:
        if cov3D_precomp.dtype != means3D.dtype:
            cov3D_precomp = cov3D_precomp.type(means3D.dtype)
            
    try:
        rendered_image, depth, radii = rasterizer(
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.projection_matrix,
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    except Exception as e:
        print("Error in __init__ gaussian_renderer:")
        print(e)
    
    return_dict = {"render": rendered_image,
            "depth": depth.squeeze(0),
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(radii)
    return return_dict