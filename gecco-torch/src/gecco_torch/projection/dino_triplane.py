from kornia.geometry.camera.perspective import project_points    
from gecco_torch.structs import GaussianContext3d, Mode
from torch import Tensor
import torch
from einops import rearrange
# from gecco_torch.projection.dino_dirs_vis import Visualize_cameras

def compute_plucker_coordinates(points, directions):
    """
    Compute Plücker coordinates for a set of rays given points and directions.
    
    Args:
    - points (torch.Tensor): Tensor of shape (N, 3) where each row represents a 3D point through which a ray passes.
    - directions (torch.Tensor): Tensor of shape (N, 3) where each row is a unit vector representing the ray's direction.
    
    Returns:
    - torch.Tensor: Tensor of shape (N, 6) where each row represents the Plücker coordinates (direction, moment) of a ray.
    """
    # Ensure directions are unit vectors
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # Compute the moment vector m = p x d
    moments = torch.cross(points, directions, dim=-1)
    
    # Concatenate direction and moment vectors to form the Plücker coordinates
    plucker_coordinates = torch.cat([directions, moments], dim=-1)
    
    return plucker_coordinates

def location_direction(ctx) -> Tensor:
    """
    Compute the Pluecker embedding of a set of points. The Pluecker embedding is a 6D vector that
    represents a line in 3D space. It is computed by taking the outer product of each pair of points
    and stacking them into a 6D vector.

    Args:
        points: A tensor of shape (batch, n_points, 3).

    Returns:
        A tensor of shape (batch, n_points * (n_points - 1) // 2, 6).
    """
    K = ctx.K
    c2w = ctx.c2w
    w2c = ctx.w2c
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    batch_size = K.shape[0]
    device = K.device
    H, W = 400, 400

    """lara"""
    scale = 1
    ixts = K
    c2ws = c2w
    H, W = int(H * scale), int(W * scale)
    ixts[:, :2] *= scale
    ixts = ixts * torch.tensor([400 + 1, 400 + 1, 1],device=device)

    rays_o = c2ws[:, :3, 3][:, None, None]
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
    XYZ = torch.cat((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, torch.ones_like(X)[:, :, None]), dim=-1).cuda()
    i2ws = torch.linalg.inv(ixts).transpose(1, 2) @ c2ws[:, :3, :3].transpose(1, 2)
    XYZ = torch.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(1, H, W, 1)

    R_transpose = R.transpose(1, 2)  # Transpose the last two dimensions: Shape (3, 3, 3)
    Rt_t = torch.bmm(R_transpose, t.unsqueeze(2)).squeeze(2)  # Batched matrix-vector multiplication: Shape (3, 3)
    neg_Rt_t = -Rt_t  # Negate the result
    neg_Rt_t = neg_Rt_t.unsqueeze(1).unsqueeze(1).expand(batch_size, H, W, 3)
    m = torch.cross(neg_Rt_t, XYZ, dim=-1)  # Cross product: Shape (3, 3)
    # XYZ sind directions
    return rays_o, XYZ, m
    """lara end"""
    # Create a meshgrid for pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, W-1, W,device=device), torch.linspace(0, H-1, H,device=device), indexing='xy')
    pixels = torch.stack([i, j, torch.ones_like(i)], -1)  # Shape: [H, W, 3]
    # pixels [0,0,1]
    #        [1,0,1]
    #        [2,0,1]
    #        ...
    #        [0,1,1]
    #        [1,1,1]
    #        ...
    #       [W-1,H-1,1]
    
    # Reshape for batch processing
    pixels = pixels.reshape(-1, 3).unsqueeze(0)  # Shape: [1, H*W, 3]
    
    # Inverse of intrinsic matrix K to transform to camera coordinates
    inv_K = torch.linalg.inv(K)  # Assuming K has shape [batch_size, 3, 3]
    R_expanded = R.unsqueeze(1).expand(-1, H*W, -1, -1)  # Shape: [3, 1600000, 3, 3]
    inv_K_expanded = inv_K.unsqueeze(1).expand(-1, H*W, -1, -1)  # Shape: [3, 1600000, 3, 3]

    # Compute d = R K^-1 u
    temp_matrix = torch.matmul(R_expanded, inv_K_expanded)  # Shape: [3, 1600000, 3, 3]
    u_expanded = pixels.expand(batch_size, -1, -1)  # Shape: [3, 1600000, 3]
    d = torch.matmul(temp_matrix, u_expanded.unsqueeze(-1)).squeeze(-1)  # Shape: [3, 1600000, 3]
    d = d.reshape(batch_size, H, W, 3)
    d = d / torch.norm(d, dim=-1, keepdim=True)

    rays_camera = torch.matmul(inv_K[:, None, :, :], pixels.unsqueeze(-1))  # Shape: [batch_size, H*W, 3, 1]
    rays_camera = rays_camera.squeeze(-1)  # Remove the last dimension, shape: [batch_size, H*W, 3]
    
    # Normalize ray directions
    rays_camera[..., :2] /= rays_camera[..., 2:3]
    
    # Apply camera-to-world transformation
    rays_world = torch.matmul(R.unsqueeze(1), rays_camera.unsqueeze(-1)).squeeze(-1).reshape(batch_size, H, W, 3)
    origins_world = t.unsqueeze(1).unsqueeze(1).expand(batch_size, H, W, 3)

    # import numpy as np
    # np.savez("camdirs.npz", cam = origins_world.detach().cpu().numpy(), dirs = d.detach().cpu().numpy())

    R_transpose = R.transpose(1, 2)  # Transpose the last two dimensions: Shape (3, 3, 3)
    Rt_t = torch.bmm(R_transpose, t.unsqueeze(2)).squeeze(2)  # Batched matrix-vector multiplication: Shape (3, 3)
    neg_Rt_t = -Rt_t  # Negate the result
    neg_Rt_t = neg_Rt_t.unsqueeze(1).unsqueeze(1).expand(batch_size, H, W, 3)
    m = torch.cross(neg_Rt_t, d, dim=-1)  # Cross product: Shape (3, 3)

    return origins_world, d, m







def extract_triplane_features_old(
        self,
        geometry_diffusion: Tensor,
        features: list[Tensor],
        ctx: GaussianContext3d,
        triplane_xy,
        triplane_yz,
        triplane_xz,
    ) -> Tensor:
        
        # print("In RayNetwork, extract image features")
        # the input geometry is in diffusion (reparameterized) space, so we need to convert it to data space
        # print(f"ctx img na: {torch.isnan(ctx.image).any()}")
        # print(f"ctx k na: {torch.isnan(ctx.K).any()}")
        features = features[0]
        features_triplanexy = triplane_xy(features)
        features_triplaneyz = triplane_yz(features)
        features_triplanexz = triplane_xz(features) # shape (b, 384, 28, 28)


        if Mode.so3_diffusion in self.mode or Mode.so3_x0 in self.mode:
            """
            in so3 werden die rotations nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:10], ctx)
        elif Mode.procrustes in self.mode:
            """
            in procrustes werden die 3x3 rot matrix nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:10], ctx)
        elif Mode.cholesky in self.mode:
            """
            in cholesky werden die Ls nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:7], ctx)
        else:
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)

        n_points = 400

        # Generate points between 0 and 1
        points = torch.linspace(0, 1, n_points,device=geometry_data.device)

        # Create a 2D grid
        xx, yy = torch.meshgrid(points, points, indexing='xy')

        # Flatten the grid to form (n_points * n_points, 2)
        coordinates = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        coordinates = coordinates.unsqueeze(0).expand(geometry_data.shape[0], -1, -1).unsqueeze(2)

        # sig: grid_sample(input, grid)
        # für grid sample: input shape (b, depth, h, w) -> (b, 384, 28, 28)
        # grid shape: (b, n, 1, 2)

        lookupxy = torch.nn.functional.grid_sample(
                features_triplanexy, coordinates * 2 - 1, align_corners=False
            ) 
        lookupxy = lookupxy.reshape((geometry_data.shape[0], 384, n_points, n_points))
        # result: shape 3, 384, 4000

        lookupyz = torch.nn.functional.grid_sample(
                features_triplaneyz, coordinates * 2 - 1, align_corners=False
            ) 
        lookupyz =lookupyz.reshape((geometry_data.shape[0], 384, n_points, n_points))

        lookupxz = torch.nn.functional.grid_sample(
                features_triplaneyz, coordinates * 2 - 1, align_corners=False
            ) 
        lookupxz = lookupxz.reshape((geometry_data.shape[0], 384, n_points, n_points))

        triplane_total = lookupxy + lookupyz + lookupxz

        # add plucker embeddings
        cam_locations, directions, m = location_direction(ctx)
        plucker_coordinates = torch.cat([directions, m], dim=-1)
        # plucker_coordinates = compute_plucker_coordinates(cam_locations, directions)
        plucker_coordinates = rearrange(plucker_coordinates, "b h w f -> b f h w")
        total_feature_map = torch.cat([triplane_total, plucker_coordinates],dim = 1)

        # kein plucker
        # total_feature_map = triplane_total # torch.cat([triplane_total, plucker_coordinates],dim = 1)
        return total_feature_map # shape (batch, 390, 400, 400) # 390 = 384 + 6 wenn dino small (produziert 384 features)

def extract_triplane_features(
        self,
        geometry_diffusion: Tensor,
        features: list[Tensor],
        ctx: GaussianContext3d,
        triplane_xy,
        triplane_yz,
        triplane_xz,
    ) -> Tensor:
        
        # print("In RayNetwork, extract image features")
        # the input geometry is in diffusion (reparameterized) space, so we need to convert it to data space
        # print(f"ctx img na: {torch.isnan(ctx.image).any()}")
        # print(f"ctx k na: {torch.isnan(ctx.K).any()}")
        features = features[0]
        features_triplanexy = triplane_xy(features)
        features_triplaneyz = triplane_yz(features)
        features_triplanexz = triplane_xz(features) # shape (b, 384, 28, 28)


        if Mode.so3_diffusion in self.mode or Mode.so3_x0 in self.mode:
            """
            in so3 werden die rotations nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:10], ctx)
        elif Mode.procrustes in self.mode:
            """
            in procrustes werden die 3x3 rot matrix nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:10], ctx)
        elif Mode.cholesky in self.mode:
            """
            in cholesky werden die Ls nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:7], ctx)
        else:
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)

        # sig: grid_sample(input, grid)
        # für grid sample: input shape (b, depth, h, w) -> (b, 384, 28, 28)
        # grid shape: (b, n, 1, 2)
    
        geometry_data_xy = geometry_data[:,:,:2].clone()
        geometry_data_xy_normalized = (geometry_data_xy - geometry_data_xy.min())/(geometry_data_xy.max() - geometry_data_xy.min())
        geometry_data_xy_normalized = geometry_data_xy_normalized.unsqueeze(2)
        geometry_data_yz = geometry_data[:,:,1:3].clone()
        geometry_data_yz_normalized = (geometry_data_yz - geometry_data_yz.min())/(geometry_data_yz.max() - geometry_data_yz.min())
        geometry_data_yz_normalized = geometry_data_yz_normalized.unsqueeze(2)
        geometry_data_xz = torch.cat([geometry_data[:,:,0].unsqueeze(-1).clone(), geometry_data[:,:,2].unsqueeze(-1).clone()], dim = -1)
        geometry_data_xz_normalized = (geometry_data_xz - geometry_data_xz.min())/(geometry_data_xz.max() - geometry_data_xz.min())
        geometry_data_xz_normalized = geometry_data_xz_normalized.unsqueeze(2)

        lookupxy = torch.nn.functional.grid_sample(
                features_triplanexy, geometry_data_xy_normalized * 2 - 1, align_corners=False
            ) 
        # result: shape 3, 384, 4000, 1
        lookupxy = rearrange(lookupxy, "b f p 1 -> b p f")
        lookupyz = torch.nn.functional.grid_sample(
                features_triplaneyz, geometry_data_yz_normalized * 2 - 1, align_corners=False
            ) 
        lookupyz = rearrange(lookupyz, "b f p 1 -> b p f")
        lookupxz = torch.nn.functional.grid_sample(
                features_triplanexz, geometry_data_xz_normalized * 2 - 1, align_corners=False
            ) 
        lookupxz = rearrange(lookupxz, "b f p 1 -> b p f")
        triplane_total = lookupxy + lookupyz + lookupxz # shape batch, 4000, 384

        # add plucker embeddings
        cam_locations = ctx.c2w[:,:3,3][:,None]
        cam_locations = cam_locations.repeat(1,4000,1)
        
        directions = geometry_data[:,:,:3] - cam_locations
        moments = torch.cross(cam_locations, directions)

        plucker_coordinates = torch.cat([directions, moments], dim=-1)

        # total_feature_map = torch.cat([triplane_total, plucker_coordinates],dim = -1)

        # kein plucker
        total_feature_map = triplane_total # torch.cat([triplane_total, plucker_coordinates],dim = 1)
        return total_feature_map # shape (batch, 390, 400, 400) # 390 = 384 + 6 wenn dino small (produziert 384 features)