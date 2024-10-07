from kornia.geometry.camera.perspective import project_points    
from gecco_torch.structs import GaussianContext3d, Mode
from torch import Tensor
import torch
from einops import rearrange


import torch
from torchvision.utils import save_image

def denormalize_points_with_intrinsics(point_2d_norm: Tensor, camera_matrix: Tensor) -> Tensor:
    # unpack coordinates
    x_coord: Tensor = point_2d_norm[..., 0]
    y_coord: Tensor = point_2d_norm[..., 1]

    # unpack intrinsics
    fx: Tensor = camera_matrix[..., 0, 0]
    fy: Tensor = camera_matrix[..., 1, 1]
    cx: Tensor = camera_matrix[..., 0, 2]
    cy: Tensor = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    u_coord: Tensor = x_coord * fx + cx
    v_coord: Tensor = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)

def convert_points_from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    # we check for points at max_val
    z_vec: Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: Tensor = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec)) # wo condition wahr ist, da 1/(z_vec + eps), wo falsch da ne 1

    return scale * points[..., :-1]

def mark_points_on_images(images, coordinates_map,name):
    B, C, H, W = images.shape  # Batch size, Channels, Height, Width
    device = images.device

    # Create a copy of the images to modify
    marked_images = images.clone()
    for b in range(B):
        # num_marked = 0
        for n in range(coordinates_map.shape[1]):
            x, y = int(coordinates_map[b, n, 0, 0]), int(coordinates_map[b, n, 0, 1])
            if x >= 0 and y >= 0 and x < W and y < H:
                # Mark the point with a small cross
                # num_marked += 1
                for dx in range(-1, 2):
                    if 0 <= x + dx < W:
                        marked_images[b, :, y, x + dx] = torch.tensor([0,0,0], device=device)  # black color
                for dy in range(-1, 2):
                    if 0 <= y + dy < H:
                        marked_images[b, :, y + dy, x] = torch.tensor([0,0,0], device=device)  # vlack color
        # print(num_marked)

    # Save the marked images
    for b in range(B):
        save_image(marked_images[b], f'images/{name}_{b}.png')

    return marked_images

def project_points_with_occlusion_handling(points, camera_transform, K, image_size):
    device = points.device
    B, N, _ = points.shape  # Batch size, number of points
    H, W = image_size

    # Transform points to camera coordinates
    points_homogeneous = torch.cat([points, torch.ones(B, N, 1, device=device)], dim=-1)
    camera_coordinates = torch.einsum('bij,bpj->bpi', camera_transform, points_homogeneous)[:, :, :3]

    points_2d = convert_points_from_homogeneous(camera_coordinates)
    image_points = H * denormalize_points_with_intrinsics(points_2d, K.unsqueeze(1)) # (4,4000,2), (4,3,3) -> 4,1,3,3

    # Initialize depth buffer   
    depth_buffer = torch.full((B, H, W), float('inf'), device=device)
    visibility_map = torch.zeros((B, N), dtype=torch.bool, device=device)

    # # Handle occlusion
    # for b in range(B):
    #     for n in range(N):
    #         x, y = int(image_points[b, n, 0]), int(image_points[b, n, 1])
    #         if 0 <= x < W and 0 <= y < H:
    #             depth = camera_coordinates[b, n, 2]
    #             if depth < depth_buffer[b, y, x]:
    #                 depth_buffer[b, y, x] = depth
    #                 visibility_map[b, n] = True

    def vectorized_occlusion_handling(image_points, camera_coordinates, B, H, W):
        device = image_points.device

        # Flatten the image points and camera coordinates for easier indexing
        x = image_points[..., 0].long()
        y = image_points[..., 1].long()
        depth = camera_coordinates[..., 2]

        # Mask for valid image coordinates within bounds
        valid_x = (x >= 0) & (x < W)
        valid_y = (y >= 0) & (y < H)
        valid_mask = valid_x & valid_y

        # Flatten everything to make indexing linear and easier
        linear_indices = y * W + x  # Convert 2D indices to 1D linear indexing
        batch_indices = torch.arange(B, device=device).view(B, 1).expand_as(linear_indices)

        # Initialize depth buffer and visibility map
        depth_buffer = torch.full((B, H * W), float('inf'), device=device)  # Using flattened buffer
        visibility_map = torch.zeros((B, image_points.shape[1]), dtype=torch.bool, device=device)

        # Update only the valid points
        valid_linear_indices = linear_indices[valid_mask]
        valid_batch_indices = batch_indices[valid_mask]
        valid_depths = depth[valid_mask]

        # Get the current depth values from the buffer for comparison
        current_depths = depth_buffer[valid_batch_indices, valid_linear_indices]
        better_depth_mask = valid_depths < current_depths

        # Update the depth buffer and visibility map where the new depths are better
        depth_buffer[valid_batch_indices[better_depth_mask], valid_linear_indices[better_depth_mask]] = valid_depths[better_depth_mask]
        visibility_map[valid_mask] = better_depth_mask

        # Reshape visibility_map back to the original shape if needed
        return visibility_map
    
    # Usage example
    # camera_coordinates = torch.rand(B, N, 3) * 100  # Random camera coordinates
    # image_points = torch.rand(B, N, 2) * 100  # Random image points within 0 to 100
    # image_points[..., 0] *= W / 100.0  # Scale x coordinates
    # image_points[..., 1] *= H / 100.0  # Scale y coordinates

    visibility_map = vectorized_occlusion_handling(image_points, camera_coordinates, B, H, W)

    return image_points, visibility_map


def extract_image_features(
        self,
        geometry_diffusion: Tensor,
        features: list[Tensor],
        ctx: GaussianContext3d,
    ) -> Tensor:
        
        # print("In RayNetwork, extract image features")
        # the input geometry is in diffusion (reparameterized) space, so we need to convert it to data space
        # print(f"ctx img na: {torch.isnan(ctx.image).any()}")
        # print(f"ctx k na: {torch.isnan(ctx.K).any()}")

        if Mode.so3_diffusion in self.mode or Mode.so3_x0 in self.mode or Mode.procrustes in self.mode:
            """
            in so3 werden die rotations nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:10], ctx)
            
        elif Mode.cholesky in self.mode:
            """
            in cholesky werden die Ls nicht reparameterisiert
            """
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion[:,:,:7], ctx)
        else:
            geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)

        output_coords, visibility_map = project_points_with_occlusion_handling(geometry_data[:,:,:3].clone(), ctx.w2c, ctx.K, image_size=(400, 400))
        output_coords = output_coords.unsqueeze(2) 
        # mark_points_on_images(ctx.image,output_coords, "all")
        output_coords_vis = output_coords.clone()
        output_coords_vis[~visibility_map] = -10
        # mark_points_on_images(ctx.image,output_coords_vis, "depth_filtered")
        output_coords_vis /= 400
        # output_coords_invis = output_coords.clone()
        # output_coords_invis[visibility_map] = -10
        # mark_points_on_images(ctx.image,output_coords_invis, "gone")

        lookups = []
        # print("Perform lookups in feature map")
        # einen wert zu geben
        for feature in features:
            # print("shape feature ", feature.shape) # shape 1x (batch, 96, 34, 34), 1x (batch, 192, 17, 17), 1x (batch, 384,8,8)
            # print(f"shape grid {(hw_flat * 2 - 1).shape}")
            lookup = torch.nn.functional.grid_sample(
                feature, output_coords_vis * 2 - 1, align_corners=False
            )

            # grid sample findet die feature werte (input) an den im grid geforderten stellen heraus
            # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
            # mit padding mode zero werden zeros ausgegeben für grid werte, die außerhalb von -1,1 liegen
            # hw_flat gibt ja die location an und zwar sozusagen relativ gesehen. Wenn feature die shape batch, 96, 100, 100 hat, 
            # dann ist die location (0,0) in hw_flat das center des der, also (50,50) in feature

            # print("shape lookup before rearanging", lookup.shape)
            lookup = rearrange(lookup, "b c n 1 -> b n c")

            
            # gebe den punkten die nicht sichtbar sind den mean des images
            means = feature.mean(dim=[2, 3])  # Resulting shape will be (4, 96)

            # Check each vector in tensor_large to see if it's a zero vector
            mask = lookup.sum(dim=2) == 0  # A mask of where entire vectors are zero
            # Expand means to match the number of vectors in tensor_large
            expanded_means = means.unsqueeze(1).expand(-1, 4000, -1)  # Expand means to (4, 4000, 96)

            # Replace zero vectors with the corresponding means
            lookup[mask] = expanded_means[mask]
            
            lookups.append(lookup)

        # concatenate the lookups
        lookups = torch.cat(lookups, dim=-1)
        # print(f"extract image features na: {torch.isnan(lookups).any()}")
        # print("shape lookups after concatenation ", lookups.shape) # (batchsize, 2048, 672) # 672 = 96+192+384, also alle features zusammengeklatscht
        return lookups
    
"""
how does grid_sample work (torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None))
Wir nehmen die Werte aus input an den Stellen spezifiziert in grid. 
Da nicht alle Werte in grid perfekt auf genau eine input Koordinate zeigt, wird interpoliert.
Nur 5D oder 4D inputs sind supported
"""
