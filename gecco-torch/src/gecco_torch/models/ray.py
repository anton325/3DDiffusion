"""
RayNetwork is a wrapper around SetTransformer that handles the projective lookup of CNN
features and integrates them in the network inputs. The CNN feature computation is handled by the Diffusion object -
this class only handles the lookup.

For unconditional modelling you should use gecco_torch.models.LinearLift instead, which simply embeds the geometry and passes it
through the SetTransformer.
"""
import torch
from torch import nn, Tensor
from einops import rearrange
from kornia.geometry.camera.perspective import project_points

from gecco_torch.reparam import Reparam
from gecco_torch.structs import Context3d
from gecco_torch.models.feature_pyramid import FeaturePyramidContext
from gecco_torch.models.set_transformer import SetTransformer


class GroupNormBNC(nn.GroupNorm):
    """
    A GroupNorm that supports batch channel last format (transformer default).
    input has batch channel as last channel, we swap that for the normal group norm
    then for the output we swab back
    """

    def forward(self, tensor_bnc: Tensor) -> Tensor:
        assert tensor_bnc.ndim == 3

        tensor_bcn = rearrange(tensor_bnc, "b n c -> b c n")
        result_bcn = super().forward(tensor_bcn)
        # print(f"result_bcn na: {torch.isnan(result_bcn).any()}")
        rearranged_bcn = rearrange(result_bcn, "b c n -> b n c")
        return rearranged_bcn


class RayNetwork(nn.Module):
    def __init__(
        self,
        backbone: SetTransformer,
        reparam: Reparam,
        context_dims: list[int],
    ):
        """
        Args:
            backbone: The SetTransformer to use.
            reparam: The reparameterization scheme to use.
            context_dims: The number of channels in each CNN feature map.
        """
        super().__init__()
        self.backbone = backbone
        self.reparam = reparam
        self.context_dims = context_dims

        self.xyz_embed = nn.Linear(reparam.dim, backbone.feature_dim)
        self.img_feature_proj = nn.Sequential(
            GroupNormBNC(16, sum(context_dims), affine=False),
            nn.Linear(sum(context_dims), backbone.feature_dim),
        )
        self.output_proj = nn.Sequential(
            GroupNormBNC(16, backbone.feature_dim, affine=False), 
            nn.Linear(backbone.feature_dim, reparam.dim),
        )

    def extra_repr(self) -> str:
        return f"context_dims={self.context_dims}"

    def extract_image_features(
        self,
        geometry_diffusion: Tensor,
        features: list[Tensor],
        ctx: Context3d,
    ) -> Tensor:
        
        # print("In RayNetwork, extract image features")
        # the input geometry is in diffusion (reparameterized) space, so we need to convert it to data space
        # print(f"ctx img na: {torch.isnan(ctx.image).any()}")
        # print(f"ctx k na: {torch.isnan(ctx.K).any()}")

        geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)
        # print(f"geometry_data na: {torch.isnan(geometry_data).any()}")
        # print(f"Shape geometry data {geometry_data.shape}") # (batch, 2048, 3)

        # print(f"Shape K {ctx.K.shape}")
        # print(f"shape input project_points ctx K unsqueezed {ctx.K.unsqueeze(1).shape}")
        # project the geometry to the image plane

        # transform pointcloud to camera space
        # print(ctx.w2c)

        
        # die in place veränderung der daten ist nicht schlimm, weil wir die daten schon embedded haben
        for b in range(ctx.w2c.shape[0]):
            geometry_data[b,:,:] = torch.einsum("ab,nb->na", ctx.w2c[b,:3, :3], geometry_data[b,:,:]) + ctx.w2c[b,:3, -1] # adding translation part

        hw_01 = project_points(geometry_data, ctx.K.unsqueeze(1))[..., :2]

        # make image
        # import numpy as np
        # import matplotlib.pyplot as plt
        # point_coords = hw_01[0].cpu().numpy()
        # image = np.ones((400,400,3))
        # for point in point_coords:
        #     coordx = int(point[1]*400)
        #     coordy = int(point[0]*400)
        #     image[coordx,coordy] = np.array([0,0,0])
        # plt.imsave("image_project.png", image)

        # plt.imsave("image_gt.png",np.transpose(ctx.image[0].cpu().numpy(),(1,2,0)))


        # print(f"hw_01 na: {torch.isnan(hw_01).any()}")
        # print(f"Shape output project_points {project_points(geometry_data, ctx.K.unsqueeze(1)).shape}")
        # print("shape output project points after indexing ", hw_01.shape) # shape (batch,2048,2), 2 weil wir die 3D Punkte auf 2D image koordinaten projizieren
        hw_flat = rearrange(hw_01, "b n t -> b n 1 t")
        # print("shape after rearrange ", hw_flat.shape) # shape (batch,2048,1,2)

        # perform the projective lookup on each feature map
        lookups = []
        # print("Perform lookups in feature map")
        for feature in features:
            # print("shape feature ", feature.shape) # shape 1x (batch, 96, 34, 34), 1x (batch, 192, 17, 17), 1x (batch, 384,8,8)
            # print(f"shape grid {(hw_flat * 2 - 1).shape}")
            lookup = torch.nn.functional.grid_sample(
                feature, hw_flat * 2 - 1, align_corners=False # durch das * 2 - 1 ist das grid zwischen -1 und 1
        )
            # grid sample findet die feature werte (input) an den im grid geforderten stellen heraus
            # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)

            # print("shape lookup before rearanging", lookup.shape)
            lookup = rearrange(lookup, "b c n 1 -> b n c")
            # print("shape lookup ", lookup.shape)
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
    
    

    def forward(
        self,
        geometry: Tensor,
        t: Tensor,
        raw_ctx: Context3d,
        post_context: FeaturePyramidContext,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor | list[Tensor] | None]:
        # print("In forward RayNetwork")
        # embed the geometry and the time. Since we use Gaussian activations, we can just use a single linear layer
        # geometry is the reparameterized 3D point cloud
        # print(geometry.dtype)
        xyz_features = self.xyz_embed(geometry)
        # print("ray forward t ",t)
        # print("ray forward t ",t.shape) # (batch_size, 1, 1)
        t_features = t
        # print(f"forward ray geometry na: {torch.isnan(geometry).any()}")
        # extract_image_features is suspected of causing divergences in fp16
        with torch.autocast(device_type="cuda", enabled=False): # enabled False sagt kein autocast
            geometry_f32 = geometry.to(dtype=torch.float32)
            features_f32 = [f.to(dtype=torch.float32) for f in post_context.features]
            # print(f"geometry_f32 na: {torch.isnan(geometry_f32).any()}")
            raw_ctx_f32 = raw_ctx.apply_to_tensors(lambda t: t.to(dtype=torch.float32)) # apply_to_tensors applied eine function (hier to) to all tensors in the context
            img_features_raw = self.extract_image_features(
                geometry_f32, features_f32, raw_ctx_f32
            )

        # the projection brings the number of channels to the dimension of the SetTransformer
        img_features = self.img_feature_proj(img_features_raw)
        # print(f"shape der img_features {img_features.shape}") 
        # attach features indem man sie einfach aufaddiert
        point_features = xyz_features + img_features
        # print(f"Shape der point features (input backbone) {point_features.shape}") # (batch,2048,384)

        # run the SetTransformer
        processed, out_cache = self.backbone(
            point_features, t_features, do_cache, cache
        )
        # print(f"out backbone: {processed}")
        # for b_index in range(processed.shape[0]):
        #     for n_index in range(processed.shape[1]):
        #         for c_index in range(processed.shape[2]):
        #             if torch.isnan(processed[b_index][n_index][c_index]):
        #                 print(f"nan in processed at b_index {b_index}, n_index {n_index}, c_index {c_index}")
        # print(f"output contains na: {torch.isnan(processed).any()}")
        output_proj = self.output_proj(processed)
        # print(f"output_proj na: {torch.isnan(output_proj).any()}")
        return output_proj, out_cache
