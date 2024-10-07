"""
RayNetwork is a wrapper around SetTransformer that handles the projective lookup of CNN
features and integrates them in the network inputs. The CNN feature computation is handled by the Diffusion object -
this class only handles the lookup.

For unconditional modelling you should use gecco_torch.models.LinearLift instead, which simply embeds the geometry and passes it
through the SetTransformer.
"""
import torch
from torch import nn, Tensor
import lietorch
from einops import rearrange

from gecco_torch.reparam import Reparam
from gecco_torch.structs import GaussianContext3d, Mode
from gecco_torch.models.feature_pyramid import FeaturePyramidContext
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.utils.rotation_utils import rotation_matrix_to_quaternion_torch_batched, mult_quats_vectorized_torch_batched
import gecco_torch.projection.gecco_projection as gecco_projection 
import gecco_torch.projection.dino_triplane as dino_triplane
import gecco_torch.projection.pc_projection as pc_projection


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
        mode: Mode,
        render_fn: callable,
        dinov2 = None,
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
        self.mode = mode
        self.render_fn = render_fn
        if Mode.dino_triplane in mode:
            self.triplane_conv_xy = nn.Conv2d(384,384,kernel_size=3,padding=1)
            self.triplane_conv_yz = nn.Conv2d(384,384,kernel_size=3,padding=1)
            self.triplane_conv_xz = nn.Conv2d(384,384,kernel_size=3,padding=1)
            # self.img_feature_proj = nn.Sequential(
            #     GroupNormBNC(15, sum(context_dims) + 6, affine=False), # 6 plücker dazu, dann 15 channels (390/15 = 26)
            #     nn.Linear(sum(context_dims) + 6, backbone.feature_dim), # 6 plücker dazu
            # )
            # kein plucker
            self.img_feature_proj = nn.Sequential(
                GroupNormBNC(16, sum(context_dims), affine=False),
                nn.Linear(sum(context_dims), backbone.feature_dim),
            )
        else:
            self.img_feature_proj = nn.Sequential(
                GroupNormBNC(16, sum(context_dims), affine=False),
                nn.Linear(sum(context_dims), backbone.feature_dim),
            )
            
        if Mode.so3_diffusion in mode or Mode.so3_x0 in mode:
            """
            Im so3 diffusion mode werden in reparam nur [xyz, rgb, scale, opacity] normalisiert, deswegen hat es nur 10 dimensionen,
            aber im Training kommen natürlich die rotations noch dazu, deswegen 4 mehr
            """
            self.xyz_embed = nn.Linear(reparam.dim + 4, backbone.feature_dim)

        elif Mode.cholesky in mode:
            """
            Im cholesky mode werden in reparam nur [xyz, rgb, opacity] normalisiert, deswegen hat es nur 7 dimensionen,
            aber im Training kommen natürlich die Ls noch dazu, deswegen 6 mehr
            """
            self.xyz_embed = nn.Linear(reparam.dim + 6, backbone.feature_dim)
        elif Mode.procrustes in mode:
            self.xyz_embed = nn.Linear(reparam.dim + 9, backbone.feature_dim)
        else:
            self.xyz_embed = nn.Linear(reparam.dim, backbone.feature_dim) # backbone feature dims sind die vom set transformer und die sind fix


        if Mode.so3_diffusion in mode or Mode.so3_x0 in mode:
            self.output_proj = nn.Sequential(
                GroupNormBNC(16, backbone.feature_dim, affine=False),
                nn.Linear(backbone.feature_dim, reparam.dim + 6), # 6 extra, weil mu aus 6 elementen besteht, aus denen konstruieren wir zum Schluss die rotation matrix
            )

            self.scale_layer = nn.Linear(6, 1) # aus dem mu die scale berechnen
        elif Mode.cholesky in mode:
            self.output_proj = nn.Sequential(
                GroupNormBNC(16, backbone.feature_dim, affine=False),
                nn.Linear(backbone.feature_dim, reparam.dim + 6), # 6 extra, wir L ausgeben, also 6 elemente
            )
        elif Mode.procrustes in mode:
            self.output_proj = nn.Sequential(
                GroupNormBNC(16, backbone.feature_dim, affine=False),
                nn.Linear(backbone.feature_dim, reparam.dim + 9), # 9 extra, wir geben die 3x3 matrix aus
            )
        else:
            self.output_proj = nn.Sequential(
                GroupNormBNC(16, backbone.feature_dim, affine=False), 
                nn.Linear(backbone.feature_dim, reparam.dim),
            )


    def extra_repr(self) -> str:
        return f"context_dims={self.context_dims}"

    def forward(
        self,
        geometry: Tensor,
        t: Tensor,
        raw_ctx: GaussianContext3d,
        post_context: FeaturePyramidContext,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor | list[Tensor] | None]:
        # print("In forward RayNetwork")
        # embed the geometry and the time. Since we use Gaussian activations, we can just use a single linear layer
        # geometry is the reparameterized 3D point cloud
        # print(geometry.dtype)
        xyz_features = self.xyz_embed(geometry) # xyz features shape (batch, 4000,384)
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

            if Mode.dino_triplane in self.mode:
                img_features_raw = dino_triplane.extract_triplane_features(self,
                    geometry_f32, features_f32, raw_ctx_f32, self.triplane_conv_xy, self.triplane_conv_yz, self.triplane_conv_xz
                )

            elif Mode.gecco_projection in self.mode:
                img_features_raw = gecco_projection.extract_image_features(self,
                    geometry_f32, features_f32, raw_ctx_f32
                )
            elif Mode.depth_projection in self.mode:
                img_features_raw = pc_projection.extract_image_features(self,
                    geometry_f32, features_f32, raw_ctx_f32
                )
            else:
                raise Exception("Invalid point cloud feature extraction mode for RayNetwork")

        # the projection brings the number of channels to the dimension of the SetTransformer
        img_features = self.img_feature_proj(img_features_raw)
        # print(f"shape der img_features {img_features.shape}")
        # attach features indem man sie einfach aufaddiert
        point_features = xyz_features + img_features
        # print(f"Shape der point features (input backbone) {point_features.shape}") # (batch,2048,384)

        # run the SetTransformer
        processed, out_cache = self.backbone(
            point_features, t_features, do_cache, cache, raw_ctx.mask_points
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

        if Mode.so3_diffusion in self.mode or Mode.so3_x0 in self.mode:
            batchsize = output_proj.shape[0]
            mu = output_proj[:,:,10:] # ab 10 sind die 6 für die rotations matrix
            scale = self.scale_layer(mu)
            scale = nn.functional.softplus(scale) + 0.0001
            
            mu = mu.reshape(-1,6)
            R1 = mu[:, 0:3] / torch.norm(mu[:,0:3], dim = -1, keepdim=True)
            R3 = torch.cross(R1, mu[:, 3:], dim=-1)
            R3 = R3 / torch.norm(R3, dim = -1, keepdim=True)
            R2 = torch.cross(R3, R1, dim = -1)

            rotation_matrix = torch.stack([R1,R2,R3],dim = -1)

            quat = lietorch.SO3(rotation_matrix,from_rotation_matrix=True).vec()
            quat = quat.reshape(batchsize,-1,4)

            output_proj = torch.cat([output_proj[:,:,:10],quat,scale],dim=-1)
            """
            so3 output: xyz, rgb, scale, opacity, quat, scale
            """
            
        elif Mode.cholesky in self.mode:
            """
            diagonale von L exponentiell machen, damit sie sicher positiv ist
            """
            output_proj[:,:,7:10] = torch.exp(output_proj[:,:,7:10])

        return output_proj, out_cache