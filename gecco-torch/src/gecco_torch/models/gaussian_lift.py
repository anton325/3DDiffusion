from typing import Any
from torch import nn, Tensor
from gecco_torch.structs import Mode
import torch
import lietorch
from gecco_torch.models.set_transformer import SetTransformer


class LinearLift(nn.Module):
    """
    Embeds the 3d geometry (xyz points) in a higher dimensional space, passes it through
    the SetTransformer, and then maps it back to 3d. "Lift" refers to the embedding action.
    This class is used in the unconditional ShapeNet experiments.
    """

    def __init__(
        self,
        inner: SetTransformer,
        feature_dim: int,
        geometry_dim: int = 14,
        mode = [Mode.normal],
        do_norm: bool = True, # layer norm
    ):
        super().__init__()
        self.mode = mode
        if Mode.so3_diffusion in self.mode:
            self.lift = nn.Linear(geometry_dim + 4, feature_dim)
            self.scale_layer = nn.Linear(6, 1) # aus dem mu die scale berechnen
        elif Mode.procrustes in self.mode:
            self.lift = nn.Linear(geometry_dim + 9, feature_dim)
        else:
            self.lift = nn.Linear(geometry_dim, feature_dim)
        self.inner = inner

        if do_norm:
            if Mode.so3_diffusion in self.mode:
                self.lower = nn.Sequential(
                    nn.LayerNorm(feature_dim, elementwise_affine=False),
                    nn.Linear(feature_dim, geometry_dim + 6),
                )
            elif Mode.procrustes in self.mode:
                self.lower = nn.Sequential(
                    nn.LayerNorm(feature_dim, elementwise_affine=False),
                    nn.Linear(feature_dim, geometry_dim + 9),
                )
            else:
                self.lower = nn.Sequential(
                    nn.LayerNorm(feature_dim, elementwise_affine=False),
                    nn.Linear(feature_dim, geometry_dim),
                )
        else:
            if Mode.so3_diffusion in self.mode:
                self.lower = nn.Linear(feature_dim, geometry_dim + 6)
            else:
                self.lower = nn.Linear(feature_dim, geometry_dim)

    def forward(
        self,
        geometry: Tensor,
        embed: Tensor,
        raw_context: Any,
        post_context: Any,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        del raw_context, post_context  # unused

        features = self.lift(geometry)
        features, out_cache = self.inner(features, embed, do_cache, cache)
        output_proj = self.lower(features)

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

            quat = lietorch.SO3(rotation_matrix, from_rotation_matrix=True).vec()
            quat = quat.reshape(batchsize,-1,4)

            output_proj = torch.cat([output_proj[:,:,:10],quat,scale],dim=-1)
            """
            so3 output: xyz, rgb, scale, opacity, quat, scale
            """

        return output_proj, out_cache
