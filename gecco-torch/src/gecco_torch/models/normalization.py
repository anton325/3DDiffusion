"""
Adapts the standard normalization layers to take in a time/noise level embedding.
"""
import torch
from torch import Tensor, nn
from einops import rearrange


class AdaNorm(nn.Module):
    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        raise NotImplementedError()


class AdaGN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        ctx_dim: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            affine=False,
        )
        self.bias = nn.Linear(ctx_dim, num_channels)
        self.scale = nn.Linear(ctx_dim, num_channels)

        with torch.no_grad():
            self.bias.weight.fill_(0.0)
            self.bias.bias.fill_(0.0)
            self.scale.weight.fill_(0.0)
            self.scale.bias.fill_(1.0)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        """
        ctx ist nicht der normale context den wir vom example kennen, sondern t_embed
        """
        # print(f"Forward AdaGN x na: {torch.isnan(x).any()}")
        x_bcn = rearrange(x, "b n c -> b c n")
        normed_bcn = self.gn(x_bcn)
        # print(f"normed bcn na: {torch.isnan(normed_bcn).any()}")
        normed = rearrange(normed_bcn, "b c n -> b n c")
        bias = self.bias(ctx)
        # print(f"bias na: {torch.isnan(bias).any()}")
        scale = self.scale(ctx)
        # print(f"scale na: {torch.isnan(scale).any()}")

        shape = (x.shape[0], *((1,) * (x.ndim - 2)), x.shape[-1])
        normed = scale.reshape(shape) * normed + bias.reshape(shape)
        # print(f"normed na: {torch.isnan(normed).any()}")
        return normed
