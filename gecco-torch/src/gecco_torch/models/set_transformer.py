"""
Implements the Set Transformer from https://arxiv.org/abs/1810.00825 with
some modifications to "inject" the diffusion noise level into the network.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange

from gecco_torch.models.mlp import MLP
from gecco_torch.models.normalization import AdaGN


class AttentionPool(nn.Module):
    """
    Uses attention to pool a set of features into a smaller set of features.
    The queries are defined by the `inducers` learnable parameter.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_inducers: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert feature_dim % num_heads == 0, (feature_dim, num_heads)
        dims_per_head = feature_dim // num_heads

        self.inducers = nn.Parameter(
            torch.randn(
                1,
                num_heads,
                num_inducers,
                dims_per_head,
            )
        )

        self.kv_proj = nn.Linear(feature_dim, feature_dim * 2, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)

        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.dims_per_head = dims_per_head
        self.num_inducers = num_inducers

    def forward(self, kv: Tensor,mask_points: Tensor = None,) -> Tensor:
        # print(f"Attention pool forward, kv na: {torch.isnan(kv).any()}")
        k_heads, v_heads = rearrange(
            self.kv_proj(kv),
            "b n (t h d) -> t b h n d",
            t=2, # t, h, d kommen im rearange vor
            h=self.num_heads,
            d=self.dims_per_head,
        )

        # inducer wurden im set transformer erfunden, damit es linear statt quadratisch skaliert, queries werden dadurch ersetzt
        queries = self.inducers.repeat(kv.shape[0], 1, 1, 1)

        # baue maske auf 
        if mask_points is not None:
            mask_attn_pool = torch.ones(self.num_heads, self.num_inducers,4000,dtype=torch.bool,device=queries.device)
            for i in range(mask_points.shape[0]):
                mask_attn_pool[:, mask_points[i]:, :] = False
        else:
            mask_attn_pool = None

        attn = F.scaled_dot_product_attention(
            queries,
            k_heads,
            v_heads,
            attn_mask=mask_attn_pool,
        )  # (batch_size, num_heads, num_inducers, feature_dim // num_heads)

        attn = rearrange(attn, "b h i d -> b i (h d)")
        attn = self.out_proj(attn)
        # print(f"scaled dot prod attn na: {torch.isnan(attn).any()}")
        return attn


class Broadcast(nn.Module):
    """
    A module that implements a pool -> mlp -> unpool sequence to return an updated
    version of the input tokens.
    """

    def __init__(
        self,
        feature_dim: int,
        num_inducers: int,
        t_embed_dim: int,
        num_heads: int = 8,
        mlp_blowup: int = 2,
        activation: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pool = AttentionPool(feature_dim, num_heads, num_inducers)
        self.norm_1 = AdaGN(feature_dim, t_embed_dim)
        # MLP(in_features,out_features,width,depth,activation)
        mlp_depth = kwargs.get("mlp_depth", 1)
        self.mlp = MLP(
            feature_dim, feature_dim, mlp_blowup * feature_dim, depth = mlp_depth, activation = activation
        )
        self.norm_2 = AdaGN(feature_dim, t_embed_dim)
        self.unpool = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True) 
        """
        trotzdem linear complexity, weil wir ne fixe Anzahl queries im Attention pool haben. Dadurch ist der output auch von der Dimension her fix, 
        weil der output von F.scaled_dot_product_attention von der Anzahl der Queries abhängt und nicht der Anzahl der keys, values. 
        """

    def forward(
        self,
        x: Tensor,
        t_embed: Tensor,
        return_h: bool = False,
        h: Tensor | None = None,
        mask_points: Tensor = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            `x` - the input features
            `t_embed` - the diffusion noise level
            `return_h` - whether to return the inducer states for efficient point cloud upsampling
            `h` - the inducer states if using cached values
        """
        if h is None:
            # h is always none, wenn man nicht upsampled
            h = self.pool(x,mask_points)
            h = self.norm_1(h, t_embed)
            h = self.mlp(h) # nach diesem schritt war noch kein nan (wissen wir weil da die activation noch nicht nan war)
            h = self.norm_2(h, t_embed)
        
        # num_points = mask_multihead.shape[1]
        # num_keyvalues = mask_multihead.shape[2]
        # expanded_mask = mask_multihead.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # expanded_mask = expanded_mask.reshape(2 * self.num_heads, num_points, num_keyvalues)
        # print(expanded_mask)
        
        # andere heangehensweise, einfach die padded points wegnehmen pro batch
        # reversed_mask = mask_multihead.flip(dims=[1]).long()
        
        # wenn man keine maske braucht, dann gibt der 0 zurück, weil es keine true position gibt
        # first_true_positions = reversed_mask.argmax(dim=1)


        # num_points_to_add_dynamic = first_true_positions[:, 0]

    

        attn, _weights = self.unpool(x, h, h, need_weights=False,attn_mask=None)
        # hinterher auf 0 setzen, weil die keine rolle spielen
        if mask_points is not None:
            for i in range(attn.shape[0]):
            #     # nur wenns nicht 0 ist, weil nur dann gibt es überhaupt ne maske
                attn[i, mask_points[i]:, :] = 0
        # attn = [attn[i, :num_points_to_add_dynamic[i], :] for i in range(attn.shape[0])]


        # print(f"attn na: {torch.isnan(attn)}")
        # print(f"multihead attention attn na: {torch.isnan(attn).any()}")

        if return_h:
            return attn, h
        else:
            return attn, None


class BroadcastingLayer(nn.Module):
    """
    A module equivalent to a standard transformer layer which uses a
    broadcast -> mlp sequence, with skip connections in a pre-norm fashion.
    """

    def __init__(
        self,
        feature_dim: int,
        num_inducers: int,
        embed_dim: int,
        num_heads: int = 8,
        mlp_blowup: int = 2,
        activation: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.broadcast_norm = AdaGN(feature_dim, embed_dim)
        self.broadcast = Broadcast(
            feature_dim,
            num_inducers,
            embed_dim,
            num_heads,
            mlp_blowup=mlp_blowup,
            activation=activation,
        )
        self.mlp_norm = AdaGN(feature_dim, embed_dim)
        mlp_depth = kwargs.get("mlp_depth", 1)
        self.mlp = MLP(
            feature_dim, feature_dim, mlp_blowup * feature_dim, depth = mlp_depth, activation=activation
        )

        with torch.no_grad():
            # scale down the skip connection weights
            self.broadcast.unpool.out_proj.weight *= 0.1
            self.mlp[-1].weight *= 0.1

    def forward(
        self,
        x: Tensor,
        t_embed: Tensor,
        return_h: bool = False,
        h: Tensor | None = None,
        mask_points: Tensor = None,
    ) -> tuple[Tensor, Tensor | None]:
        # print(f"input broadcasting layer na: {torch.isnan(x).any()}")
        y = self.broadcast_norm(x, t_embed)
        x_b, h = self.broadcast(y, t_embed, return_h, h, mask_points)
        # skip connection
        x = x + x_b

        y = self.mlp_norm(x, t_embed)
        # skip connection
        x = x + self.mlp(y)

        return x, h


class SetTransformer(nn.Module):
    """
    A set transformer is just a sequence of broadcasting layers.
    """

    def __init__(
        self,
        n_layers: int,
        feature_dim: int,
        num_inducers: int,
        t_embed_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BroadcastingLayer(
                    feature_dim=feature_dim,
                    num_inducers=num_inducers,
                    embed_dim=t_embed_dim,
                    **kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.feature_dim = feature_dim

    def forward(
        self,
        features: Tensor,
        t_embed: Tensor,
        return_h: bool = False,
        hs: list[Tensor] | None = None,
        mask_points: Tensor = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        # print("In SetTransformer, forward")
        if hs is None: # make it fit
            hs = [None] * len(self.layers)
        else:
            pass
            # print(f"hs na: {[torch.isnan(h).any() for h in hs]}")
        # print(f"features na: {torch.isnan(features).any()}")
        # print(f"t_embed na: {torch.isnan(t_embed).any()}")
        # if not hs is None:
        #     print(f"hs na: {[torch.isnan(h).any() for h in hs]}")
        stored_h = []
        for i,(layer, h) in enumerate(zip(self.layers, hs)):
            # print(f"shape features layer {i}: {features.shape}")
            features, h = layer(features, t_embed, return_h=return_h, h=h, mask_points=mask_points)
            # print(f"features layer {i} na: {torch.isnan(features).any()}")
            stored_h.append(h)

        if return_h:
            return features, stored_h
        else:
            return features, None

if __name__ == "__main__":
    pass