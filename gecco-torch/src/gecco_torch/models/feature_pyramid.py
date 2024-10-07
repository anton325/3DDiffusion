"""
Wraps a pretrained feature extractor to produce a feature pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn, Tensor
import torchvision.transforms as T
import torchvision.models as tvm

from gecco_torch.structs import GaussianContext3d


@dataclass
class FeaturePyramidContext:
    features: list[Tensor]
    K: Tensor


class FeaturePyramidExtractor(nn.Module):
    def forward(self, ctx_raw: GaussianContext3d) -> FeaturePyramidContext:
        raise NotImplementedError()


class ConvNeXtExtractor(FeaturePyramidExtractor):
    def __init__(
        self,
        n_stages: int = 3,
        model: Literal["tiny", "small"] = "tiny",
        pretrained: bool = True,
    ):
        super().__init__()

        if model == "tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            convnext = tvm.convnext_tiny(weights=weights)
        elif model == "small":
            weights = tvm.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            convnext = tvm.convnext_small(weights=weights)
        else:
            raise ValueError(f"Unknown model {model}")

        self.stages = nn.ModuleList()
        for i in range(0, len(convnext.features), 2):
            # group together each downsampling + processing stage
            self.stages.append(
                nn.Sequential(convnext.features[i], convnext.features[i + 1])
            )

        self.stages = self.stages[:n_stages]
        self._remove_stochastic_depth()

    def _remove_stochastic_depth(self):
        """We found SD to harm generative performance"""
        for submodule in self.modules():
            if isinstance(submodule, tvm.convnext.CNBlock):
                submodule.stochastic_depth = torch.nn.Identity()

    def forward(self, raw_ctx: GaussianContext3d) -> FeaturePyramidContext:
        images = raw_ctx.image

        features = []
        for stage in self.stages:
            # immer die zwei stages die direkt hintereinander liegen ausführen, und extrahierten features in die liste einfügen
            images = stage(images) # shapes: 1: (batch, 96, 100,100), 2: (batch, 192, 50,50), 3: (batch, 384, 25,25)
            features.append(images)

        return FeaturePyramidContext(
            features=features,
            K=raw_ctx.K, # K is camera matrix
        )
    

class DinoV2Extractor(FeaturePyramidExtractor):
    def __init__(
        self,
        model: Literal["small","base"] = "small",
    ):
        super().__init__()

        if model == "small":
            self.dinov2_vit14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif model == "base":
            self.dinov2_vit14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        else:
            raise ValueError(f"Unknown model {model}")
    
        self.transform = T.Compose([
                        T.Resize(392),
                        ])

    def forward(self, raw_ctx: GaussianContext3d) -> FeaturePyramidContext:
        images = raw_ctx.image
        images = self.transform(images)

        latent_feature_map = self.dinov2_vit14(images, return_patches=True)
        latent_feature_map_patchy = latent_feature_map.reshape(-1, 384, 28, 28) # 768 ist die number of features

        return FeaturePyramidContext(
            features = [latent_feature_map_patchy],
            K=raw_ctx.K, # K is camera matrix
        )

if __name__ == "__main__":
    d = DinoV2Extractor("small")
