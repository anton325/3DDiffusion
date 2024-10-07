from kornia.geometry.camera.perspective import project_points    
from gecco_torch.structs import GaussianContext3d, Mode
from torch import Tensor
import torch
from einops import rearrange


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
        
        if Mode.visibile_filter in self.mode:
            img,depth = self.render_fn(geometry_data.clone(),ctx,self.mode, return_depth = True)

        if Mode.in_world_space in self.mode:
            # wenns im world space war dann muss es fürs projizieren in den camera space
            for b in range(ctx.w2c.shape[0]):
                # :3 sind immer xyz, unabhängig von der genauen Modellierung
                geometry_data[b,:,:3] = torch.einsum("ab,nb->na", ctx.w2c[b,:3, :3], geometry_data[b,:,:3]) + ctx.w2c[b,:3, -1] # adding translation part
            # geometry_data[:,:,:3] = torch.einsum('bij,bnj->bni', ctx.w2c[:, :3, :3], geometry_data[:, :, :3])
            # translation = ctx.w2c[:, :3, -1].unsqueeze(1)  # Reshape to [batch_size, 1, 3] for broadcasting
            # geometry_data[b,:,:3] = geometry_data[b,:,:3] + translation

        hw_01 = project_points(geometry_data[:,:,:3], ctx.K.unsqueeze(1))[..., :2]

        hw_flat = rearrange(hw_01, "b n t -> b n 1 t")
        
        lookups = []
        # print("Perform lookups in feature map")
        # einen wert zu geben
        for feature in features:
            # print("shape feature ", feature.shape) # shape 1x (batch, 96, 34, 34), 1x (batch, 192, 17, 17), 1x (batch, 384,8,8)
            # print(f"shape grid {(hw_flat * 2 - 1).shape}")
            lookup = torch.nn.functional.grid_sample(
                feature, hw_flat * 2 - 1, align_corners=False
            )

            # grid sample findet die feature werte (input) an den im grid geforderten stellen heraus
            # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
            # mit padding mode zero werden zeros ausgegeben für grid werte, die außerhalb von -1,1 liegen
            # hw_flat gibt ja die location an und zwar sozusagen relativ gesehen. Wenn feature die shape batch, 96, 100, 100 hat, 
            # dann ist die location (0,0) in hw_flat das center des der, also (50,50) in feature

            # print("shape lookup before rearanging", lookup.shape)
            lookup = rearrange(lookup, "b c n 1 -> b n c")

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
