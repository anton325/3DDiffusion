"""
Definitions of the diffusion model itself, along with preconditioning, loss and sampling functions.
"""

from __future__ import annotations

import datetime
import pathlib
import math
from typing import Any, Sequence
import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn, Tensor
from tqdm.auto import tqdm

from gecco_torch.reparam import Reparam, NoReparam
from gecco_torch.structs import Example, Context3d

from gecco_torch.benchmark_jax import BenchmarkCallback


def ones(n: int):
    return (1,) * n


class EDMPrecond(nn.Module): # preconditioning to improve convergence?
    """
    Preconditioning module wrapping a diffusion backbone. Implements the logic proposed in
    "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_data=1.0,
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        raw_context: Any,  # raw_context comes from the dataset, before any preprocessing
        post_context: Any,  # post_context comes from the conditioner
        do_cache: bool = False,  # whether to return a cache of inducer states for upsampling
        cache: list[Tensor] | None = None,  # cache of inducer states for upsampling
    ) -> tuple[Tensor, list[Tensor] | None]:  # denoised, optional cache
        sigma = sigma.reshape(-1, *ones(x.ndim - 1))

        # sind das alles nur die preconditioning calculations -> scale inputs according to the 
        # "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al. 
        # p. 3 section "any"
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        # print(f"c_skip: {c_skip}")
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        # print(f"c_out: {c_out}")
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        # print(f"c_in: {c_in}")
        c_noise = sigma.log() / 4
        # print(f"c_noise: {c_noise}")
        c_noise = c_noise

        F_x, cache = self.model(
            (c_in * x), c_noise, raw_context, post_context, do_cache, cache
        )
        denoised = c_skip * x + c_out * F_x
        # print(f"denoised: {denoised}")
        # print(f"denoised na: {denoised.isnan().any()}")

        if not do_cache:
            return denoised
        else:
            return denoised, cache


class LogNormalSchedule(nn.Module):
    """
    LogNormal noise schedule as proposed in "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    # da sagen sie: 
    sigma min 0.002, max 80,

    """

    def __init__(self, sigma_max: float, mean=-1.2, std=1.2):
        super().__init__()

        self.sigma_max = sigma_max
        self.mean = mean
        self.std = std

    def extra_repr(self) -> str:
        return f"sigma_max={self.sigma_max}, mean={self.mean}, std={self.std}"

    def forward(self, data: Tensor) -> Tensor:
        rnd_normal = torch.randn(
            [data.shape[0], *ones(data.ndim - 1)], device=data.device
        )
        return (rnd_normal * self.P_std + self.P_mean).exp()


class LogUniformSchedule(nn.Module):
    """
    LogUniform noise schedule which seems to work better in our (GECCO) context.

    alle schedules returnen einfach nur für jedes n ein sigma, 
    sie werden gecalled mit schedule(samples) und samples hat shape (batchsize, num_points, 3)
    und dann gibt er für jedes element im batch ein sigma
    """

    def __init__(self, max: float, min: float = 0.002, low_discrepancy: bool = True):
        super().__init__()

        self.sigma_min = min
        self.sigma_max = max
        self.log_sigma_min = math.log(min)
        self.log_sigma_max = math.log(max)
        self.low_discrepancy = low_discrepancy

    def extra_repr(self) -> str:
        return f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, low_discrepancy={self.low_discrepancy}"

    def forward(self, data: Tensor) -> Tensor:
        u = torch.rand(data.shape[0], device=data.device)

        if self.low_discrepancy:
            div = 1 / data.shape[0]
            u = div * u
            u = u + div * torch.arange(data.shape[0], device=data.device)

        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma.reshape(-1, *ones(data.ndim - 1))


class EDMLoss(nn.Module):
    """
    A loss function for training diffusion models. Implements the loss proposed in
    "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(
        self, schedule: nn.Module, sigma_data: float = 1.0, loss_scale: float = 100.0 # nur schedule wird vorgegeben, rest die standardvalues
    ):
        super().__init__()

        self.schedule = schedule
        self.sigma_data = sigma_data
        self.loss_scale = loss_scale

    def extra_repr(self) -> str:
        return f"sigma_data={self.sigma_data}, loss_scale={self.loss_scale}"

    def forward(self, net: Diffusion, examples: torch.Tensor, context: Context3d) -> Tensor:
        """
        Im dataloader stellen wir eigentlich immer batches vom typ Example bereit, aber irgendwie
        baut der das auseinander, dass hier examples=Example.data und context=Example.ctx ankommt??
        """
        # print("In edm loss foward")
        # print(f"examples na: {examples.data.isnan().any()}")
        # print(type(examples)) # torch tensor
        # print(type(context)) # context 3d
        # print(f"context examples: {context.image.shape}") # (batchsize,1,3,3)
        ex_diff = net.reparam.data_to_diffusion(examples, context) # reparametrisierte Punktwolke
        # print(f"forward shape ex diff: {ex_diff.shape}") # (batchsize,num points, 3)
        sigma = self.schedule(ex_diff)
        # print(f"Sigma: {sigma}")
        # print(f"forward shape sigma: {sigma.shape}") # (batchsize,1,1)
        weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)
        # print(f"Weight: {weight}")
        # print(f"weight na: {weight.isnan().any()}")

        # calculate the noise
        n = torch.randn_like(ex_diff) * sigma

        # print(f"forward shape noise: {n.shape}") # (batchsize,num points, 3)
        # predict the original data
        """In-place operations in forward von net ändern ex_diff nicht, da durch die addition ein neuer tensor erstellt wird"""
        D_yn = net(ex_diff + n, sigma, context) # input ist pointcloud, die mit noise verändert wurde, und das sigma
        # print(f"forward shape pred: {D_yn.shape}") # (batchsize,num points, 3)

        """fürs bilder machen kein noise auf die pc geben"""
        # D_yn = net(ex_diff, sigma, context) # input ist pointcloud, die mit noise verändert wurde, und das sigma

        # compare the original data with the denoised data
        loss = self.loss_scale * weight * ((D_yn - ex_diff) ** 2) # wegen preconditioning mehr stability?
        # print(f"loss na: {loss.isnan().any()}")
        mean_loss = loss.mean()
        # print(mean_loss)
        return mean_loss


class Conditioner(nn.Module):
    """
    An abstract class for a conditioner. Conditioners are used to preprocess the context 
    -> # conditioner is ConvNeXt and gets features from image and appends them to corresponding points in point cloud
    before it is passed to the diffusion backbone.

    NOT TO BE CONFUSED with preconditioning the diffusion model itself (done by EDMPrecond). # preconditioning is improving convergence properties of network
    """

    def forward(self, raw_context):
        raise NotImplementedError()


class IdleConditioner(Conditioner):
    """
    A placeholder conditioner that does nothing, for unconditional models.
    """

    def forward(self, raw_context: Context3d | None) -> None:
        del raw_context
        return None


class Diffusion(pl.LightningModule):
    """
    The main diffusion model. It consists of a backbone, a conditioner, a loss function
    and a reparameterization scheme.

    It derives from PyTorch Lightning's LightningModule, so it can be trained with PyTorch Lightning trainers.
    """

    def __init__(
        self,
        backbone: nn.Module,
        conditioner: Conditioner,
        loss: EDMLoss,
        reparam: Reparam = NoReparam(dim=3), # no reparam for ShapeNet?
        save_path_benchmark: str = None,
        start_epoch: int = 0,
    ):
        super().__init__()

        self.backbone = backbone
        self.conditioner = conditioner
        self.loss = loss
        self.reparam = reparam

        self.validation_steps_count = 0
        self.validation_epochs_count = 0
        self.step_train_count = 0
        self.start_epoch = start_epoch
        
        self.collected_validation_batches = []

        self.validation_samples = []
        self.save_path_benchmark = save_path_benchmark #pathlib.Path(pathlib.Path.home(),"Documents","gecco","gecco-torch",f'meshes_sampled_{str(datetime.datetime.now()).replace(" ","")}')
        
        # set default sampler kwargs
        self.sampler_kwargs = dict(
            num_steps=64,
            sigma_min=0.002,
            sigma_max=self.sigma_max,
            rho=7,
            S_churn=0.5,
            S_min=0,
            S_max=float("inf"),
            S_noise=1,
            with_pbar=False,
        )

    def extra_repr(self) -> str:
        return str(self.sampler_kwargs)

    @property
    def sigma_max(self) -> float:
        return self.loss.schedule.sigma_max

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4) # lr 1e-4

    def training_step(self, batch: Example, batch_idx):
        # print(f"training step: start in epoch {self.start_epoch}")
        x, ctx = batch
        loss = self.loss(
            self,
            x,
            ctx,
        )
        
        self.log("train_loss", loss, on_step=True)
        # self.logger.experiment.add_scalar("training_loss", scalar_value=loss, global_step=self.step_train_count)
        self.step_train_count += 1
        return loss

    def on_validation_start(self):
        print("validation start")
        self.validation_samples = []

    def validation_step(self, batch: Example, batch_idx):
        print("validation step")
        x, ctx = batch
        self.batch_size = x.shape[0]
        loss = self.loss(
            self,
            x,
            ctx,
        )
        self.log("val_loss", loss,on_step=True)
        # self.logger.experiment.add_scalar("validation_loss", scalar_value=loss, global_step=self.validation_steps_count)
        self.validation_steps_count += 1
        print(f"val step shape batch data: {batch.data.shape}")
        print(f"val step type batch: {type(batch)}")
        # if self.validation_epochs_count == 0:
        self.collected_validation_batches.append((batch.data,batch.ctx))

    def on_validation_epoch_end(self):
        # if self.validation_epochs_count == 0:
            # all_val_batches = torch.cat(self.collected_validation_batches,dim=0)
            # bei conditional generation nur L1 distance als metrik
            # self.benchmark = BenchmarkCallback(all_val_batches.cpu().numpy())
        self.benchmark = BenchmarkCallback(self.collected_validation_batches,
                                           batch_size=self.collected_validation_batches[0][0].data.shape[0],
                                           save_path=self.save_path_benchmark)
        
        print("validation epoch end")   
        print(f"validation steps count: {self.validation_steps_count}")
        print(f"validation epochs count: {self.validation_epochs_count}")
        print(f"validation batches: {len(self.collected_validation_batches)}")
        self.collected_validation_batches = []
        self.benchmark(self,self.logger,self.log,self.validation_epochs_count+self.start_epoch)
        self.validation_epochs_count += 1

    def on_train_epoch_end(self):
        # This method is called at the end of each epoch
        # You can perform any custom actions or computations here
        print(f"Epoch {self.current_epoch} has ended")

    def forward(
        self,
        data: Tensor,
        sigma: Tensor,
        raw_context: Any | None,
        post_context: Any | None = None,
        do_cache: bool = False,
        cache: Any | None = None,
    ) -> Tensor:
        """
        Applies the denoising network to the given data, with the given noise level and context.
        """
        if post_context is None:
            post_context = self.conditioner(raw_context)
        return self.backbone(data, sigma, raw_context, post_context, do_cache, cache)

    @property
    def example_param(self) -> Tensor:
        return next(self.parameters())


#------------------------------------------------------------ SAMPLING ---------------------------------------------------------------------- # 


    def t_steps(
        self, num_steps: int, sigma_max: float, sigma_min: float, rho: float
    ) -> Tensor:
        """
        Returns an array of sampling time steps for the given parameters.
        """
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=self.example_param.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
        return t_steps

    @torch.no_grad()
    def sample_stochastic(
        self,
        shape: Sequence[int], # for example (48, 2048, 3), # sample 48 times a pointcloud consisting of 2048 points (3D)
        context: Context3d | None, 
        rng: torch.Generator = None,
        **kwargs,
    ) -> Tensor:
        """
        SAMPLE ARCHAIC
        """
        # print(f"sample: shape context: {context.image.shape}")
        # print(f"shape shape: {shape}")
        """
        A stochastic sampling function that samples from the diffusion model with the given context. Corresponds to
        the `SDE` sampler in the paper. The `ODE` sampler is not currently implemented in PyTorch.
        """
        kwargs = {**self.sampler_kwargs, **kwargs}
        num_steps = kwargs["num_steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]
        rho = kwargs["rho"]
        S_churn = kwargs["S_churn"]
        S_min = kwargs["S_min"]
        S_max = kwargs["S_max"]
        S_noise = kwargs["S_noise"]
        with_pbar = kwargs["with_pbar"]

        device = self.example_param.device
        dtype = self.example_param.dtype
        if rng is None:
            rng = torch.Generator(device).manual_seed(42)

        B = shape[0] # batch size
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1
        # latents is in shape of desired output (batchsize,points in pointcloud, 3)


        # context is features of the reference image -> generate pointcloud for that image
        # only used once - as input to SetTransformer
        post_context = self.conditioner(context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
        # print(f"t steps in sampling(): {t_steps.shape}")
        reverse_ode = kwargs.get("reverse", False)
        if reverse_ode:
            t_steps = torch.flip(t_steps, [0])[1:]
            gt_data = kwargs["gt_data"]
            latents = self.reparam.data_to_diffusion(gt_data, context)
            x_next = latents.to(torch.float64)
        else:
            # Main sampling loop.
            x_next = latents.to(torch.float64) * t_steps[0]
        
        ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
        # ts ist liste von tuplen -> t,t+1

        if with_pbar:
            ts = tqdm(ts, unit="step")

        for i, (t_cur, t_next) in ts:  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.

            # Euler step.
            denoised = self(
                x_cur.to(dtype), # tensor.to creates a new tensor, so in place changes there dont affect the original x_hat
                t_cur.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            if i < num_steps - 1:
                noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * t_cur
                x_next = denoised + noise
            else:
                x_next = denoised

        if with_pbar:
            ts.close()
        if not reverse_ode:
            output = self.reparam.diffusion_to_data(x_next, context)
        else: 
            output = x_next
        return output
    
    def sample_stochastic_originial(
        self,
        shape: Sequence[int], # for example (48, 2048, 3), # sample 48 times a pointcloud consisting of 2048 points (3D)
        context: Context3d | None, 
        rng: torch.Generator = None,
        **kwargs,
    ) -> Tensor:
        # print(f"sample: shape context: {context.image.shape}")
        # print(f"shape shape: {shape}")
        """
        A stochastic sampling function that samples from the diffusion model with the given context. Corresponds to
        the `SDE` sampler in the paper. The `ODE` sampler is not currently implemented in PyTorch.
        """
        kwargs = {**self.sampler_kwargs, **kwargs}
        num_steps = kwargs["num_steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]
        rho = kwargs["rho"]
        S_churn = kwargs["S_churn"]
        S_min = kwargs["S_min"]
        S_max = kwargs["S_max"]
        S_noise = kwargs["S_noise"]
        with_pbar = kwargs["with_pbar"]

        device = self.example_param.device
        dtype = self.example_param.dtype
        if rng is None:
            rng = torch.Generator(device).manual_seed(42)

        B = shape[0] # batch size
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1
        # latents is in shape of desired output (batchsize,points in pointcloud, 3)


        # context is features of the reference image -> generate pointcloud for that image
        # only used once - as input to SetTransformer
        post_context = self.conditioner(context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
        # print(f"t steps in sampling(): {t_steps.shape}")
        reverse_ode = kwargs.get("reverse", False)
        if reverse_ode:
            t_steps = torch.flip(t_steps, [0])[1:]
            gt_data = kwargs["gt_data"]
            latents = self.reparam.data_to_diffusion(gt_data, context)
            x_next = latents.to(torch.float64)
        else:
            # Main sampling loop.
            x_next = latents.to(torch.float64) * t_steps[0]
        
        ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
        # ts ist liste von tuplen -> t,t+1

        if with_pbar:
            ts = tqdm(ts, unit="step")

        for i, (t_cur, t_next) in ts:  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, math.sqrt(2.0) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * noise

            # Euler step.
            denoised = self(
                x_hat.to(dtype), # tensor.to creates a new tensor, so in place changes there dont affect the original x_hat
                t_hat.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self(
                    x_next.to(dtype),
                    t_next.repeat(B).to(dtype),
                    context,
                    post_context,
                ).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if with_pbar:
            ts.close()
        if not reverse_ode:
            output = self.reparam.diffusion_to_data(x_next, context)
        else: 
            output = x_next
        return output


#------------------------------------------------------ UPSAMPLING ------------------------------------#
    @torch.no_grad()
    def upsample(
        self,
        data: Tensor,
        new_latents: Tensor | None = None,
        n_new: int | None = None,
        context: Context3d | None = None,
        seed: int | None = 42,
        num_substeps=5,
        **kwargs,
    ):
        """
        An upsampling function that upsamples the given data to the given number of new points.

        Args:
            `data` - the data to be upsampled
            `new_latents` or `n_new` - either the specific latent variables to use for upsampling
                if the user wants to control them, or the number of new points to generate.
            `context` - the context to condition the upsampling with

        Returns:
            The newly sampled points.
        """
        kwargs = {**self.sampler_kwargs, **kwargs}
        num_steps = kwargs["num_steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]
        rho = kwargs["rho"]
        S_churn = kwargs["S_churn"]
        S_min = kwargs["S_min"]
        S_max = kwargs["S_max"]
        S_noise = kwargs["S_noise"]
        with_pbar = kwargs["with_pbar"]

        net_dtype: torch.dtype = self.example_param.dtype
        net_device: torch.device = self.example_param.device

        rng: torch.Generator = torch.Generator(device=net_device)
        if seed is not None:
            rng: torch.Generator = rng.manual_seed(seed)
        randn = lambda shape: torch.randn(
            shape, device=net_device, dtype=net_dtype, generator=rng
        )

        if (new_latents is None) == (n_new is None):
            raise ValueError(
                "Either new_latents or n_new must be specified, but not both."
            )
        if new_latents is None:
            new_latents = randn((data.shape[0], n_new, data.shape[2]))
        assert isinstance(new_latents, Tensor)

        data = self.reparam.data_to_diffusion(data, context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

        def call_net_cached(x: Tensor, t: Tensor, ctx: Context3d, cache: list[Tensor]):
            """
            A helper function that calls the diffusion backbone with the given inputs and cache.
            """
            return self(
                x.to(net_dtype),
                t.to(net_dtype).expand(x.shape[0]),
                ctx,
                do_cache=False,
                cache=cache,
            ).to(torch.float64)

        x_next = new_latents.to(torch.float64) * t_steps[0]

        steps = enumerate(zip(t_steps[:-1], t_steps[1:]))
        if with_pbar:
            steps = tqdm(steps, total=t_steps.shape[0] - 1)
        # Main sampling loop.
        for i, (t_cur, t_next) in steps:  # 0, ..., N-1
            data_ctx = data + randn(data.shape) * t_cur
            _, cache = self(
                data_ctx.to(net_dtype),
                t_cur.to(net_dtype).expand(data_ctx.shape[0]),
                context,
                do_cache=True,
                cache=None,
            )
            for u in range(num_substeps):
                x_cur = x_next

                # Increase noise temporarily.
                gamma = (
                    min(S_churn / num_steps, math.sqrt(2) - 1)
                    if S_min <= t_cur <= S_max
                    else 0
                )
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn(
                    x_cur.shape
                )

                # Euler step.
                denoised = call_net_cached(x_hat, t_hat, context, cache)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < num_steps - 1:
                    denoised = call_net_cached(x_next, t_next, context, cache)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                if u < num_substeps - 1 and i < num_steps - 1:
                    redo_noise = (t_cur**2 - t_next**2).sqrt()
                    x_next = x_next + redo_noise * randn(x_next.shape)

        if with_pbar:
            steps.close()

        return self.reparam.diffusion_to_data(x_next, context)
