"""
Definitions of the diffusion model itself, along with preconditioning, loss and sampling functions.
"""

from __future__ import annotations
import os
import math
from typing import Any, Sequence
import numpy as np
import torch
import lightning.pytorch as pl
from torch import nn, Tensor
from tqdm.auto import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from collections import deque
from typing import List
from pathlib import Path
import roma
from lightning.pytorch.utilities import grad_norm
import wandb

from gecco_torch.reparam import Reparam, NoReparam
from gecco_torch.structs import Example, GaussianContext3d, Mode
from gecco_torch.train_forward import splatting_loss, vanilla, noise_lie_rotations, cholesky_L, so3, so3_x0
from gecco_torch.benchmark_jax_splat import BenchmarkCallback

class Saver:
    """
    Save format:
    save_path / saver_checkpoints / epoch_{epoch}_lpips_{lpips}.pth
    """
    def __init__(self, save_path, unconditional):
        self.save_path = Path(save_path / "saver_checkpoints")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_lpips = 100
        self.best_epoch = 100
        self.unconditional = unconditional

    def save(self, model, epoch, lpips = None):
        print("Check if saving necessary...")
        if self.check_is_better(lpips):
            # lösche das alte
            if self.best_epoch != 100:
                print(f"Remove {self.save_path / f'epoch_{self.best_epoch}_lpips_{self.best_lpips}.pth'}")
                os.remove(self.save_path / f"epoch_{self.best_epoch}_lpips_{self.best_lpips}.pth")
            self.best_epoch = epoch
            self.best_lpips = lpips
            print(f'save to {self.save_path / f"epoch_{epoch}_lpips_{lpips}.pth"}')
            torch.save(model.state_dict(), self.save_path / f"epoch_{epoch}_lpips_{lpips}.pth")
    
    def check_is_better(self, lpips):
        if lpips is None:
            return True
        if lpips < self.best_lpips:
            print(f"{lpips} better than {self.best_lpips}, saving...")
            return True
        print(f"{lpips} worse than {self.best_lpips}, dont save")
        return False

def ones(n: int):
    return (1,) * n

class NoEDMPrecond(nn.Module):
    def __init__(
            self,
            model):
        self.model = model
    
    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        raw_context: Any,  # raw_context comes from the dataset, before any preprocessing
        post_context: Any,  # post_context comes from the conditioner
        do_cache: bool = False,  # whether to return a cache of inducer states for upsampling
        cache: list[Tensor] | None = None,  # cache of inducer states for upsampling
    ):
        denoised, cache = self.model(x, sigma, raw_context, post_context, do_cache, cache)
        
        if not do_cache:
            return_dict = {
                'denoised' : denoised,
                }
        else:
            return_dict = {
            'denoised' : denoised,
            'cache' : cache
                }
        return return_dict
    
class NoEDMPrecond(nn.Module):
    def __init__(
            self,
            model):
        self.model = model
    
    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        raw_context: Any,  # raw_context comes from the dataset, before any preprocessing
        post_context: Any,  # post_context comes from the conditioner
        do_cache: bool = False,  # whether to return a cache of inducer states for upsampling
        cache: list[Tensor] | None = None,  # cache of inducer states for upsampling
    ):
        denoised, cache = self.model(x, sigma, raw_context, post_context, do_cache, cache)
        
        if not do_cache:
            return_dict = {
                'denoised' : denoised,
                }
        else:
            return_dict = {
            'denoised' : denoised,
            'cache' : cache
                }
        return return_dict

class EDMPrecond(nn.Module): # preconditioning to improve convergence?
    """
    Preconditioning module wrapping a diffusion backbone. Implements the logic proposed in
    "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Mode,
        sigma_data=1.0,
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.mode = mode

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        raw_context: Any,  # raw_context comes from the dataset, before any preprocessing
        post_context: Any,  # post_context comes from the conditioner
        do_cache: bool = False,  # whether to return a cache of inducer states for upsampling
        cache: list[Tensor] | None = None,  # cache of inducer states for upsampling
    ) -> tuple[Tensor, list[Tensor] | None]:  # denoised, optional cache
        if Mode.so3_diffusion in self.mode:
            return_dict = so3.forward_preconditioning(self, x, sigma, raw_context, post_context, do_cache, cache)
        elif Mode.so3_x0 in self.mode:
            return_dict = so3_x0.forward_preconditioning(self, x, sigma, raw_context, post_context, do_cache, cache)
        elif Mode.cholesky in self.mode:
            return_dict = cholesky_L.forward_preconditioning(self, x, sigma, raw_context, post_context, do_cache, cache)
        else:
            return_dict = vanilla.forward_preconditioning(self, x, sigma, raw_context, post_context, do_cache, cache)
            # return_dict = vanilla.forward_preconditioning_not_rotation(self, x, sigma, raw_context, post_context, do_cache, cache)

        if not do_cache:
            return return_dict['denoised']
        else:
            return return_dict['denoised'], return_dict['cache']

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
        u = torch.rand(data.shape[0], device=data.device) # uniform 0-1

        if self.low_discrepancy:
            div = 1 / data.shape[0]
            u = div * u
            u = u + div * torch.arange(data.shape[0], device=data.device)

        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma.reshape(-1, *ones(data.ndim - 1))
    
    def return_schedule(self,n):
        u = torch.linspace(0,1,n).cuda()
        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma


class EDMLoss(nn.Module):
    """
    A loss function for training diffusion models. Implements the loss proposed in
    "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al.
    """

    def __init__(
        self, 
        schedule: nn.Module, 
        mode: List[Mode], 
        splatting_loss,
        sigma_data: float = 1.0,
        loss_scale: float = 100.0, # nur schedule wird vorgegeben, rest die standardvalues
    ):
        super().__init__()

        self.schedule = schedule
        self.sigma_data = sigma_data
        self.loss_scale = loss_scale
        self.mode = mode
        self.splatting_loss = splatting_loss

        self.running_queue = deque()
        self.max_size = 50
        #self.sum -= self.queue.popleft()


    def extra_repr(self) -> str:
        return f"sigma_data={self.sigma_data}, loss_scale={self.loss_scale}"

    def forward(self, net: Diffusion, examples: torch.tensor, context: GaussianContext3d, phase: str, train_step,log_fun, render_fun) -> Tensor:
        if Mode.rotation_matrix_mode in self.mode:
            data, loss, sigma = noise_lie_rotations.forward(net,examples,context, log_fun)
        elif Mode.so3_diffusion in self.mode:
            data, loss, sigma = so3.forward(self, net, examples, context, train_step, log_fun)
        elif Mode.so3_x0 in self.mode:
            data, loss, sigma = so3_x0.forward(self, net, examples, context, train_step, log_fun)
        elif Mode.cholesky in self.mode:
            data, loss, sigma = cholesky_L.forward(self, net, examples, context, train_step, log_fun)
        else:
            data, loss, sigma = vanilla.forward(self, net, examples, context, log_fun, train_step)

        if Mode.splatting_loss in self.mode and self.splatting_loss['starting_step'] <= train_step:
            mean_splatting_losses = splatting_loss.splatting_loss(self, train_step, phase, render_fun, data, context, log_fun, sigma)
            if mean_splatting_losses is not None:
                if torch.isnan(mean_splatting_losses):
                    print("NA IN SPLATTING LOSS")
                splatting_loss_adjusted = self.splatting_loss['lambda'] * mean_splatting_losses
                if torch.isnan(splatting_loss_adjusted):
                    print("NA IN SPLATTING LOSS ADJUSTED")

                if Mode.only_splatting_loss in self.mode:
                    combined_loss = splatting_loss_adjusted
                else:
                    combined_loss = loss + splatting_loss_adjusted
            else:
                print("splatting_loss is None")
                if Mode.only_splatting_loss in self.mode:
                    return None
                combined_loss = loss
        else:
            combined_loss = loss

        if torch.isnan(combined_loss):
            print("NA IN LOSS")
        log_fun("Combined_loss",combined_loss,on_step=True)
        # torch.save({
        #     'model_state_dict': net.state_dict(),
        # }, 'model_checkpoint.pth')
        # with torch.no_grad():
        #     example_tensor = torch.ones_like(examples.data)
        #     example_image = torch.ones_like(context.image)
        #     for c in range(example_image.shape[0]):
        #         context.image[c] = example_image[c]
        #     sigma_example = torch.ones_like(sigma)
        #     example_output = net(example_tensor, sigma_example, context, do_cache=False, cache=None)
        return combined_loss

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

    def forward(self, raw_context: GaussianContext3d | None) -> None:
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
        render_fn: callable,
        mode: Mode,
        reparam: Reparam = NoReparam(dim=3), # no reparam for ShapeNet?
        save_path_benchmark: str = None,
        start_epoch: int = 0,
        unconditional_bool: dict = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.conditioner = conditioner
        self.loss = loss
        self.reparam = reparam
        self.mode = mode

        self.validation_steps_count = 0
        self.validation_epochs_count = 0
        self.step_train_count = 0
        self.start_epoch = start_epoch
        self.render_fn = render_fn
        
        self.collected_validation_batches = []

        self.validation_samples = []
        self.save_path_benchmark = save_path_benchmark #pathlib.Path(pathlib.Path.home(),"Documents","gecco","gecco-torch",f'meshes_sampled_{str(datetime.datetime.now()).replace(" ","")}')
        self.saver = Saver(save_path_benchmark, unconditional_bool)
        self.unconditional = unconditional_bool
        
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
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch: Example, batch_idx):
        # print(f"training step: start in epoch {self.start_epoch}")
        x, ctx = batch
        loss = self.loss(
            self,
            x,
            ctx,
            "train",
            self.step_train_count,
            self.log,
            self.render_fn,
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
        loss = self.loss(
            self,
            x,
            ctx,
            "val",
            self.validation_steps_count,
            self.log,
            self.render_fn,
        )
        self.log("val_loss", loss,on_step=True)
        # self.logger.experiment.add_scalar("validation_loss", scalar_value=loss, global_step=self.validation_steps_count)
        self.validation_steps_count += 1
        # print(f"val step shape batch data: {batch.data.shape}")
        # print(f"val step type batch: {type(batch)}")
        # if self.validation_epochs_count == 0:
        self.collected_validation_batches.append((batch.data,batch.ctx))

    def on_validation_epoch_end(self):
        # if self.validation_epochs_count == 0:
        #     all_val_batches = torch.cat(self.collected_validation_batches,dim=0)
        #     bei conditional generation nur L1 distance als metrik
        #     self.benchmark = BenchmarkCallback(all_val_batches.cpu().numpy())
        if not self.unconditional:
            self.benchmark = BenchmarkCallback(list_data_and_ctx=self.collected_validation_batches,
                                            epoch=self.validation_epochs_count+self.start_epoch,
                                            batch_size=self.collected_validation_batches[0][0].data.shape[0],
                                            save_path=self.save_path_benchmark,
                                            render_fn = self.render_fn,
                                            mode=self.mode,
                                            )
            
            print("validation epoch end")   
            print(f"validation steps count: {self.validation_steps_count}")
            print(f"validation epochs count: {self.validation_epochs_count}")
            print(f"validation batches: {len(self.collected_validation_batches)}")
            self.collected_validation_batches = []
            mean_lpips = self.benchmark(self,
                        self.logger,
                        self.log,
                        self.validation_epochs_count+self.start_epoch,)
        else:
            # das zeichen für den saver, das neueste zu saven
            mean_lpips = None

        self.validation_epochs_count += 1
        self.saver.save(self, self.validation_epochs_count+self.start_epoch, mean_lpips)
        pass

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if Mode.log_grads in self.mode:
            norms = grad_norm(self, norm_type=2)
            # self.log_dict(norms)
            norms_for_logging = {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in norms.items()}
            wandb.log(norms_for_logging)

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
        res = self.backbone(data, sigma, raw_context, post_context, do_cache, cache)

        if Mode.procrustes in self.mode: 
            with torch.autocast(device_type="cuda", enabled=False): # enabled False sagt kein autocast
                # map rotations to closest actual rotation matrix
                rotations = res[:,:,10:].view(res.shape[0],-1,3,3)
                # wir brauchen special_procrustes weil das returned eine rotation matrix, procrustes returns orthonormal matrix
                changed_type = False
                if rotations.dtype == torch.float16:
                    rotations = rotations.type(torch.float32)
                    changed_type = True
                map_rotations = roma.special_procrustes(rotations)
                if changed_type:
                    map_rotations = map_rotations.type(torch.float16)
                

                # man muss die aufteilung machen (oder clone) weil res[:,:,9:18] = map_rotations.view(res.shape[0],-1,9) ist eine
                # in place operation die backprop in roma procrustes kaputt macht
                part1 = res[:, :, :10]
                # Create the new middle part from 'map_rotations'
                new_middle = map_rotations.view(res.shape[0], -1, 9)

                # Concatenate all parts along the third dimension
                new_res = torch.cat([part1, new_middle], dim=2)

            return new_res 
        else:
            return res

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
        context: GaussianContext3d | None, 
        rng: torch.Generator = None,
        **kwargs,
    ) -> Tensor:
        """
        A stochastic sampling function that samples from the diffusion model with the given context. Corresponds to
        the `SDE` sampler in the paper. The `ODE` sampler is not currently implemented in PyTorch.
        """
        print("Saving model...")
    # torch.save({
    #     'model_state_dict': self.state_dict(),
    # }, 'model_checkpoint.pth')
        if Mode.so3_diffusion in self.mode:
            data = so3.sample_1step(self, shape, context, rng, **kwargs)
        elif Mode.so3_x0 in self.mode:
            data = so3_x0.sample(self, shape, context, rng, **kwargs)
        elif Mode.cholesky in self.mode:
            data = cholesky_L.sample_logdirection_tangent(self, shape, context, rng, **kwargs)
        else:
            data = vanilla.sample(self, shape, context, rng, **kwargs)

        return data


#------------------------------------------------------ UPSAMPLING ------------------------------------#
    @torch.no_grad()
    def upsample(
        self,
        data: Tensor,
        new_latents: Tensor | None = None,
        n_new: int | None = None,
        context: GaussianContext3d | None = None,
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

        def call_net_cached(x: Tensor, t: Tensor, ctx: GaussianContext3d, cache: list[Tensor]):
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
