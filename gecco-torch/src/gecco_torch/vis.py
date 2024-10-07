from typing import Any
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import numpy as np
import pathlib
import datetime
import wandb
from gecco_torch.diffusion import Diffusion
from gecco_torch.structs import Example


def plot_3d(clouds, colors=["blue", "red", "green"], shared_ax=False, images=None):
    assert len(clouds) <= len(colors)
    if images is not None:
        assert len(images) == len(clouds)
        assert not shared_ax

    width = 1 if shared_ax else len(clouds)
    height = 1 if images is None else 2
    figsize = (5 * width, 5 * height)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    def init_ax(ax):
        y_max = max(cloud[:, 1].max().detach().cpu().numpy() for cloud in clouds)
        y_min = min(cloud[:, 1].min().detach().cpu().numpy() for cloud in clouds)
        ax.view_init(azim=0, elev=0)
        ax.set_zlim(y_max, y_min)
        ax.set_aspect("equal")

    if shared_ax:
        ax = fig.add_subplot(projection="3d")
        init_ax(ax)

    for i, (cloud, color) in enumerate(zip(clouds, colors)):
        if not shared_ax:
            ax = fig.add_subplot(height, width, i + 1, projection="3d")
            init_ax(ax)

        x, y, z = cloud.detach().cpu().numpy().T
        ax.scatter(z, x, y, s=0.1, color=color)

    if images is not None:
        for i, image in enumerate(images):
            ax = fig.add_subplot(height, width, width + i + 1)
            ax.imshow(image.detach().cpu().numpy().transpose(1, 2, 0))
            ax.axis("off")

    return fig


class PCVisCallback(pl.Callback):
    """
    A callback which visualizes two things
        1. The context images (only once)
        2. The ground truth and sampled point clouds (once per validation phase)
    """

    def __init__(self, n: int = 8, n_steps: int = 64, point_size: int = 0.1,mesh_save_folder_id:str = None,start_epoch=0):
        super().__init__()
        self.n = n
        self.n_steps = n_steps
        self.point_size = point_size
        self.batch: Example | None = None
        if mesh_save_folder_id is None:
            raise Exception("mesh_save_folder_id must be set")
        self.mesh_save_folder_id = pathlib.Path(mesh_save_folder_id,"meshes") #str(datetime.datetime.now()).replace(" ","")
        self.mesh_save_folder_id.mkdir(parents=True, exist_ok=True)
        self.start_epoch = start_epoch

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Diffusion,
        outputs: Any,
        batch: Example,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        # nur beim ende des ersten validation batch
        if batch_idx != 0:
            return

        if self.batch is None:
            # cache the first batch for reproducible visualization
            # and display the context images (only once)
            self.batch = batch.apply_to_tensors(lambda t: t[: self.n].clone())
            examples = []
            for i in range(3):
                pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
                image = wandb.Image(pixels, caption=f"random field {i}")
                examples.append(image)
            if bool(self.batch.ctx): # context
                images = []
                for i, image in enumerate(self.batch.ctx.image):
                    # print(type(pl_module.logger))
                    # print(type(pl_module.logger.experiment))
                    # pl_module.logger.experiment.add_image(
                    #     tag=f"val/context_image_{i}",
                    #     img_tensor=image,
                    #     global_step=trainer.current_epoch,
                    #     dataformats="CHW",
                    # )
                    image = image.permute(1, 2, 0)
                    image = wandb.Image(image.cpu().numpy(), caption=f"img_{i}")
                    images.append(image)
                wandb.log({f"context images": images})

        with torch.random.fork_rng(), torch.no_grad():
            # set seed to start with same gaussian noise
            torch.manual_seed(42)
            if not bool(self.batch.ctx):
                print(f"in vis sample from model with shape {self.batch.data.shape} and context None")
            else:
                print(f"in vis sample from model with shape {self.batch.data.shape} and context {self.batch.ctx.image.shape}")

            # SAMPLE FROM NOISE
            samples = pl_module.sample_stochastic(
                shape=self.batch.data.shape,
                context=self.batch.ctx,
                sigma_max=pl_module.sigma_max,
                num_steps=self.n_steps,
            )
            # print("append validation samples")
            # pl_module.validation_samples.append(samples.cpu().numpy())

        if not bool(self.batch.ctx):
            # no point showing "ground truth" for unconditional generation
            vertices = samples
            colors = None
        else:
            # concatenate context and samples for visualization
            # distinguish them by color
            vertices = torch.cat([self.batch.data, samples], dim=1)

            colors = torch.zeros(
                *vertices.shape, device=vertices.device, dtype=torch.uint8
            )
            colors[:, : self.batch.data.shape[1], 1] = 255  # green for ground truth
            colors[:, self.batch.data.shape[1] :, 0] = 255  # red for samples

        print("Add mesh")
        # pathlib.Path(pathlib.Path.home(),"Documents","gecco",f"meshes_{self.mesh_save_folder_id}",f"epoch_{trainer.current_epoch}").mkdir(parents=True, exist_ok=True)
        # with open(pathlib.Path(pathlib.Path.home(),"Documents","gecco",f"meshes_{self.mesh_save_folder_id}",f"epoch_{trainer.current_epoch}","vertices.npz"), "wb") as f:
        #     np.savez(f, vertices=vertices.cpu().numpy())
        with open(pathlib.Path(self.mesh_save_folder_id,f"verticies_{self.start_epoch+trainer.current_epoch}.npz"), "wb") as f:
            np.savez(f, vertices=samples.cpu().numpy())
        with open(pathlib.Path(self.mesh_save_folder_id,f"gt_verticies_{self.start_epoch+trainer.current_epoch}.npz"), "wb") as f:
            np.savez(f, vertices=self.batch.data.cpu().numpy())
        # pl_module.logger.experiment.add_mesh(
        #     tag=f"val/samples",
        #     vertices=vertices,
        #     colors=colors,
        #     global_step=trainer.current_epoch,
        #     # config_dict={
        #     #     "material": {
        #     #         "cls": "PointsMaterial",
        #     #         "size": self.point_size,
        #     #     },
        #     # },
            
        # )
            if bool(self.batch.ctx):
                gt_data = self.batch.data
            else:
                gt_data = samples
            for i,(gt_pc,pc) in enumerate(zip(gt_data, samples)):
                clouds = np.concatenate((gt_pc.cpu().numpy(), pc.cpu().numpy()), axis=0)
                colors = np.zeros((clouds.shape[0], 3))
                # color the gt greenf
                colors[:int(len(colors)/2), 1] = 255 # rgb
                
                # color the samples red
                colors[int(len(colors)/2):, 0] = 255
                colors[int(len(colors)/2):, 1] = 0

                """
                Das funktioniert für alle arten von pointclouds mit shape (x,y) solange y>3 und die xyz positionen 
                in y[0],y[1],y[2] stehen
                """
                # nur die gesampled punktwolke
                points_sampled_rgb = np.array([[p[0], p[1], p[2], 255,0,0] for p in pc.cpu().numpy()])

                # gesampled und gt punktwolke zusammen
                points_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in zip(clouds, colors)])

                if bool(self.batch.ctx):
                    # we have context -> conditional, no ground truth to show
                    pl_module.logger.experiment.log({
                        f"val_pc_{i}": [wandb.Object3D({"type": "lidar/beta",'points':points_rgb})],
                    },
                    )

                pl_module.logger.experiment.log({
                    f"val_pc_sampled_{i}": [wandb.Object3D({"type": "lidar/beta",'points':points_sampled_rgb})],
                },
                )

