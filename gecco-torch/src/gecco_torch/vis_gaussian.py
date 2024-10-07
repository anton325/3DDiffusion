from typing import Any
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import numpy as np
import pathlib
import datetime
import wandb
from gecco_torch.diffusion import Diffusion
from gecco_torch.structs import GaussianExample,GaussianContext3d, Camera, Mode
from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.utils.loss_utils import l1_loss
from gecco_torch.additional_metrics.metrics_so3 import minimum_distance, rotational_distance_between_pairs_dot_product
from gecco_torch.utils.riemannian_helper_functions import geodesic_distance
# from gecco_torch.utils.isotropic_plotting import visualize_so3_probabilities
# import jaxlie
# import jax.numpy as jnp
# from PIL import Image
# from io import BytesIO

from gecco_torch.utils.render_tensors import render_fn_options


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

    def __init__(self, 
                 render_fn : callable, 
                 mode,
                 n: int = 8,
                 n_steps: int = 64, 
                 point_size: int = 0.1,
                 visualize_save_folder:str = None,
                 start_epoch=0,
                 unconditional_bool = False):
        super().__init__()
        
        self.n = n
        self.splatting_camera_choice = 0
        self.n_steps = n_steps
        self.mode = mode
        self.point_size = point_size
        self.batch: GaussianExample | None = None
        self.unconditional_bool = unconditional_bool
        if visualize_save_folder is None:
            raise Exception("visualize_save_folder must be set")
        self.gaussian_scene_folder_id = pathlib.Path(visualize_save_folder,"meshes") #str(datetime.datetime.now()).replace(" ","")
        self.gaussian_scene_folder_id.mkdir(parents=True, exist_ok=True)

        if Mode.rotational_distance in mode or Mode.cholesky_distance in mode:
            self.rotations_save_folder_id = pathlib.Path(visualize_save_folder,"rotations") #str(datetime.datetime.now()).replace(" ","")
            self.rotations_save_folder_id.mkdir(parents=True, exist_ok=True)
        self.start_epoch = start_epoch
        self.render_fn = render_fn
    
    def visualize_special_feature(self, mode, special_feature_gt, special_feature = None):
        special_feature_distances = []
        log_images = []
        if special_feature is None:
            chosen_feature = special_feature_gt
        else:
            chosen_feature = special_feature

        for i in range(chosen_feature.shape[0]):

            # visualize_so3_probabilities(jnp.array([jaxlie.SO3(x.detach().cpu().numpy()).as_matrix() for x in chosen_rotations[i]]), 0.001)
            # plt.savefig(f"images/rotations_.png")
            # Convert Matplotlib figure to a PNG image in memory
            # buf = BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)
            # Optionally convert to a PIL image if you need to manipulate the image further
            # image = Image.open(buf)

            if special_feature is not None:
                if Mode.rotational_distance in mode:
                    print("Calc rotational distance...")
                    feature_distance = minimum_distance(special_feature[i], special_feature_gt[i], rotational_distance_between_pairs_dot_product)
                    log_name = 'vis_rotational_distance'

                elif Mode.cholesky_distance in mode:
                    print("Calc geodesic distance...")
                    feature_distance = minimum_distance(special_feature[i], special_feature_gt[i], geodesic_distance)
                    log_name = 'vis_geodesic_distance'

                special_feature_distances.append(feature_distance)
                # log_images.append(wandb.Image(image, caption=f"rt: {rot_dis:0.1f}"))
            # else:
            #     log_images.append(wandb.Image(image))
        
        if len(special_feature_distances) > 0:
            wandb.log({log_name: float(sum(special_feature_distances)/len(special_feature_distances))})
            # wandb.log({"Rotations_sampled": log_images})

    def save_rotations(self, rotations, gt_rotations, epoch):
        folder_rotations = pathlib.Path(self.rotations_save_folder_id, f"epoch_{epoch}")
        folder_rotations.mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path(folder_rotations,"gt_rotations.npz"), "wb") as f:
            np.savez(f, rotations=gt_rotations.cpu().numpy())
        with open(pathlib.Path(folder_rotations,"rotations.npz"), "wb") as f:
            np.savez(f, rotations=rotations.cpu().numpy())

    def images_to_wandb(self,samples, gt_data, images, images_gt, images_ctx, images_gt_ctx, pl_module):
        images_wandb,images_gt_wandb, rendered_images_ctx_wandb, images_ctx_gt_wandb = [],[],[],[]
        if self.unconditional_bool:
            gt_data = samples
        for i,(gt_pc,pc) in enumerate(zip(gt_data, samples)):
            if not self.unconditional_bool:
                clouds = np.concatenate((gt_pc.cpu().numpy()[:,:3], pc.cpu().numpy()[:,:3]), axis=0)
            else:
                clouds = pc.cpu().numpy()[:,:3]
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

            if not self.unconditional_bool:
                # we have context -> conditional, no ground truth to show
                pl_module.logger.experiment.log({
                    f"val_pc_{i}": [wandb.Object3D({"type": "lidar/beta",'points':points_rgb})],
                },
                )

            pl_module.logger.experiment.log({
                f"val_pc_sampled_{i}": [wandb.Object3D({"type": "lidar/beta",'points':points_sampled_rgb})],
            },
            )
            if not self.unconditional_bool:
                image_gt = images_gt[i] # render_dict['render']
                # import torchvision
                # torchvision.utils.save_image(rendered_image,f"temp_{i}.png")

                image_gt = image_gt.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                image_gt = wandb.Image(image_gt, caption=f"{self.batch.ctx.insinfo.instance[i][:8]}_gt_{i}")
                images_gt_wandb.append(image_gt)

            rendered_image = images[i]
            if not self.unconditional_bool:
                loss = l1_loss(rendered_image, images_gt[i])
            else:
                loss = torch.tensor(0.0)
            rendered_image = rendered_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            rendered_image = wandb.Image(rendered_image, caption=f"rend_{i}_loss_{loss.cpu().item():0.4f}")
            images_wandb.append(rendered_image)
            
            if not self.unconditional_bool:
                rendered_image_ctx = images_ctx[i] # render_dict['render']
                loss = l1_loss(rendered_image_ctx, self.batch.ctx.image[i])
                # rendered_image = rendered_image.permute(1, 2, 0)
                # rendered_image = (255*rendered_image).type(torch.uint8)
                # rendered_image = wandb.Image(rendered_image.cpu().numpy(), caption=f"rend_{i}_loss_{loss.cpu().item():0.4f}")
                
                rendered_image_ctx = rendered_image_ctx.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                rendered_image_ctx = wandb.Image(rendered_image_ctx, caption=f"rend_{i}_loss_{loss.cpu().item():0.4f}")
                rendered_images_ctx_wandb.append(rendered_image_ctx)

                image_gt_ctx = images_gt_ctx[i]
                image_gt_ctx = image_gt_ctx.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                image_gt_ctx = wandb.Image(image_gt_ctx, caption=f"gt_{i}")
                images_ctx_gt_wandb.append(image_gt_ctx)

        wandb.log({f"images_rendered" : images_wandb})
        if not self.unconditional_bool:
            wandb.log({f"images_gt" : images_gt_wandb})
            wandb.log({"images_ctx_rendered" : rendered_images_ctx_wandb})
            wandb.log({"images_ctx_gt" : images_ctx_gt_wandb})

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Diffusion,
        outputs: Any,
        batch: GaussianExample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        # nur beim ende des ersten validation batch
        if batch_idx != 0:
            return

        # ONLY DO THIS ONCE
        if self.batch is None:
            # cache the first batch for reproducible visualization
            # and display the context images (only once)
            self.batch = batch.apply_to_tensors(lambda t: t[: self.n].clone())
            # examples = []
            # for i in range(3):
            #     pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
            #     image = wandb.Image(pixels, caption=f"random field {i}")
            #     examples.append(image)
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
            if Mode.cholesky in self.mode:
                sample_shape = (self.batch.data.shape[0], self.batch.data.shape[1], 13)
            else:
                sample_shape = self.batch.data.shape

            # example_tensor = torch.ones_like(self.batch.data)#.cuda()
            # example_image = torch.ones_like(self.batch.ctx.image)#.cuda()
            # for c in range(example_image.shape[0]):
            #     self.batch.ctx.image[c] = example_image[c]
            # # example = self.example_to_cuda(example)
            # sigma_example = torch.ones((3,1,1))# .cuda()
            # example_output = pl_module(example_tensor, sigma_example, self.batch.ctx, do_cache=False, cache=None)
            with torch.no_grad():
                samples = pl_module.sample_stochastic(
                    shape=sample_shape,
                    context=self.batch.ctx,
                    sigma_max=pl_module.sigma_max,
                    num_steps=self.n_steps,
                )
            # print("Saving model to model_checkpoint.pth ...")
            # torch.save({
            #     'model_state_dict': pl_module.state_dict(),
            # }, 'model_checkpoint.pth')
            # print("append validation samples")
            # pl_module.validation_samples.append(samples.cpu().numpy())

        if self.unconditional_bool:
            # no point showing "ground truth" for unconditional generation
            vertices = samples
            colors = None
            images_dict = self.render_fn(samples,self.batch.ctx, self.mode, self.splatting_camera_choice)
            images = images_dict['render']
            images_gt_ctx = None
            gt_data = None
            images_gt = None
            images_ctx = None
        else:
            # concatenate context and samples for visualization
            # distinguish them by color
            vertices = torch.cat([self.batch.data[:,:,:3], samples[:,:,:3]], dim=1)

            colors = torch.zeros(
                *vertices.shape, device=vertices.device, dtype=torch.uint8
            )
            colors[:, : self.batch.data.shape[1], 1] = 255  # green for ground truth
            colors[:, self.batch.data.shape[1] :, 0] = 255  # red for samples
            if torch.isnan(samples).any():
                print("NA in samples, dont render")
            else:
                print("PCVCallback: Save gaussian scene tensor")
                # pathlib.Path(pathlib.Path.home(),"Documents","gecco",f"meshes_{self.gaussian_scene_folder_id}",f"epoch_{trainer.current_epoch}").mkdir(parents=True, exist_ok=True)
                # with open(pathlib.Path(pathlib.Path.home(),"Documents","gecco",f"meshes_{self.gaussian_scene_folder_id}",f"epoch_{trainer.current_epoch}","vertices.npz"), "wb") as f:
                #     np.savez(f, vertices=vertices.cpu().numpy())
                with open(pathlib.Path(self.gaussian_scene_folder_id,f"verticies_{self.start_epoch+trainer.current_epoch}.npz"), "wb") as f:
                    np.savez(f, vertices=samples.cpu().numpy())
                # print(self.batch.data)
                with open(pathlib.Path(self.gaussian_scene_folder_id,f"gt_verticies_{self.start_epoch+trainer.current_epoch}.npz"), "wb") as f:
                    np.savez(f, vertices=self.batch.data.cpu().numpy())

                if bool(self.batch.ctx):
                    gt_data = self.batch.data
                else:
                    gt_data = samples
                with torch.no_grad():
                    self.splatting_camera_choice = np.random.randint(0, len(self.batch.ctx.splatting_cameras))
                    print("Render gt")
                    kwargs = {}
                    if Mode.rotational_distance in self.mode:
                        kwargs['plot_rotations'] = True
                    elif Mode.cholesky_distance in self.mode:
                        kwargs['cholesky_distance'] = True
                    images_gt_dict = self.render_fn(gt_data.clone(),self.batch.ctx, self.mode, self.splatting_camera_choice, **kwargs)
                    if Mode.rotational_distance in self.mode:
                        special_feature_gt = images_gt_dict['rotations']
                        self.visualize_special_feature(self.mode, special_feature_gt = special_feature_gt)
                    elif Mode.cholesky_distance in self.mode:
                        special_feature_gt = images_gt_dict['Ls']
                        self.visualize_special_feature(self.mode, special_feature_gt = special_feature_gt)

                    images_gt = self.batch.ctx.splatting_cameras[self.splatting_camera_choice][1]
                    images_gt_ctx = self.batch.ctx.image

                    print("Render samples")
                    images_dict = self.render_fn(samples,self.batch.ctx, self.mode, self.splatting_camera_choice, **kwargs)
                    images = images_dict['render']
                    if Mode.rotational_distance in self.mode:
                        special_feature = images_dict['rotations']
                        self.visualize_special_feature(self.mode, special_feature = special_feature,special_feature_gt = special_feature_gt)
                        self.save_rotations(special_feature, special_feature_gt, self.start_epoch+trainer.current_epoch)
                    elif Mode.cholesky_distance in self.mode:
                        special_feature = images_dict['Ls']
                        self.visualize_special_feature(self.mode, special_feature = special_feature,special_feature_gt = special_feature_gt)
                        self.save_rotations(special_feature, special_feature_gt, self.start_epoch+trainer.current_epoch)

                    print("Render samples from ctx")
                    images_ctx = self.render_fn(samples, self.batch.ctx, self.mode, None, **kwargs)['render']

        self.images_to_wandb(samples, gt_data, images, images_gt, images_ctx, images_gt_ctx, pl_module)
        # print(self.batch.data)