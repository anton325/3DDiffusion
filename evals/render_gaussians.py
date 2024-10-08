import torch
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import io
import json
import cv2
from piqa import SSIM
import lpips
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import argparse


from gecco_torch.data.gaussian_pc_dataset import Gaussian_pc_DataModule
from LookAtPoseSampler import LookAtPoseSampler
from utils import plot_heatmaps, plot_heatlines, plot_polar_heatmaps
from gecco_torch.gaussian_renderer import render
from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.utils.render_tensors import ctx_to_cuda, make_camera
from gecco_torch.structs import GaussianContext3d, Camera, Mode, GaussianExample
from gecco_torch.utils.riemannian_helper_functions import  L_to_scale_rotation
from gecco_torch.train_forward import cholesky_L, so3_x0, so3, vanilla
from gecco_torch.models.load_model import load_model_saver_checkpoints
from gecco_torch.utils.render_gaussian_circle import get_cameras
from torchvision.utils import save_image
from torchvision.io import read_image
from gecco_torch.utils.loss_utils import l1_loss
from gecco_torch.utils.image_utils import psnr
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde

from gecco_torch import benchmark_jax_splat
from sklearn.decomposition import PCA
from All_Evals import AllEvals

def plot_pca(data, dest):
    data_np = data.cpu().numpy()
    pca = PCA(n_components=2)

    # Fit and transform the data to get the principal components
    principal_components = pca.fit_transform(data_np[0])

    x = principal_components[:, 0]
    y = principal_components[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Create grid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(xx, yy, f, cmap='viridis', edgecolor='none')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Density')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Density')
    plt.title('3D Density Plot of PCA Components')
    plt.savefig(dest / "pca2d_density.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of the Dataset')
    plt.grid(True)
    plt.savefig(dest / "pca2d.png")

    # Initialize PCA for 3 components
    pca = PCA(n_components=3)

    # Fit and transform the data
    principal_components = pca.fit_transform(data_np[0])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], alpha=0.5)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA of the Dataset')

    plt.savefig(dest / "pca3d.png")

    x = principal_components[:, 0]
    y = principal_components[:, 1]
    z = principal_components[:, 2]
    values = np.vstack([x, y, z])
    kernel = gaussian_kde(values)
    densities = kernel(values)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(x, y, z, c=densities, cmap='viridis', marker='o')

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Density')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('3D Scatter Plot of PCA Components with Density Coloring')
    plt.savefig(dest / "pca3d_density.png")

    # Original data
    x = principal_components[:, 0]
    y = principal_components[:, 1]
    z = principal_components[:, 2]

    # Sampling data: randomly select half of the indices
    indices = np.random.choice(x.shape[0], size=x.shape[0] // 4, replace=False)
    x_sampled = x[indices]
    y_sampled = y[indices]
    z_sampled = z[indices]

    # Calculate densities for sampled data
    values_sampled = np.vstack([x_sampled, y_sampled, z_sampled])
    kernel = gaussian_kde(values_sampled)
    densities_sampled = kernel(values_sampled)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with sampled data
    scatter = ax.scatter(x_sampled, y_sampled, z_sampled, c=densities_sampled, cmap='viridis', marker='o')

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Density')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('3D Scatter Plot of PCA Components with Density Coloring')
    plt.savefig(dest / "pca3d_density_downsampled.png")

class GaussianGeccoEvaluator:
    def __init__(self, 
                 version,
                 variant,
                 ) -> None:
        self.version = version
        self.variant = variant

        self.camera_split = "test"

        self.lpips_fn_net = lpips.LPIPS(net = 'alex').cuda()
        self.ssim_fn = SSIM().cuda()

        # self.model, self.render_fn, self.data, self.mode, self.reparam, self.epoch = load_model(version,epoch)
        self.model, self.render_fn, self.data, self.mode, self.reparam, self.epoch, self.unconditional_bool = load_model_saver_checkpoints(version, pathlib.Path.cwd())
        # self.mode.append(Mode.gecco_projection)
        
        self.model = self.model.cuda().eval()
        if self.unconditional_bool:
            self.batch_size = 10
        else:
            self.batch_size = 10
        self.all_categories = True
        self.number_of_steps = 128
            
        self.setup_out_dir(self.variant)
        self.data_module = self.setup_datamodule()
        self.load_srn_cams()
        self.number_of_angles = 360
        self.number_of_angles_autoregressive = 36
        self.load_cameras()
        self.all_evals = AllEvals(self.output_dir, self.batch_size)
        self.circle_cams,_,_ = get_cameras()
        self.sample_and_render()

    def setup_datamodule(self):
        print("Data module setup...")
        dataset_path = Path("/globalwork/giese/gaussians")
        self.dataset_images = Path("/globalwork/giese/shapenet_rendered")
        if self.all_categories:
            restrict_to_categories = None
        else:
            restrict_to_categories = ['02958343']
            # restrict_to_categories = ['03001627']
        data_module = Gaussian_pc_DataModule(
            dataset_path,
            self.dataset_images,
            group = "good",
            epoch_size = 10000, # 5_000 , # 10k for 500k steps
            batch_size=self.batch_size, #  original conditional: A100 GPU, 40GB, 2048 points, batch 48, SDE sampler with steps
            num_workers=8,
            val_size = 100000, # mit None wird über das ganze dataset iteriert, das bedeutet für jede pc kriegen wir 50 samples, weil von jedem kamera view einmal
            single_example = False,
            worker_init_fn = None,
            worker_zipfile_instances = {},
            mode = self.mode,
            restrict_to_categories = restrict_to_categories, # ['02958343'], # 02958343 car # 02691156 plane
            include_test_idx=-1, # 1 wenn die nicht inkludiert sein sollen
            exclude_test_pics=True, #
        )
        # data_module.setup()
        data_module.setup(stage="test")
        return data_module

    def get_train_dataloader(self):
        train_dataloader = self.data_module.train_dataloader()
        return train_dataloader

    def get_val_dataloader(self):
        val_dataloader = self.data_module.val_dataloader()
        return val_dataloader
    
    def get_test_dataloader(self):
        test_dataloader = self.data_module.test_dataloader()
        return test_dataloader
    
    def setup_out_dir(self, variant):
        print("Set up output directory...")
        if variant == "":
            self.output_dir = Path(f"/globalwork/giese/evals/gen_gaussians_eval/{self.version}/epoch_{self.epoch}")
        else:
            self.output_dir = Path(f"/globalwork/giese/evals/gen_gaussians_eval/{self.version}_{variant}/epoch_{self.epoch}")
        self.output_dir.mkdir(parents=True,exist_ok=True)
    
    def load_cameras(self):
        print("Load cameras...")
        root_cameras = Path("/home/giese/Documents/gaussian-splatting/cameras")
        with open(Path(root_cameras,"train_cameras.json"),"r") as f:
            self.train_cams = json.load(f)
        with open(Path(root_cameras,"val_cameras.json"),"r") as f:
            self.val_cams = json.load(f)
        with open(Path(root_cameras,"test_cameras.json"),"r") as f:
            self.test_cams = json.load(f)
        # with open(Path(root_cameras,"circle_cameras.json"),"r") as f:
        #     self.cicle_cams = json.load(f)
    
    def load_srn_cams(self):
        print("load srn cams...")
        def _load_pose_txt(path_to_txt:str) -> np.array:
            with open(path_to_txt, 'r') as file:
                contents = file.read()
                pose = np.array([float(x) for x in contents.split(" ")]).reshape(4,4)
                # pose = pose.transpose()
                return pose
        
        # shapenet srn von https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR
        root = Path("/globalwork/giese/srn_cars/cars_test/7c5e0fd7e8c62d047eed1e11d741a3f1/pose")
        files = sorted(list(root.iterdir()))
        poses = [_load_pose_txt(f) for f in files]
        poses = torch.tensor(poses).cuda()
        poses[:,:3, 1:3] *= -1
        self.srn_poses = poses

        # read split
        with open("/globalwork/giese/srn_cars/cars_test.lst", "r") as f:
            self.cars_srn_split = [l.strip() for l in f.readlines()]

        with open("/globalwork/giese/srn_chairs/chairs_test.lst", "r") as f:
            self.chairs_srn_split = [l.strip() for l in f.readlines()]

    def get_camera_object(self, cam, batch_size):
        camera = Camera(
                world_view_transform = torch.cat(
                    [torch.tensor(cam[key]["world_view_transform"]).unsqueeze(0) for key in cam],
                    dim=0).repeat(batch_size,1,1).cuda(),
                projection_matrix = torch.cat(
                    [torch.tensor(cam[key]["projection_matrix"]).unsqueeze(0) for key in cam],
                    dim=0).repeat(batch_size,1,1).cuda(),
                tanfovy = torch.tensor([0.45714] * len(cam) * batch_size, dtype=torch.float64),
                tanfovx = torch.tensor([0.45714] * len(cam) * batch_size, dtype=torch.float64),
                imsize=torch.tensor([400] * len(cam) * batch_size),
            )   
        return camera
    
    def prepare_context(self, batch_size, camera_split):
        if camera_split == "test":
            cam = self.test_cams
        elif camera_split == "val":
            cam = self.val_cams
        elif camera_split == "train":
            cam = self.train_cams
        context = GaussianContext3d(
            image = torch.tensor(1).cuda(), # der muss cuda, damit es nicht nochmal auf die gpu kopiert wird in self.render_fn
            K = None,
            c2w = None,
            w2c = None,
            camera = self.get_camera_object(cam, batch_size),
            splatting_cameras=None,
            mask_points = None,
            insinfo = None,
        )
        return context
        

    def sample_batch(self, example, **kwargs):
        shape = example.data.shape
        if Mode.cholesky in self.mode:
            shape = (example.data.shape[0],example.data.shape[1], 13)
            # sample = cholesky_L.sample(self.model, shape, example.ctx, None, **kwargs)
            kwargs['gt_data'] = example.data
            # sample = cholesky_L.sample_logdirection_test(self.model, shape, example.ctx, None, **kwargs) # -> geht, aber eher schlecht (eig kein unterschied?)
            sample = cholesky_L.sample_logdirection_tangent(self.model, shape, example.ctx, None, **kwargs) # -> geht
            # sample = cholesky_L.sample_logdirection_tangent_doesntwork(self.model, shape, example.ctx, None, **kwargs) # -> geht mittlerweile
            # sample = cholesky_L.sample_logdirection_tangent_worked(self.model, shape, example.ctx, None, **kwargs) # -> geht mittlerweile
            # sample = cholesky_L.sample_standard_ode(self.model, shape, example.ctx, None, **kwargs)
            # sample = cholesky_L.sample_ddpm(self.model, shape, example.ctx, None, **kwargs)
        elif Mode.so3_x0 in self.mode:
            shape = (example.data.shape[0],example.data.shape[1], 13)
            sample = so3_x0.sample(self.model, shape, example.ctx, None, **kwargs)
            # sample = so3_x0.sample_rot_step(self.model, shape, example.ctx, None)
        elif Mode.procrustes in self.mode:
            gen = torch.Generator(example.data.device).manual_seed(int(torch.rand(1).item()*1000))
            shape = (example.data.shape[0],example.data.shape[1], 19)
            # sample = vanilla.sample_churn(self.model, shape, example.ctx, gen, **kwargs)
            sample = vanilla.sample(self.model, shape, example.ctx, gen, **kwargs)
            # sample = vanilla.sample_procrustes_so3(self.model, shape, example.ctx, gen, **kwargs)
        elif Mode.so3_diffusion in self.mode:
            shape = (example.data.shape[0], example.data.shape[1], 1000) # last dim doesnt matter
            t1 = time.time()
            sample = so3.sample_1step(self.model, shape, example.ctx, None, **kwargs)
            # sample = so3.sample_2step(self.model, shape, example.ctx, None, **kwargs)
            # sample = so3.sample_2step_both(self.model, shape, example.ctx, None, **kwargs)
            print("Sampling took:", time.time()-t1)
        else:
            sample = vanilla.sample(self.model, shape, example.ctx, None, **kwargs)
        print("Sampling done")
        return sample
    
    def create_target_dirs(self, example):
        for i in range(example.data.shape[0]):
            val_target_dir = Path(self.output_dir, example.ctx.insinfo.category[i], example.ctx.insinfo.instance[i], "metrics", "val")
            val_target_dir.mkdir(parents=True,exist_ok=True)
            test_target_dir = Path(self.output_dir, example.ctx.insinfo.category[i], example.ctx.insinfo.instance[i], "metrics", "test")
            test_target_dir.mkdir(parents=True,exist_ok=True)

    def save_imgs_to_target_dirs(self, rendered_images, example):
        for i in range(example.data.shape[0]):
            target_dir = Path(self.output_dir, example.ctx.insinfo.category[i], example.ctx.insinfo.instance[i], "metrics", self.camera_split)
            for j in range(5):
                img = rendered_images[i*5+j]
                save_image(img, Path(target_dir, f"img_{str(j).zfill(4)}.png"))

    def white_background(self, img):
        img = img / 255
        background_mask = img[3] == 0
        img[:,background_mask] = 1
        img = img[:3,:,:]
        return img
        
    def compute_metrics(self, rendered_images, category, instance, gt_data, sampled_data):
        """
        geht nur für test split
        """
        out_csv_dir = Path(self.output_dir, category, instance, "metrics", self.camera_split)
        path_gt_images = Path(self.dataset_images, category, instance, self.camera_split)
        ssims = []
        lpipss = []
        psnrs = []
        chamfers_xyz = []
        losses = []
        chamfer_xyz = benchmark_jax_splat.chamfer_formula_smart_l1_np(sampled_data[:,:3].cpu().numpy(), gt_data[:,:3].cpu().numpy())
        chamfers_xyz.append(chamfer_xyz)
        for i,(sampled_img, gt_img_path) in enumerate(zip(rendered_images, [x for x in sorted(path_gt_images.glob("*.png")) if not "depth" in x.name])):
            gt_img = read_image(str(gt_img_path)).cuda() # range 0-255
            gt_img = self.white_background(gt_img)
            # save_image(gt_img, self.output_dir / category / instance / "metrics" / self.camera_split / f"{gt_img_path.name.split('.')[0]}_gt.png")
            # save_image(gt_img, f"temp_{i}.png")
            # save_image(sampled_img, f"temp_samp{i}.png")
            ssims.append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), gt_img.unsqueeze(0)).item())
            lpipss.append(self.lpips_fn_net(sampled_img, gt_img).item())
            psnrs.append(psnr(sampled_img, gt_img).mean().item())
            losses.append(l1_loss(sampled_img, gt_img).item())
            self.all_evals.append(ssims[-1], lpipss[-1], losses[-1], psnrs[-1])
            
        res = pd.DataFrame({
            'iterations' : [10000],
            'test_psnr' : [sum(psnrs) / len(psnrs)],
            'test_l1' : [sum(losses) / len(losses)],
            'test_lpsips' : [sum(lpipss) / len(lpipss)],
            'test_ssim' : [sum(ssims) / len(ssims)],
            'chamfer_xyz' : [chamfer_xyz.item()],
        })
        print(f"results for instance {instance}:")
        print(res)
        res.to_csv(out_csv_dir / "eval_10000.csv",
                  index=False)
        self.all_evals.summarize()
        
    def example_to_cuda(self, example):
        return GaussianExample(
            data = example.data.cuda(),
            ctx = ctx_to_cuda(example.ctx)
        )

    def reverse_sampling(self, example):
        kwargs = {
            'reverse_ode' : True,
            'gt_data' : example.data,
            'reparam' : self.reparam,
        }
        if Mode.cholesky in self.mode:
            reverse_sample = cholesky_L.sample_logdirection_tangent(self.model, example.data.shape, example.ctx, None, **kwargs)
            # reverse_sample = cholesky_L.sample_ddpm(self.model, example.data.shape, example.ctx, None, **kwargs)
        else:
            reverse_sample = vanilla.sample(self.model, example.data.shape, example.ctx, None, **kwargs)
        return reverse_sample
    
    def calc_likelihood(self, reverse_sample):
        data = reverse_sample.cpu().numpy().reshape(-1)  # Replace with your actual data; should be an array of 13-dimensional vectors
        mu = np.zeros(1)  # Replace with the 13-dimensional mean vector
        sigma = np.diag(165*np.ones(1))  # Replace with the 13x13 covariance matrix

        # Initialize the multivariate normal distribution
        mvn = multivariate_normal(mean=mu, cov=sigma)

        # Calculate the probability density of each observation
        prob_densities = mvn.pdf(data)

        # Calculate the joint likelihood
        sum_likelihood = np.sum(prob_densities)
        self.all_evals.append_likelihood(sum_likelihood)
        print(sum(self.all_evals.likelihoods)/len(self.all_evals.likelihoods))

        print("sum Likelihood:", sum_likelihood)

    def prob_density_reverse(self, example):
        reverse_sample = self.reverse_sampling(example)
        # Example data and parameters
        if Mode.cholesky in self.mode:
            noisy_L = reverse_sample[:,:,7:13]
            scale, rotation = L_to_scale_rotation(noisy_L.reshape(-1,6))
            likelihood_data = torch.cat([reverse_sample[:,:,:7], scale.reshape(-1,4000,3)], dim = -1)
            plot_pca(likelihood_data, self.output_dir)
            # plot_pca(reverse_sample[:,:,:7], self.output_dir)
            # plot_pca(scale.reshape(-1,4000,3), self.output_dir)
            self.calc_likelihood(likelihood_data)
        else:
            plot_pca(reverse_sample)
            self.calc_likelihood(reverse_sample)

    def save_ctx_images(self, images, example, name):
        for i in range(images.shape[0]):
            save_image(images[i], Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"{name}.png"))

    def make_circle_video(self, samples, example):
        height = 400
        width = 400
        old_world_view_transforms = []
        for j in range(samples.shape[0]):
            old_world_view_transforms.append(example.ctx.camera.world_view_transform[j].clone())
        writers = [cv2.VideoWriter(str(Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"circle.avi")), cv2.VideoWriter_fourcc(*'MJPG'), fps=60,frameSize=(height,width)) for i in range(example.data.shape[0])]
        gt_writers = [cv2.VideoWriter(str(Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"circle_gt.avi")), cv2.VideoWriter_fourcc(*'MJPG'), fps=60,frameSize=(height,width)) for i in range(example.data.shape[0])]
        for n in range(len(self.circle_cams)):
            ssims = []
            lpipss = []
            psnrs = []
            losses = []
            for j in range(samples.shape[0]):
                example.ctx.camera.world_view_transform[j] = self.circle_cams[n].world_view_transform
            rendered = self.render_fn(samples, example.ctx, self.mode)['render']
            gt_rendered = self.render_fn(example.data, example.ctx, self.mode)['render']
            for i,(sampled_img,gt_img) in enumerate(zip(rendered, gt_rendered)):
                ssims.append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item())
                lpipss.append(self.lpips_fn_net(sampled_img, gt_img).item())
                psnrs.append(psnr(sampled_img, gt_img).mean().item())
                losses.append(l1_loss(sampled_img, gt_img).item())
                sampled_img = sampled_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                gt_img = gt_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                writers[i].write(cv2.cvtColor(sampled_img, cv2.COLOR_RGB2BGR))
                gt_writers[i].write(cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
            self.all_evals.add_circle_eval(psnrs, ssims, lpipss, losses, n)

        for i in range(len(writers)):
            writers[i].release()
            gt_writers[i].release()
            with open(Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"download.txt"), "w") as f:
                f.write(f"scp -r vision:{Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f'circle.avi')} . \n")
                f.write(f"scp -r vision:{Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f'circle_gt.avi')} . \n")

        for j in range(samples.shape[0]):
            example.ctx.camera.world_view_transform[j] = old_world_view_transforms[j]

    def eval_360_one_pose(self, example):
        train_ctx = self.prepare_context(1, "train")
        test_ctx = self.prepare_context(1, "test")
        splat_cam = make_camera(test_ctx, 0)
        wvtransform = splat_cam.world_view_transform.clone()
        step_z = 170 # azimuth, also aus 360
        step_x = 40 # elevation, also aus 90
        total_rotation = 360
        heatmap_ssims = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_lpips = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_psnrs = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_losses = [[] for _ in range(int(total_rotation/step_z))]

        for i in range(int(total_rotation/ step_z)):
            for j in range(int(90/step_x)):
                ssims = []
                lpipss = []
                psnrs = []
                losses = []
                theta_z = torch.tensor(i * step_z * torch.pi/180)
                theta_x = torch.tensor(j * step_x * torch.pi/180)
                R_z = torch.tensor([
                    [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                    [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]).cuda()
                R_x = torch.tensor([
                    [1, 0, 0, 0],
                    [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                    [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                    [0, 0, 0, 1]
                ]).cuda()
                bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
                wv_rotated = wvtransform.clone()
                # wv_rotated = torch.matmul(R_y, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
                splat_cam.world_view_transform[:] = wv_rotated
                cond_renders = []
                for example_index in range(example.data.shape[0]):
                    gm = GaussianModel(3)
                    gm.load_ply(f"/globalwork/giese/gaussians/{example.ctx.insinfo.category[example_index]}/{example.ctx.insinfo.instance[example_index]}/point_cloud/iteration_10000/point_cloud.ply")
                    rend = render(gm, bg, splat_cam)['render']
                    cond_renders.append(rend.unsqueeze(0))
                save_image(cond_renders[0], f"image.png")
                kwargs = {
                        'num_steps' : self.number_of_steps,
                    }
                gen_example = GaussianExample(torch.zeros_like(example.data),
                                            GaussianContext3d(image=torch.cat(cond_renders, dim = 0).cuda(),
                                                                K = example.ctx.K,
                                                                c2w = None,
                                                                w2c = wv_rotated.repeat(example.data.shape[0],1,1).transpose(1,2)[:,:3,:],
                                                                camera = splat_cam,
                                                                splatting_cameras = None,
                                                                mask_points = None,
                                                                insinfo = None))
                                                            
                sampled = self.sample_batch(gen_example,**kwargs)
                # example.ctx.camera.world_view_transform[0] = wv_rotated.repeat(example.data.shape[0],1,1)[0]
                # rendered = self.render_fn(sampled, example.ctx, self.mode)['render']
                # save_image(rendered[0], f"image.png")
                print(i,j)
                psnrs, ssims, lpipss, losses = self.calc_srn_metrics(sampled, example)
                print(f"SSIM: {sum(ssims)/len(ssims)}")
                print(f"LPIPS: {sum(lpipss)/len(lpipss)}")
                print(f"PSNR: {sum(psnrs)/len(psnrs)}")
                heatmap_ssims[i].append(sum(ssims)/len(ssims))
                heatmap_lpips[i].append(sum(lpipss)/len(lpipss))
                heatmap_psnrs[i].append(sum(psnrs)/len(psnrs))
                heatmap_losses[i].append(sum(losses)/len(losses))

        plot_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)
        plot_heatlines(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)

        b = 1/0

    def eval_based_on_conditioning_pose(self, example):
        train_ctx = self.prepare_context(1, "train")
        test_ctx = self.prepare_context(1, "test")
        splat_cam = make_camera(test_ctx, 0)
        wvtransform = splat_cam.world_view_transform.clone()
        step_z = 10 # azimuth, also aus 360
        step_x = 10 # elevation, also aus 90
        total_rotation = 360
        heatmap_ssims = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_lpips = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_psnrs = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_losses = [[] for _ in range(int(total_rotation/step_z))]

        for i in range(int(total_rotation/ step_z)):
            for j in range(int(90/step_x)):
                ssims = []
                lpipss = []
                psnrs = []
                losses = []
                theta_z = torch.tensor(i * step_z * torch.pi/180)
                theta_x = torch.tensor(j * step_x * torch.pi/180)
                R_z = torch.tensor([
                    [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                    [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]).cuda()
                R_x = torch.tensor([
                    [1, 0, 0, 0],
                    [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                    [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                    [0, 0, 0, 1]
                ]).cuda()
                bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
                wv_rotated = wvtransform.clone()
                # wv_rotated = torch.matmul(R_y, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
                splat_cam.world_view_transform[:] = wv_rotated
                cond_renders = []
                for example_index in range(example.data.shape[0]):
                    gm = GaussianModel(3)
                    gm.load_ply(f"/globalwork/giese/gaussians/{example.ctx.insinfo.category[example_index]}/{example.ctx.insinfo.instance[example_index]}/point_cloud/iteration_10000/point_cloud.ply")
                    rend = render(gm, bg, splat_cam)['render']
                    cond_renders.append(rend.unsqueeze(0))
                save_image(cond_renders[0], f"image.png")
                kwargs = {
                        'num_steps' : self.number_of_steps,
                    }
                gen_example = GaussianExample(torch.zeros_like(example.data),
                                            GaussianContext3d(image=torch.cat(cond_renders, dim = 0).cuda(),
                                                                K = example.ctx.K,
                                                                c2w = None,
                                                                w2c = wv_rotated.repeat(example.data.shape[0],1,1).transpose(1,2)[:,:3,:],
                                                                camera = splat_cam,
                                                                splatting_cameras = None,
                                                                mask_points = None,
                                                                insinfo = None))
                                                            
                sampled = self.sample_batch(gen_example,**kwargs)
                # example.ctx.camera.world_view_transform[0] = wv_rotated.repeat(example.data.shape[0],1,1)[0]
                # rendered = self.render_fn(sampled, example.ctx, self.mode)['render']
                # save_image(rendered[0], f"image.png")
                print(i,j)
                psnrs, ssims, lpipss, losses = self.calc_srn_metrics(sampled, example)
                print(f"SSIM: {sum(ssims)/len(ssims)}")
                print(f"LPIPS: {sum(lpipss)/len(lpipss)}")
                print(f"PSNR: {sum(psnrs)/len(psnrs)}")
                heatmap_ssims[i].append(sum(ssims)/len(ssims))
                heatmap_lpips[i].append(sum(lpipss)/len(lpipss))
                heatmap_psnrs[i].append(sum(psnrs)/len(psnrs))
                heatmap_losses[i].append(sum(losses)/len(losses))

        plot_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)
        plot_heatlines(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)

        b = 1/0

    def explore_ctx(self, samples, example):
        old_world_view_transforms = []
        for j in range(samples.shape[0]):
            old_world_view_transforms.append(example.ctx.camera.world_view_transform[j].clone())

        theta = torch.tensor(int(360)/self.number_of_angles * torch.pi / 180).cuda()
        R_z = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0, 0],
        [torch.sin(theta), torch.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]).cuda()
        
        height = 400
        width = 400
        writers = [cv2.VideoWriter(str(Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"circle_ctx.avi")), cv2.VideoWriter_fourcc(*'MJPG'), fps=60,frameSize=(height,width)) for i in range(example.data.shape[0])]
        gt_writers = [cv2.VideoWriter(str(Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"circle_ctx_gt.avi")), cv2.VideoWriter_fourcc(*'MJPG'), fps=60,frameSize=(height,width)) for i in range(example.data.shape[0])]
        for n in range(self.number_of_angles):
            ssims = []
            lpipss = []
            psnrs = []
            losses = []
            rendered_images = self.render_fn(samples, example.ctx, self.mode)['render']
            rendered_gts = self.render_fn(example.data, example.ctx, self.mode)['render']
            for i,(sampled_img,gt_img) in enumerate(zip(rendered_images, rendered_gts)):
                ssims.append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item())
                lpipss.append(self.lpips_fn_net(sampled_img, gt_img).item())
                psnrs.append(psnr(sampled_img, gt_img).mean().item())
                losses.append(l1_loss(sampled_img, gt_img).item())
                writers[i].write(cv2.cvtColor((sampled_img.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                gt_writers[i].write(cv2.cvtColor((gt_img.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            self.all_evals.add_ctx_eval(psnrs, ssims, lpipss, losses, n)

            world_view_transforms = example.ctx.camera.world_view_transform
            Rs_rotated = torch.stack([torch.matmul(R_z, world_view_transform).cuda() for world_view_transform in world_view_transforms])
            for j in range(len(Rs_rotated)):
                example.ctx.camera.world_view_transform[j] = Rs_rotated[j]
        for i in range(len(writers)):
            writers[i].release()
            filename = Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f"download.txt")
            if not filename.exists():
                mode = "w"
            else:
                mode = "a"
            with open(filename, mode) as f:
                f.write(f"scp -r vision:{Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f'circle_ctx.avi')} . \n")
                f.write(f"scp -r vision:{Path(self.output_dir,example.ctx.insinfo.category[i],example.ctx.insinfo.instance[i],f'circle_ctx_gt.avi')} . \n")
        for j in range(samples.shape[0]):
            example.ctx.camera.world_view_transform[j] = old_world_view_transforms[j]

    def explore_ctx_regenerate(self, samples, example):
            theta = torch.tensor(int(360)/self.number_of_angles_autoregressive * torch.pi / 180).cuda()
            R_z = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0, 0],
            [torch.sin(theta), torch.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ]).cuda()
            for n in range(self.number_of_angles):
                samples = self.sample_batch(example)
                ssims = []
                lpipss = []
                psnrs = []
                losses = []
                # render with rotated view

                rendered_images = self.render_fn(samples, example.ctx, self.mode)['render']
                save_image(rendered_images[0],f"img{n}.png")
                rendered_gts = self.render_fn(example.data, example.ctx, self.mode)['render']
                for _,(sampled_img,gt_img) in enumerate(zip(rendered_images, rendered_gts)):
                    ssims.append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item())
                    lpipss.append(self.lpips_fn_net(sampled_img, gt_img).item())
                    psnrs.append(psnr(sampled_img, gt_img).mean().item())
                    losses.append(l1_loss(sampled_img, gt_img).item())

                self.all_evals.add_ctx_eval_autoregressive(psnrs, ssims, lpipss, losses, n)

                world_view_transforms = example.ctx.camera.world_view_transform
                Rs_rotated = torch.stack([torch.matmul(R_z, world_view_transform).cuda() for world_view_transform in world_view_transforms])
                for j in range(len(Rs_rotated)):
                    example.ctx.camera.world_view_transform[j] = Rs_rotated[j]
                rendered_images_next = self.render_fn(samples, example.ctx, self.mode)['render']

                for j in range(len(Rs_rotated)):
                    example.ctx.image[j] = rendered_images_next[j]

    def swap_in_test_ctx(self,example):
        for i in range(example.data.shape[0]):
            example.ctx.w2c[i] = example.ctx.splatting_cameras[-1][0].world_view_transform[i].transpose(0,1)[:3,:]
            example.ctx.image[i] = example.ctx.splatting_cameras[-1][1][i]

    def rotate_conditional_image(self,example):
        gm = GaussianModel(3)
        gm.load_ply(f"/globalwork/giese/gaussians/02958343/8c6c271a149d8b68949b12cf3977a48b/point_cloud/iteration_10000/point_cloud.ply")
        cond = read_image("/globalwork/giese/shapenet_rendered/02958343/8c6c271a149d8b68949b12cf3977a48b/test/0000.png").cuda().type(torch.float32)
        cond = self.white_background(cond)
        # save_image(cond, "cond.png")
        # gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
        train_ctx = self.prepare_context(1, "train")
        test_ctx = self.prepare_context(1, "test")
        splat_cam = make_camera(test_ctx, 0)
        wvtransform = splat_cam.world_view_transform.clone()
        step_z = 170 # azimuth, also aus 360
        step_x = 40 # elevation, also aus 90
        total_rotation = 360
        heatmap_ssims = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_lpips = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_psnrs = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_losses = [[] for _ in range(int(total_rotation/step_z))]

        for i in range(int(total_rotation/ step_z)):
            for j in range(int(90/step_x)):
                ssims = []
                lpipss = []
                psnrs = []
                losses = []
                theta_z = torch.tensor(i * step_z * torch.pi/180)
                theta_x = torch.tensor(j * step_x * torch.pi/180)
                R_z = torch.tensor([
                    [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                    [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]).cuda()
                R_x = torch.tensor([
                    [1, 0, 0, 0],
                    [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                    [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                    [0, 0, 0, 1]
                ]).cuda()
                bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
                wv_rotated = wvtransform.clone()
                wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
                splat_cam.world_view_transform[:] = wv_rotated
                rend = render(gm, bg, splat_cam)['render']
                # save_image(rend, "image.png") # hat logischerweise white background
                kwargs = {
                        'num_steps' : self.number_of_steps,
                    }
                example = GaussianExample(torch.zeros_like(example.data[0]).unsqueeze(0),
                                            GaussianContext3d(image=rend.unsqueeze(0),
                                                                K = example.ctx.K[0].unsqueeze(0),
                                                                c2w = None,
                                                                w2c = test_ctx.camera.world_view_transform[0].unsqueeze(0),
                                                                camera = splat_cam,
                                                                splatting_cameras = None,
                                                                mask_points = None,
                                                                insinfo = None))
                                                            
                sampled = self.sample_batch(example,**kwargs)
                print(i,j)
                rendered_images_train = self.render_fn(sampled.repeat(len(train_ctx.camera.world_view_transform),1,1), train_ctx, self.mode)['render']
                for z in range(rendered_images_train.shape[0]):
                    # save_image(rendered_images_train[z], f"image_{z}.png")
                    sampled_img = rendered_images_train[z]
                    # save_image(render(gm,bg,make_camera(train_ctx,z))['render'], f"image_{z}_gt.png")
                    gt_img = render(gm,bg,make_camera(train_ctx,z))['render']
                    ssims.append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item())
                    lpipss.append(self.lpips_fn_net(sampled_img, gt_img).item())
                    psnrs.append(psnr(sampled_img, gt_img).mean().item())
                    losses.append(l1_loss(sampled_img, gt_img).item())
                print(f"SSIM: {sum(ssims)/len(ssims)}")
                print(f"LPIPS: {sum(lpipss)/len(lpipss)}")
                print(f"PSNR: {sum(psnrs)/len(psnrs)}")
                heatmap_ssims[i].append(sum(ssims)/len(ssims))
                heatmap_lpips[i].append(sum(lpipss)/len(lpipss))
                heatmap_psnrs[i].append(sum(psnrs)/len(psnrs))
                heatmap_losses[i].append(sum(losses)/len(losses))

        plot_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)
        plot_heatlines(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)

        b = 1/0

    def threesixty_eval_one_cond_imag(self,example):
        gm = GaussianModel(3)
        gm.load_ply(f"/globalwork/giese/gaussians/02958343/8c6c271a149d8b68949b12cf3977a48b/point_cloud/iteration_10000/point_cloud.ply")
        # gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
        train_ctx = self.prepare_context(1, "train")
        test_ctx = self.prepare_context(1, "test")
        splat_cam = make_camera(test_ctx, 0)
        wvtransform = splat_cam.world_view_transform.clone()
        step_z = 2 # azimuth, also aus 360
        step_x = 2 # elevation, also aus 90
        total_rotation = 360
        heatmap_ssims = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_lpips = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_psnrs = [[] for _ in range(int(total_rotation/step_z))]
        heatmap_losses = [[] for _ in range(int(total_rotation/step_z))]

        theta_z = torch.tensor(30 * torch.pi/180)
        theta_x = torch.tensor(15 * torch.pi/180)
        # theta_z = torch.tensor(0 * torch.pi/180)
        # theta_x = torch.tensor(0 * torch.pi/180)
        R_z = torch.tensor([
            [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
            [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ]).cuda()
        R_x = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
            [0, torch.sin(theta_x), torch.cos(theta_x), 0],
            [0, 0, 0, 1]
        ]).cuda()
        wv_rotated = wvtransform.clone()
        wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
        wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
        splat_cam.world_view_transform[:] = wv_rotated
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
        rend = render(gm, bg, splat_cam)['render']
        save_image(rend, "image.png") # hat logischerweise white background
        kwargs = {
                'num_steps' : self.number_of_steps,
            }
        example = GaussianExample(torch.zeros_like(example.data[0]).unsqueeze(0),
                                    GaussianContext3d(image=rend.unsqueeze(0),
                                                        K = example.ctx.K[0].unsqueeze(0),
                                                        c2w = None,
                                                        w2c = wv_rotated.transpose(0,1)[:3,:].unsqueeze(0),
                                                        camera = splat_cam,
                                                        splatting_cameras = None,
                                                        mask_points = None,
                                                        insinfo = None))
                                                    
        sampled = self.sample_batch(example,**kwargs)
        for i in range(int(total_rotation/ step_z)):
            for j in range(int(90 / step_x)):
                print(i,j)
                theta_z = torch.tensor(i*step_x * torch.pi/180)
                theta_x = torch.tensor(j*step_z * torch.pi/180)
                R_z = torch.tensor([
                    [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                    [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]).cuda()
                R_x = torch.tensor([
                    [1, 0, 0, 0],
                    [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                    [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                    [0, 0, 0, 1]
                ]).cuda()
                wv_rotated = wvtransform.clone()
                wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
                splat_cam.world_view_transform[:] = wv_rotated.clone()
                gt_img = render(gm,bg,splat_cam)['render']
                train_ctx.camera.world_view_transform[0] = wv_rotated
                rendered_images_train = self.render_fn(sampled.repeat(len(train_ctx.camera.world_view_transform),1,1), train_ctx, self.mode)['render']
                sampled_img = rendered_images_train[0]
                # save_image(render(gm,bg,make_camera(train_ctx,z))['render'], f"image_{z}_gt.png")
                ssim_val =self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item()
                lpips_val = self.lpips_fn_net(sampled_img, gt_img).item()
                psnr_val = psnr(sampled_img, gt_img).mean().item()
                l1_val = l1_loss(sampled_img, gt_img).item()
                print(f"SSIM: {ssim_val}")
                print(f"LPIPS: {lpips_val}")
                print(f"PSNR: {psnr_val}")
                print(f"L1: {l1_val}")
                heatmap_ssims[i].append(ssim_val)
                heatmap_lpips[i].append(lpips_val)
                heatmap_psnrs[i].append(psnr_val)
                heatmap_losses[i].append(l1_val)
        plot_polar_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses)
        plot_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)
        plot_heatlines(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z)

        b = 1/0

    def find_srn_pose_param(self):
        gm = GaussianModel(3)
        gm.load_ply(f"/globalwork/giese/gaussians/02958343/8c6c271a149d8b68949b12cf3977a48b/point_cloud/iteration_10000/point_cloud.ply")
        select = "a0a1b0377d72e86bab3dd76bf33b0f5e"
        gm.load_ply(f"/globalwork/giese/gaussians/02958343/{select}/point_cloud/iteration_10000/point_cloud.ply")
        bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
        test_ctx = self.prepare_context(1, "test")
        splat_cam: Camera = make_camera(test_ctx, 2)
        # 64th conditioning camera
        for i in range(251):
            p = self.srn_poses[i].clone()
            p[:3,1:3] *= -1
            p[3,2] = 1.363
            splat_cam.world_view_transform[:] = p
            rend_dict = render(gm, bg, splat_cam)
            rend = rend_dict['render']
            rend_depth = rend_dict['depth']
            obj_ratio_render = (rend_depth > 0).sum() / (rend_depth.shape[0] * rend_depth.shape[1])
            # save_image(rend, "image.png")
            orig = read_image(f"/globalwork/giese/srn_cars/cars_test/{select}/rgb/{str(i).zfill(6)}.png")
            import shutil
            shutil.copy(f"/globalwork/giese/srn_cars/cars_test/{select}/rgb/{str(i).zfill(6)}.png", "image1.png")
            # checke wo es überall weiß ist
            obj_mask_srn = torch.all(orig[:3] != 255, dim=0)
            obj_ratio_srn = obj_mask_srn.sum() / (obj_mask_srn.shape[0] * obj_mask_srn.shape[1])
            print(i, obj_ratio_render, obj_ratio_srn)

    def uncond_eval(self, samples, example):
        # sample random pose, as predefined in https://github.com/NIRVANALAN/LN3Diff/blob/a79eb8360a39f7cf591becf00cba608f70868e5c/nsr/train_util.py#L1177
        device = samples.device
        pitch_range = 1
        yaw_range = 2
        num_keyframes = 10  # how many nv poses to sample from
        w_frames = 1
        cam_radius = 1.7
        camera_pivot = torch.tensor([0,0,0],dtype=torch.float32,device=device)
        print("Rendering...")
        # check if done
        num_eles = len(list(Path(self.output_dir).rglob('*.png')))
        print(f"Number of elements in {self.output_dir}: {num_eles}")
        if num_eles > 50000:
            1/0
        for frame_idx in range(num_keyframes):
            yaw_val = 3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx /(num_keyframes * w_frames))
            pitch_val = 3.14 / 2 - 0.05 +pitch_range * np.cos(2 * 3.14 * frame_idx /(num_keyframes * w_frames))
            print(f"frame_id {frame_idx} Yaw: {yaw_val}, Pitch: {pitch_val}")
            cam2world_pose = LookAtPoseSampler.sample(
                    yaw_val, # horizontal mean
                    pitch_val,  # vertical mean
                    camera_pivot,
                    radius=cam_radius,
                    device=device,)
            cam2world_pose = cam2world_pose.squeeze(0)
            # cam2world_pose[3,2] = cam2world_pose[2,3]
            test_ctx = self.prepare_context(1, "test")
            splat_cam: Camera = make_camera(test_ctx, 2)
            pose = cam2world_pose.squeeze(0)
            pose[3,2] = 1.7
            pose[2,3] = 1.7
            splat_cam.world_view_transform[:] = pose
            for i in range(example.data.shape[0]):
                gm = GaussianModel(3)
                gm.load_ply(f"/globalwork/giese/gaussians/{example.ctx.insinfo.category[i]}/{example.ctx.insinfo.instance[i]}/point_cloud/iteration_10000/point_cloud.ply")
                bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
                rend = render(gm, bg, splat_cam)['render']
                save_image(rend, "image__.png")
                example.ctx.camera.world_view_transform[i] = pose
                rendered = self.render_fn(samples, example.ctx, self.mode)['render']
                save_index = 0
                save_path = Path(self.output_dir, example.ctx.insinfo.category[i]+"_"+example.ctx.insinfo.instance[i]+f"_{frame_idx}_{save_index}.png")
                while save_path.exists():
                    save_index += 1
                    save_path = Path(self.output_dir, example.ctx.insinfo.category[i]+"_"+example.ctx.insinfo.instance[i]+f"_{frame_idx}_{save_index}.png")
                save_image(rendered[i], save_path)

    def calc_srn_metrics(self, samples, example):
        ssims = []
        lpipss = []
        psnrs = []
        losses = []
        selected_model_index = 8
        for i in range(251):
            if i == 64:
                continue
            p = self.srn_poses[i].clone()
            p[:3,1:3] *= -1
            p[3,2] = 1.363
            for j in range(example.data.shape[0]):
                example.ctx.camera.world_view_transform[j] = p
            rendered_images_samples = self.render_fn(samples, example.ctx, self.mode)['render']
            rendered_images_gt = self.render_fn(example.data, example.ctx, self.mode)['render']
            for j in range(samples.shape[0]):
                if not self.all_categories:
                    if example.ctx.insinfo.instance[j] not in self.cars_srn_split and example.ctx.insinfo.instance[j] not in self.chairs_srn_split:
                        continue
                sampled_img = rendered_images_samples[j]
                gt_img = rendered_images_gt[j]

                # save_image(sampled_img, Path(self.output_dir, f"{example.ctx.insinfo.instance[j]}_{i}_{j}.png"))
                # save_image(gt_img, Path(self.output_dir, f"{example.ctx.insinfo.instance[j]}_gt_{i}_{j}.png"))

                ssim = self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item()
                ssims.append(ssim)
                lpips = self.lpips_fn_net(sampled_img, gt_img).item()
                lpipss.append(lpips)
                psnr_value = psnr(sampled_img, gt_img).mean().item()
                psnrs.append(psnr_value)
                l1 = l1_loss(sampled_img, gt_img).item()
                losses.append(l1)
                self.all_evals.add_srn_eval_category(example.ctx.insinfo.category[j], psnr_value, ssim, lpips, l1)
        print(f"PSNR srn: {sum(psnrs)/len(psnrs)}")
        print(f"SSIM srn: {sum(ssims)/len(ssims)}")
        print(f"LPIPS srn: {sum(lpipss)/len(lpipss)}")
        print(f"L1 srn: {sum(losses)/len(losses)}")
        return psnrs, ssims, lpipss, losses

    def srn_eval(self, example):
        """
        erstmal muss das model gesampled werden mit der 64th conditioning camera
        """
        old_world_view_transforms = []
        for j in range(example.ctx.camera.world_view_transform.shape[0]):
            old_world_view_transforms.append(example.ctx.camera.world_view_transform[j].clone())
        print("SRN eval...")
        # 64th conditioning camera
        test_ctx = self.prepare_context(1, "test")
        p = self.srn_poses[64].clone()
        splat_cam: Camera = make_camera(test_ctx, 2)
        p[:3,1:3] *= -1
        p[3,2] = 1.363
        splat_cam.world_view_transform[:] = p
        print(splat_cam.world_view_transform.shape)

        # place the 64th image as conditioning image in the context
        for i in range(example.data.shape[0]):
            gm = GaussianModel(3)
            gm.load_ply(f"/globalwork/giese/gaussians/{example.ctx.insinfo.category[i]}/{example.ctx.insinfo.instance[i]}/point_cloud/iteration_10000/point_cloud.ply")
            bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
            rend = render(gm, bg, splat_cam)['render']
            """
            jetzt muss das bild UND die w2c transformation in den ctx eingesetzt werden, weil der zum samplen benutzt wird
            """
            example.ctx.image[i] = rend
            example.ctx.w2c[i] = p.transpose(0,1)[:3,:] # srn datensatz (aus dem p geladen ist), hat die w2c transpositioniert
        kwargs = {'num_steps' : self.number_of_steps}
        samples = self.sample_batch(example,**kwargs)
        psnrs, ssims, lpipss, losses = self.calc_srn_metrics(samples, example)
        self.all_evals.add_srn_eval(psnrs, ssims, lpipss, losses)

        for j in range(example.ctx.camera.world_view_transform.shape[0]):
            example.ctx.camera.world_view_transform[j] = old_world_view_transforms[j]

    def gaussian_gt_eval(self, rendered_images, rendered_gt):
        ssims = []
        lpipss = []
        psnrs = []
        losses = []
        for i,(sampled_img,gt_img) in enumerate(zip(rendered_images, rendered_gt)):
            ssims.append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item())
            lpipss.append(self.lpips_fn_net(sampled_img, gt_img).item())
            psnrs.append(psnr(sampled_img, gt_img).mean().item())
            losses.append(l1_loss(sampled_img, gt_img).item())
            self.all_evals.append_gaussian_gt(ssims[-1], lpipss[-1], losses[-1], psnrs[-1])

    def take_360_video(self, sampled, example):
        # gm = GaussianModel(3)
        # gm.load_ply(f"/globalwork/giese/gaussians/02958343/{example.ctx.insinfo.instance[8]}/point_cloud/iteration_10000/point_cloud.ply")
        # bg = torch.tensor([1.0,1.0,1.0],device=torch.device('cuda')) # weiß, weil das orig
        # splat_cam: Camera = make_camera(test_ctx, 2)

        for b in range(example.data.shape[0]):
            train_ctx = self.prepare_context(1, "train")
            test_ctx = self.prepare_context(1, "test")
            splat_cam = make_camera(test_ctx, 0)
            wvtransform = splat_cam.world_view_transform.clone()
            step_z = 2 # azimuth, also aus 360
            step_x = 2 # elevation, also aus 90
            total_rotation = 360

            theta_z = torch.tensor(30 * torch.pi/180)
            theta_x = torch.tensor(15 * torch.pi/180)
            R_z = torch.tensor([
                [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ]).cuda()
            R_x = torch.tensor([
                [1, 0, 0, 0],
                [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                [0, 0, 0, 1]
            ]).cuda()
            wv_rotated = wvtransform.clone()
            wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
            wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
            splat_cam.world_view_transform[:] = wv_rotated
            frames = []              
            selected_model_index = b            
            for i in range(int(total_rotation/ step_z)):
                j = 7
                print(i,j)
                theta_z = torch.tensor(i*step_x * torch.pi/180)
                theta_x = torch.tensor(j*step_z * torch.pi/180)
                R_z = torch.tensor([
                    [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                    [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]).cuda()
                R_x = torch.tensor([
                    [1, 0, 0, 0],
                    [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                    [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                    [0, 0, 0, 1]
                ]).cuda()
                wv_rotated = wvtransform.clone()
                wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
                wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
                splat_cam.world_view_transform[:] = wv_rotated.clone()
                train_ctx.camera.world_view_transform[selected_model_index] = wv_rotated
                rendered_images_train = self.render_fn(sampled, train_ctx, self.mode)['render']
                sampled_img = rendered_images_train[selected_model_index]
                buf = io.BytesIO()
                save_image(sampled_img, buf, format='png')
                buf.seek(0)
                img = imageio.imread(buf)
                frames.append(img)
                buf.close()

        # Create a video from the frames
            if Mode.so3_diffusion in self.mode:
                name = "so3"
            elif Mode.cholesky in self.mode:
                name = "cholesky"
            elif Mode.activated_scales in self.mode:
                name = "act_scales"
            elif Mode.procrustes in self.mode:
                name = "proc"
            elif Mode.log_L in self.mode:
                name = "logL"
            elif Mode.normal in self.mode:
                name = "geccop"
            imageio.mimwrite(f'videos/{name}_{b}.mp4', frames, fps=60)  # Adjust fps as needed
            print(f"Created video {'videos/'+name}_{b}.mp4")
        1/0
    
    def take_ctx_video(self, sampled, example):
        train_ctx = self.prepare_context(1, "train")
        test_ctx = self.prepare_context(1, "test")
        step_z = 2 # azimuth, also aus 360
        step_x = 2 # elevation, also aus 90
        ctx_pose = example.ctx.camera.world_view_transform[8].clone()


        def pose_at_i_j(i,j):
            theta_z = torch.tensor(i*step_x * torch.pi/180)
            theta_x = torch.tensor(j*step_z * torch.pi/180)
            R_z = torch.tensor([
                [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ]).cuda()
            R_x = torch.tensor([
                [1, 0, 0, 0],
                [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                [0, 0, 0, 1]
            ]).cuda()
            wv_rotated = ctx_pose.clone()
            wv_rotated = torch.matmul(R_x, wv_rotated).cuda()
            wv_rotated = torch.matmul(R_z, wv_rotated).cuda()
            return wv_rotated
        
        image_poses = []
        # erstmal 10 nach rechts
        for i in range(10):
            j = 0
            wv_rotated = pose_at_i_j(i,j)
            image_poses.append(wv_rotated)
        # dann 10 hoch
        for j in range(10):
            i = 10
            wv_rotated = pose_at_i_j(i,j)
            image_poses.append(wv_rotated)
        # dann 20 links
        for i in range(10,0,-1):
            j = 10
            wv_rotated = pose_at_i_j(i,j)
            image_poses.append(wv_rotated)
        # dann 10 runter
        for j in range(10,0,-1):
            i = 0
            wv_rotated = pose_at_i_j(i,j)
            image_poses.append(wv_rotated)
        # 10 nach rechts
        for i in range(1,11):
            j = 0
            wv_rotated = pose_at_i_j(i,j)
            image_poses.append(wv_rotated) 

        
        def get_circle_coordinates(radius, num_points, x_center=0, y_center=0):
            coordinates = []
            
            # Divide the circle into equal angles
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            
            for theta in angles:
                x = x_center + radius * np.cos(theta)
                y = y_center + radius * np.sin(theta)
                coordinates.append((x, y))
            
            return coordinates

        # Parameters
        radius = 5  # Radius of the circle
        num_points = 200  # Number of points along the circle
        x_center = 0  # Center of the circle (x-coordinate)
        y_center = 0  # Center of the circle (y-coordinate)

        # Get coordinates of points lying on the circle
        circle_points = torch.from_numpy(np.array(get_circle_coordinates(radius, num_points, x_center, y_center))).type(torch.float32).cuda()
        image_poses = []
        for i, (x, y) in enumerate(circle_points):
            wv_rotated = pose_at_i_j(x,y)
            wv_rotated[3,2] -= 0.3
            image_poses.append(wv_rotated)

        for i, (x, y) in enumerate(circle_points):
            wv_rotated = pose_at_i_j(x,y)
            wv_rotated[3,2] -= 0.3
            if i < num_points / 2:
                wv_rotated[3,2] -= + i * 1/num_points
            # if i == int(num_points / 2):
            #     normal_circle = torch.stack(image_poses[:len(circle_points)],dim=0)
            #     normal_circle[:,3,2] -= 0.8
            #     [image_poses.append(w) for w in normal_circle]
            else:
                wv_rotated[3,2] -= (num_points - i) * 1/num_points
            image_poses.append(wv_rotated)

        
            
        frames = []
        selected_model_index = 8              
        for wv_rotated in image_poses:

            train_ctx.camera.world_view_transform[selected_model_index] = wv_rotated
            rendered_images_train = self.render_fn(sampled, train_ctx, self.mode)['render']
            sampled_img = rendered_images_train[selected_model_index]
            buf = io.BytesIO()
            save_image(sampled_img, buf, format='png')
            buf.seek(0)
            img = imageio.imread(buf)
            frames.append(img)
            buf.close()

        # Create a video from the frames
        if Mode.so3_diffusion in self.mode:
            name = "so3"
        elif Mode.cholesky in self.mode:
            name = "cholesky"
        elif Mode.activated_scales in self.mode:
            name = "act_scales"
        elif Mode.procrustes in self.mode:
            name = "proc"
        elif Mode.log_L in self.mode:
            name = "logL"
        elif Mode.normal in self.mode:
            name = "geccop"
        imageio.mimwrite(f'videos/ssplat_{name}.mp4', frames, fps=120)  # Adjust fps as needed
        print(f"Created video {'videos/ssplat_'+name}.mp4")
        1/0
        save_image(example.ctx.image[selected_model_index], "image.png")


    def save_train_cams_unconditional(self, samples, example):
        train_ctx = self.prepare_context(samples.shape[0], "train")
        num_train_views = 40
        samples_repeated = torch.repeat_interleave(samples, num_train_views, dim = 0)
        rendered_images = self.render_fn(samples_repeated, train_ctx, self.mode)['render'] # shape (48 * 5, 3, 400, 400)
        for i in range(samples.shape[0]):
            for j in range(num_train_views):
                save_image(rendered_images[i*num_train_views+j], f"{self.output_dir}_{example.ctx.insinfo.instance[i]}_{i}_{j}.png")
        
    
    def save_views_conditional(self, samples, example):
        a = 1
        train_ctx = self.prepare_context(samples.shape[0], "train")
        num_train_views = 40
        samples_repeated = torch.repeat_interleave(samples, num_train_views, dim = 0)
        rendered_images = self.render_fn(samples_repeated, train_ctx, self.mode)['render'] # shape (48 * 5, 3, 400, 400)
        for i in range(samples.shape[0]):
            for j in range(num_train_views):
                save_image(rendered_images[i*num_train_views+j], f"{self.output_dir}/{example.ctx.insinfo.category[i]}_{example.ctx.insinfo.instance[i]}_{i}_{j}.png")
            save_image(example.ctx.image[i], f"{self.output_dir}/{example.ctx.insinfo.category[i]}_{example.ctx.insinfo.instance[i]}_{i}_gt.png")

    def sample_and_render(self):
        self.all_evals.init_ctx_eval(self.number_of_angles)
        self.all_evals.init_circle_eval(len(self.circle_cams))
        data_loader = self.get_test_dataloader()
        wanted = ['9fda50a8','ef0703db','8e39ba11','3d6e798b']
        for i,example in enumerate(data_loader):
            is_present = any(substring in full_string for substring in wanted for full_string in example.ctx.insinfo.instance)
            print(f"Sample and render {example.ctx.insinfo.instance}...")
            # if not is_present:
            #     continue
            example = self.example_to_cuda(example)
            # self.prob_density_reverse(example)
            kwargs = {
                'gt_rotation_wxyz' : example.data[:,:,10:14],
                'gt_scaling' : example.data[:,:,7:10],
                'num_steps' : self.number_of_steps,
            }

            
            # self.srn_eval(example)
            samples = self.sample_batch(example,**kwargs)
            # self.take_360_video(samples, example)
            # self.take_ctx_video(samples, example)

            if self.unconditional_bool:
                # self.save_train_cams_unconditional(samples, example)
                self.uncond_eval(samples, example)
            else:
                # self.eval_based_on_conditioning_pose(example)
                # self.save_views_conditional(samples, example)
                if i > 200:
                    a = 1/0
                """
                self.create_target_dirs(example)
                """
                # self.find_srn_pose_param()
                # self.make_circle_video(samples, example)
                # self.explore_ctx(samples, example)
                # self.explore_ctx_regenerate(samples, example)
                # self.all_evals.plot_ctx_eval()

                """
                rendered_images_ctx = self.render_fn(samples, example.ctx, self.mode)['render'] # shape (48 * 5, 3, 400, 400)
                [ctx_evals['psnr'].append(psnr(sampled_img, gt_img).mean().item()) for sampled_img,gt_img in zip(rendered_images_ctx, example.ctx.image)]
                [ctx_evals['ssim'].append(self.ssim_fn(torch.clip(sampled_img.unsqueeze(0),min=0,max=1), torch.clip(gt_img.unsqueeze(0),0,1)).item()) for sampled_img,gt_img in zip(rendered_images_ctx, example.ctx.image)]
                [ctx_evals['lpips'].append(self.lpips_fn_net(sampled_img, gt_img).item()) for sampled_img,gt_img in zip(rendered_images_ctx, example.ctx.image)]
                [ctx_evals['l1'].append(l1_loss(sampled_img, gt_img).item()) for sampled_img,gt_img in zip(rendered_images_ctx, example.ctx.image)]
                print(f"in it {i} batch_size {self.batch_size}")
                print(f"average ctx psnr: {sum(ctx_evals['psnr'])/len(ctx_evals['psnr'])}")
                print(f"average ctx ssim: {sum(ctx_evals['ssim'])/len(ctx_evals['ssim'])}")
                print(f"average ctx lpips: {sum(ctx_evals['lpips'])/len(ctx_evals['lpips'])}")
                print(f"average ctx l1: {sum(ctx_evals['l1'])/len(ctx_evals['l1'])}")

                #Alle 48 sample gleichzeitig rendern für die 5 test images: 5*48 = 240
                samples_repeated = torch.repeat_interleave(samples, num_repeats, dim = 0)
                rendered_images = self.render_fn(samples_repeated, test_ctx, self.mode)['render'] # shape (48 * 5, 3, 400, 400)
                gt_rendered_images = self.render_fn(torch.repeat_interleave(example.data, num_repeats, dim = 0), test_ctx, self.mode)['render'] # shape (48 * 5, 3, 400, 400)
                
                # self.save_imgs_to_target_dirs(rendered_images, example)
                # self.save_ctx_images(example.ctx.image,example,"ctx_img")
                # self.save_ctx_images(rendered_images_ctx, example, "ctx_img_rendered")
                # example.ctx.image[2] = rendered_images_ctx[2]
                # self.srn_eval(example)
                # self.gaussian_gt_eval(rendered_images, gt_rendered_images)
                # for b in range(samples.shape[0]):
                #     self.compute_metrics(rendered_images[b*num_repeats:(b*num_repeats)+num_repeats],
                #                         example.ctx.insinfo.category[b],
                #                         example.ctx.insinfo.instance[b],
                #                         example.data[b],
                #                         samples[b],)
                print(f"eval {self.version} variante {self.variant} {i}: {time.time()-t1}")
                """


if __name__ == "__main__":
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description="Process a model to evaluate")

    # parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="48638814") # proc_uncond car
    # parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="48638932") # proc_uncond chair
    # parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="48638753") # proc_uncond plane

    # parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="48659067") # Proc all categories

    parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="47702620") # gecco++ car
    # parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="48516943") # proc car 
    # parser.add_argument('--model_name', type=str, help='Name of the model to evaluate',default="48600907") # so3 car
    parser.add_argument('--variant', type=str, help='Specific circumstances to eval', default="vid2")
    # Parse the arguments
    args = parser.parse_args()

    variant = args.variant
    # 48183265

    # You can now use args.model_name to access the string provided at the command line
    print(f"Model Name: {args.model_name}, variant: {variant}")
    GaussianGeccoEvaluator(args.model_name, variant)
    # best so3 48600907 2 varianten: 2step, 2step both

    # bild das genutzt wird: '02958343' '54514b3e6ea7ad944361eef216dfeaa6'