import os
import wandb
import pathlib
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from typing import Iterable, Optional, Callable, Union, List, Tuple
from functools import partial

from gecco_torch.utils.loss_utils import l1_loss
from gecco_torch.utils.image_utils import psnr
from piqa import SSIM
import lpips
from gecco_torch.structs import Camera, GaussianContext3d, InsInfo, Mode
from gecco_torch.additional_metrics.distance_pointclouds import chamfer_distance_naive_occupany_networks, chamfer_formula_smart_l1_np

from gecco_torch.metrics import (
    chamfer_distance,
    sinkhorn_emd,
    chamfer_distance_squared,
)

def batched_pairwise_distance(a, b, distance_fn, block_size):
    """
    Returns an N-by-N array of the distance between a[i] and b[j]
    """
    assert a.shape == b.shape, (a.shape, b.shape)

    dist = jax.vmap(jax.vmap(distance_fn, in_axes=(None, 0)), in_axes=(0, None))
    dist = jax.jit(dist)

    n_batches = int(math.ceil(a.shape[0] / block_size))
    distances = []
    for a_block in tqdm(
        np.array_split(a, n_batches), desc="Computing pairwise distances"
    ):
        row = []
        for b_block in np.array_split(b, n_batches):
            row.append(np.asarray(dist(a_block, b_block)))
        distances.append(np.concatenate(row, axis=1))
    return np.concatenate(distances, axis=0)


def extract_data(loader: Iterable, n_examples: Optional[int]):
    if n_examples is not None:
        assert len(loader.dataset) >= n_examples, len(loader.dataset)

    collected = []
    for batch in loader:
        collected.append(batch.points.numpy())
        if n_examples is not None and sum(c.shape[0] for c in collected) >= n_examples:
            break
    data = np.concatenate(collected, axis=0)[:n_examples]
    return data


class BenchmarkCallback:
    def __init__(
        self,
        list_data_and_ctx: List[Tuple[np.array,np.array]],
        epoch: int,
        render_fn: callable,
        mode,
        batch_size: int = 64,
        tag_prefix: str = "benchmark",
        rng_seed: int = 42,
        block_size: int = 16,
        distance_fn: Union[str, Callable] = chamfer_distance_squared,
        save_path: Optional[str] = None,
    ):

        self.data = [d[0] for d in list_data_and_ctx]
        self.data = torch.cat(self.data,dim=0).cpu().numpy()
        if Mode.cholesky in mode:
            self.gaussian_dimensions = 13
        else:
            self.gaussian_dimensions = self.data.shape[2]
        print(f"Create benchmark with data shape {self.data.shape} and batch size {batch_size}")
        self.mode = mode
        self.render_fn = render_fn
        """
        für jeden validation step ein context. In jedem context sind batchsize viele elemente. Aber nicht als Liste sondern im batch format
        e.g. self.contexts[0] hat genau EINE image, K, und camera. Aber image hat shape (batch size, 3, 400,400), K (batch size, 3,3)
        und self.contexts[0].camera world_view_transform,... und shape world_view_transform hat (batch_size,4,4)
        """
        self.contexts = [d[1] for d in list_data_and_ctx] 

        # [print(x) for x in self.contexts]
        if self.contexts[0] == []:
            self.contexts = None
            self.conditional = False
        else:
            self.conditional = True
            [print(ctx.image.shape) for ctx in self.contexts]
            # print(f"context shape {self.contexts.shape} and batch size {batch_size}")
        print(f"Benchmark initialized for conditional: {self.conditional}")
        self.number_of_examples = self.data.shape[0]
        self.n_points = self.data.shape[1]
        self.batch_size = batch_size
        self.tag_prefix = tag_prefix
        self.n_batches = int(math.ceil(self.data.shape[0] / self.batch_size))
        self.rng_seed = rng_seed
        self.block_size = block_size
        self.epoch = epoch

        self.lpips_fn_net = lpips.LPIPS(net = 'alex').cuda()
        self.ssim_fn = SSIM().cuda()

        if isinstance(distance_fn, str):
            distance_fn = {
                "chamfer": chamfer_distance,
                "chamfer_squared": chamfer_distance_squared,
                "emd": partial(sinkhorn_emd, epsilon=0.1),
            }[distance_fn]

        if hasattr(distance_fn, "func"):
            self.distance_fn_name = distance_fn.func.__name__
        else:
            self.distance_fn_name = distance_fn.__name__

        self.distance_fn = partial(
            batched_pairwise_distance,
            distance_fn=distance_fn,
            block_size=self.block_size,
        )

        if not self.conditional:
            self.dd_dist = self.distance_fn(self.data, self.data)

        if save_path is not None:
            save_path = os.path.join(
                save_path, "benchmark-checkpoints", self.distance_fn_name
            )
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.lowest_1nn = float("inf")

    @classmethod
    def from_loader(
        cls,
        loader,
        n_examples: Optional[int] = None,
        **kwargs,
    ) -> "BenchmarkCallback":
        data = extract_data(loader, n_examples)
        return cls(data, batch_size=loader.batch_size, **kwargs)

    def sample_from_model(self, model):
        print(f"sample {self.number_of_examples} times from model")
        samples = []
        num_full_batches = int(np.floor(self.number_of_examples/self.batch_size))
        # which_batch = np.random.choice(num_full_batches)
        with torch.no_grad():
            for i in range(num_full_batches):
                print(f"sample batch {i} from model")
                sample = model.sample_stochastic(
                    shape = (self.batch_size,self.n_points, self.gaussian_dimensions),
                    context=None if self.contexts is None else self.contexts[i],
                    rng=None,
                )
                samples.append(sample.cpu().numpy())
            if self.number_of_examples-num_full_batches*self.batch_size == 0:
                return np.concatenate(samples, axis=0)
            print(f"sample last batch from model (with shape {(self.number_of_examples-num_full_batches*self.batch_size,self.n_points, self.gaussian_dimensions)})")
            sample = model.sample_stochastic(
                shape = (self.number_of_examples-num_full_batches*self.batch_size,self.n_points, self.gaussian_dimensions),
                context=None if self.contexts is None else self.contexts[-1],
                rng=None,
            )
            samples.append(sample.cpu().numpy())
        return np.concatenate(samples, axis=0)

    def _assemble_dist_m(self, ss_dist, sd_dist):
        dd_dist = self.dd_dist
        ds_dist = sd_dist.T

        return np.concatenate(
            [
                np.concatenate([ss_dist, sd_dist], axis=1),
                np.concatenate([ds_dist, dd_dist], axis=1),
            ],
            axis=0,
        )

    def _one_nn_acc(self, ss_dist, sd_dist):
        dist_m = self._assemble_dist_m(ss_dist, sd_dist)

        n = ss_dist.shape[0]
        np.fill_diagonal(dist_m, float("inf"))

        amin = dist_m.argmin(axis=0)
        one_nn_1 = amin[:n] <= n
        one_nn_2 = amin[n:] > n

        return np.concatenate([one_nn_1, one_nn_2]).mean()

    def _mmd(self, sd_dist):
        return sd_dist.min(axis=0).min()

    def _cov(self, sd_dist):
        return np.unique(sd_dist.argmin(axis=1)).size / sd_dist.shape[1]

    def _distance_hist(self, ss_dist, sd_dist):
        dd_dist = self.dd_dist

        fig, ax = plt.subplots(tight_layout=True)
        kw = dict(
            histtype="step",
            bins=np.linspace(0, dd_dist.max() * 1.3, 20),
        )
        ax.hist(dd_dist.flatten(), color="r", label="data-data", **kw)
        ax.hist(ss_dist.flatten(), color="b", label="sample-sample", **kw)
        ax.hist(sd_dist.flatten(), color="g", label="sample-data", **kw)
        fig.legend()

        return fig

    def _plot_dist_m(self, ss_dist, sd_dist):
        dist_m = self._assemble_dist_m(ss_dist, sd_dist)
        dist_inf = dist_m + np.diag(np.ones(dist_m.shape[0]) * float("inf"))

        fig, ax = plt.subplots(tight_layout=True, figsize=(6, 6))
        ax.imshow(dist_inf, vmax=self.dd_dist.max())
        ax.set_xticks([ss_dist.shape[0]])
        ax.set_yticks([ss_dist.shape[0]])
        return fig

    def call_without_logging(self, samples):
        ss_dist = self.distance_fn(samples, samples)
        print(f"call without logging shape samples {samples.shape}, data shape: {self.data.shape}")
        sd_dist = self.distance_fn(samples, self.data)

        one_nn_acc = self._one_nn_acc(ss_dist, sd_dist)
        mmd = self._mmd(sd_dist)
        cov = self._cov(sd_dist)

        histogram = self._distance_hist(ss_dist, sd_dist)
        dist_m_fig = self._plot_dist_m(ss_dist, sd_dist)

        scalars = {
            f"{self.tag_prefix}/1-nn-acc/{self.distance_fn_name}": one_nn_acc,
            f"{self.tag_prefix}/mmd/{self.distance_fn_name}": mmd,
            f"{self.tag_prefix}/cov/{self.distance_fn_name}": cov,
        }

        plots = {
            f"{self.tag_prefix}/histograms/{self.distance_fn_name}": histogram,
            f"{self.tag_prefix}/dist-mat/{self.distance_fn_name}": dist_m_fig,
        }

        return scalars, plots

    def __call__(
        self,
        model,
        logger,
        log_fun,
        epoch: int,
    ):
        print("forward __call__ benchmark")
        samples = self.sample_from_model(model)
        print(f"sampled from model: {samples.shape}")
        if np.isnan(samples).any():
            print("benchmark NA in samples, return")
            return
        else:
            # samples = np.concatenate(samples, axis=0)
            path_samples = pathlib.Path(self.save_path,f"epoch_{epoch}")
            path_samples.mkdir(parents=True,exist_ok=True)
            with open(pathlib.Path(path_samples,"benchmark_vertices.npz"), "wb") as f:
                np.savez(f, vertices=samples)
            with open(pathlib.Path(path_samples,"benchmark_gt_vertices.npz"), "wb") as f:
                np.savez(f, vertices=self.data)
                
            if not self.conditional:
                # unconditional
                print(f"benchmark FORWARD shape samples {samples.shape}")
                scalars, plots = self.call_without_logging(samples)

                for key, value in scalars.items():
                    log_fun(key,value,on_step=False, on_epoch=True)
                    # logger.experiment.add_scalar(key, scalar_value=value, global_step=epoch)
                    # log(key,value, on_step=False, on_epoch=True)

                for key, value in plots.items():
                    # logger.log(key,value, on_step=False, on_epoch=True)
                    # logger.experiment.add_figure(key, figure=value, global_step=epoch)
                    wandb_fig = wandb.Image(value)
                    wandb.log({key:[wandb_fig]})

                if self.save_path is None:
                    return
                _1nn_tag = f"{self.tag_prefix}/1-nn-acc/{self.distance_fn_name}"
                _1nn_score = scalars[_1nn_tag]
                if not _1nn_score < self.lowest_1nn:
                    return
                print(f"{_1nn_score} improves over {self.lowest_1nn} at {_1nn_tag}.")
                self.lowest_1nn = _1nn_score

            elif self.conditional:
                all_categories = []
                for ctx in self.contexts:
                    all_categories.extend(ctx.insinfo.category)
                gecco_chamfer_l1 = []
                gecco_chamfer_l2 = []
                gecco_chamfer_xyz_l1 = []
                gecco_chamfer_xyz_l2 = []
                chamfer_formula_xyz_l1 = []
                dict_chamfer_formula_smart_l1 = {}


                """
                samples hat shape (num_validation_samples, 4000,14), also NICHT unbedingt die gleiche len wie self.contexts
                """
                flat_contexts = []
                for i in range(len(self.contexts)):
                    for j in range(self.contexts[i].camera.projection_matrix.shape[0]):
                        flat_contexts.append(
                            GaussianContext3d(
                                image=self.contexts[i].image[j],
                                K = self.contexts[i].K[j],
                                c2w=self.contexts[i].c2w[j],
                                w2c=self.contexts[i].w2c[j],
                                camera = Camera(
                                            world_view_transform=self.contexts[i].camera.world_view_transform[j], # shape 4,4
                                            projection_matrix=self.contexts[i].camera.projection_matrix[j], # shape 4,4
                                            tanfovx=self.contexts[i].camera.tanfovx[j], #scalar
                                            tanfovy=self.contexts[i].camera.tanfovy[j], #scalar
                                            imsize=self.contexts[i].camera.imsize[j]
                                        ), #scalar
                                splatting_cameras = None,
                                mask_points = self.contexts[i].mask_points[j],
                                insinfo = InsInfo(
                                    category = self.contexts[i].insinfo.category[j],
                                    instance = self.contexts[i].insinfo.instance[j],
                                ),
                        )
                        )
                one_big_context = GaussianContext3d(
                    image = torch.concat([self.contexts[i].image for i in range(len(self.contexts))]),
                    K = torch.concat([self.contexts[i].K for i in range(len(self.contexts))]),
                    c2w = torch.concat([self.contexts[i].c2w for i in range(len(self.contexts))]),
                    w2c = torch.concat([self.contexts[i].w2c for i in range(len(self.contexts))]),
                    camera = Camera(
                        world_view_transform = torch.concat([self.contexts[i].camera.world_view_transform for i in range(len(self.contexts))]),
                        projection_matrix = torch.concat([self.contexts[i].camera.projection_matrix for i in range(len(self.contexts))]),
                        tanfovx = torch.concat([self.contexts[i].camera.tanfovx for i in range(len(self.contexts))]),
                        tanfovy = torch.concat([self.contexts[i].camera.tanfovy for i in range(len(self.contexts))]),
                        imsize = torch.concat([self.contexts[i].camera.imsize for i in range(len(self.contexts))]),
                    ),
                    splatting_cameras = None,
                    mask_points = torch.concat([self.contexts[i].mask_points for i in range(len(self.contexts))]),
                    insinfo = InsInfo(
                        category=[self.contexts[i].insinfo.category[j] for i in range(len(self.contexts)) for j in range(len(self.contexts[i].insinfo.category))],
                        instance=[self.contexts[i].insinfo.instance[j] for i in range(len(self.contexts)) for j in range(len(self.contexts[i].insinfo.instance))],
                    )
                    
                )
                """
                dadurch ist contexts flat geworden und jedes element hat nur 2 dimensionen z.b.
                camera.projection_matrix.shape = (4,4)
                """
                # np.savez(pathlib.Path("/home/giese/Documents/gecco","samples.npz"),samples)
                # np.savez(pathlib.Path("/home/giese/Documents/gecco","gt.npz"),self.data)
                use_numpy_based_method = False
                if np.isnan(samples).any():
                    use_numpy_based_method = False
                else:
                    distances_chamfer_naive_l1 = chamfer_distance_naive_occupany_networks(samples[:,:,:3], self.data[:,:,:samples.shape[-1]][:,:,:3], norm=1)
                    distances_chamfer_naive_l2 = chamfer_distance_naive_occupany_networks(samples[:,:,:3], self.data[:,:,:samples.shape[-1]][:,:,:3], norm=2)
                    # da kommt immer das gleiche raus
                    # distances_chamfer_kdtree_l1 = chamfer_distance_kdtree_occupancy_network(samples[:,:,:3], self.data[:,:,:samples.shape[-1]][:,:,:3], norm=1)
                    # distances_chamfer_kdtree_l2 = chamfer_distance_kdtree_occupancy_network(samples[:,:,:3], self.data[:,:,:samples.shape[-1]][:,:,:3], norm=2)


                splatting_losses = []
                ssims = []
                lpipss = []
                psnrs = []
                """
                Samples shape (val_size, 4000, 13/14)
                für jedes einzelne sample wollen wir alle splatting_cameras durchgehen
                """
                for i, sample in enumerate(samples):
                    for splat_cam_idx in range(len(self.contexts[int(i / self.batch_size)].splatting_cameras)):
                        rendered_image_dict = self.render_fn(sample[None,...], self.contexts[int(i / self.batch_size)], self.mode, splat_cam_idx)
                        rendered_image = rendered_image_dict['render'].squeeze(0)
                        gt_image = self.contexts[int(i / self.batch_size)].splatting_cameras[splat_cam_idx][1][i % self.batch_size]
                        # save_image(gt_image, f"gt_img_{splat_cam_idx}.png")
                        loss = l1_loss(rendered_image, gt_image)
                        ssims.append(self.ssim_fn(torch.clip(rendered_image.unsqueeze(0),min=0,max=1), gt_image.unsqueeze(0)).item())
                        lpipss.append(self.lpips_fn_net(rendered_image, gt_image).item())
                        psnrs.append(psnr(rendered_image, gt_image).mean().item())
                        splatting_losses.append(float(loss.detach().cpu()))


                chamfer_worked = True
                # rotational_distances = []
                images_dict = self.render_fn(samples,one_big_context,self.mode)
                images = images_dict['render']
                splatting_losses_ctx_image = []
                ssims_ctx = []
                lpipss_ctx = []
                psnrs_ctx = []
                for i,(sample,gt,context) in enumerate(zip(samples,self.data,flat_contexts)):
                    # gecco_chamfer_xyz_l1.append(chamfer_distance(sample[:,0:3],gt[:,0:3]))
                    # gecco_chamfer_xyz_l2.append(chamfer_distance_squared(sample[:,0:3],gt[:,0:3]))
                    try:
                        d_cd_smart_l1 = chamfer_formula_smart_l1_np(sample[:,0:3],gt[:,0:3])
                        chamfer_formula_xyz_l1.append(d_cd_smart_l1)
                        # print(f"i {i} cond. chamf distance shape sample {sample.shape}")
                        # cd = chamfer_distance(sample,gt)
                        # print(f"cd {cd}")
                        # gecco_chamfer_l1.append(cd)
                        # cd2 = chamfer_distance_squared(sample,gt)
                        # print(f"cd squared {cd2}")
                        # gecco_chamfer_l2.append(cd2)

                        category = all_categories[i]
                        if category not in dict_chamfer_formula_smart_l1:
                            dict_chamfer_formula_smart_l1[category] = []
                        dict_chamfer_formula_smart_l1[category].append(d_cd_smart_l1)
                    except Exception as e:
                        print(e)
                        chamfer_worked = False
                    
                    # if not Mode.cholesky in self.mode:
                    #     if Mode.so3_diffusion in self.mode:
                    #         sample_rotations = torch.from_numpy(sample[:,10:14]).to(dtype=torch.float32).cuda()
                    #         gt_rotations = torch.from_numpy(gt[:,10:14]).to(dtype=torch.float32).cuda()
                    #     else:
                    #         sample_rotations = torch.from_numpy(sample[:,9:13]).to(dtype=torch.float32).cuda()
                    #         gt_rotations = torch.from_numpy(gt[:,9:13]).to(dtype=torch.float32).cuda()

                        # sum_distance = rotational_distance(sample_rotations, gt_rotations, rotational_distance_between_pairs_dot_product)
                        # rotational_distances.append(sum_distance)

                    rendered_image = images[i] # render_dict['render']
                    loss = l1_loss(rendered_image,context.image)
                    splatting_losses_ctx_image.append(float(loss.detach().cpu()))
                    ssims_ctx.append(self.ssim_fn(torch.clip(rendered_image.unsqueeze(0),min=0,max=1), context.image.unsqueeze(0)).item())
                    lpipss_ctx.append(self.lpips_fn_net(rendered_image, context.image).item())
                    psnrs_ctx.append(psnr(rendered_image, context.image).mean().item())

                    # imageio.imsave(f"/home/giese/Documents/gecco/rend/val_rend/e_{str(self.epoch).zfill(2)}_{str(i).zfill(3)}_loss_{loss.detach().cpu().numpy()}.png",(255*rendered_image).type(torch.uint8).permute(1,2,0).cpu().numpy())
                    # imageio.imsave(f"/home/giese/Documents/gecco/rend/val_gt/e_{str(self.epoch).zfill(2)}_{str(i).zfill(3)}.png",(255*context.image).type(torch.uint8).permute(1,2,0).cpu().numpy())

                if use_numpy_based_method:
                    # mean_distance_kdtree_l1 = sum(distances_chamfer_kdtree_l1)/len(distances_chamfer_kdtree_l1)
                    # log_fun('meanchamferdistance_kdtree_l1',float(mean_distance_kdtree_l1), on_step=False, on_epoch=True)
                    # mean_distance_kdtree_l2 = sum(distances_chamfer_kdtree_l2)/len(distances_chamfer_kdtree_l2)
                    # log_fun('meanchamferdistance_kdtree_l2',float(mean_distance_kdtree_l2), on_step=False, on_epoch=True)
                    mean_distance_naive_l1 = sum(distances_chamfer_naive_l1)/len(distances_chamfer_naive_l1)
                    log_fun('meanchamferdistance_naive_l1',float(mean_distance_naive_l1), on_step=False, on_epoch=True)
                    mean_distance_naive_l2 = sum(distances_chamfer_naive_l2)/len(distances_chamfer_naive_l2)
                    log_fun('meanchamferdistance_naive_l2',float(mean_distance_naive_l2), on_step=False, on_epoch=True)


                # mean_distance_xyz = sum(gecco_chamfer_xyz_l1)/len(gecco_chamfer_xyz_l1)
                # mean_distance_squared_xyz = sum(gecco_chamfer_xyz_l2)/len(gecco_chamfer_xyz_l2)
                # log_fun('mean gecco chamfer xyz l1',float(mean_distance_xyz), on_step=False, on_epoch=True)
                # log_fun('mean gecco chamfer xyz l2',float(mean_distance_squared_xyz), on_step=False, on_epoch=True)

                if chamfer_worked:
                    for key in dict_chamfer_formula_smart_l1:
                        dict_chamfer_formula_smart_l1[key] = sum(dict_chamfer_formula_smart_l1[key])/len(dict_chamfer_formula_smart_l1[key])
                    for key in dict_chamfer_formula_smart_l1:
                        print(f"{dict_chamfer_formula_smart_l1[key]}: {dict_chamfer_formula_smart_l1[key]}")

                    # print(f"Eval conditional, chamfer distances: {gecco_chamfer_l1}")
                    # print(f"Eval conditional, chamfer distances square: {gecco_chamfer_l2}")
                    
                    # mean_distance = sum(gecco_chamfer_l1)/len(gecco_chamfer_l1)
                    # mean_distance_squared = sum(gecco_chamfer_l2)/len(gecco_chamfer_l2)
                    # print(f"mean chamfer gecco all l1: {mean_distance}")
                    # print(f"mean chamfer gecco all l2: {mean_distance_squared}")
                    # logger.log('mean chamfer distance',np.asarray(mean_distance), on_step=False, on_epoch=True)
                    # log_fun('mean chamfer gecco all l1',float(mean_distance), on_step=False, on_epoch=True)
                    # log_fun('mean chamfer gecco all l2',float(mean_distance_squared), on_step=False, on_epoch=True)

                    mean_distance_formula_xyz = sum(chamfer_formula_xyz_l1)/len(chamfer_formula_xyz_l1)
                    log_fun('mean gecco chamfer formula xyz l1',float(mean_distance_formula_xyz), on_step=False, on_epoch=True)
                else:
                    print("Chamfer formula didnt work")

                mean_splatting_loss_ctx_image = sum(splatting_losses_ctx_image)/len(splatting_losses_ctx_image)
                log_fun('mean splatting loss benchmark ctx',float(mean_splatting_loss_ctx_image), on_step=False, on_epoch=True)
                mean_psnr_ctx = sum(psnrs_ctx)/len(psnrs_ctx)
                log_fun('mean psnr benchmark ctx',float(mean_psnr_ctx), on_step=False, on_epoch=True)
                mean_lpips_ctx = sum(lpipss_ctx)/len(lpipss_ctx)
                log_fun('mean lpips benchmark ctx',float(mean_lpips_ctx), on_step=False, on_epoch=True)
                mean_ssim_ctx = sum(ssims_ctx)/len(ssims_ctx)
                log_fun('mean ssim benchmark ctx',float(mean_ssim_ctx), on_step=False, on_epoch=True)

                mean_splatting_loss = sum(splatting_losses)/len(splatting_losses)
                print(f"Mean splattign loss: {mean_splatting_loss}")
                log_fun('benchmark_mean_splatting_loss',float(mean_splatting_loss), on_step=False, on_epoch=True)
                mean_psnr = sum(psnrs)/len(psnrs)
                print(f"Mean psnr: {mean_psnr}")
                log_fun('benchmark_mean_psnr',float(mean_psnr), on_step=False, on_epoch=True)
                mean_lpips = sum(lpipss)/len(lpipss)
                log_fun('benchmark_mean_lpips',float(mean_lpips), on_step=False, on_epoch=True)
                print(f"Mean lpips: {mean_lpips}")
                mean_ssim = sum(ssims)/len(ssims)
                log_fun('benchmark_mean_ssim',float(mean_ssim), on_step=False, on_epoch=True)
                print(f"Mean ssim: {mean_ssim}")

                # combined
                mean_splatting_loss_combined = (sum(splatting_losses_ctx_image)+sum(splatting_losses))/(len(splatting_losses_ctx_image)+len(splatting_losses))
                log_fun('mean splatting loss benchmark combined',float(mean_splatting_loss_combined), on_step=False, on_epoch=True)
                mean_psnr_combined = (sum(psnrs_ctx)+sum(psnrs))/(len(psnrs_ctx)+len(psnrs))
                log_fun('mean psnr benchmark combined',float(mean_psnr_combined), on_step=False, on_epoch=True)
                mean_lpips_combined = (sum(lpipss_ctx)+sum(lpipss))/(len(lpipss_ctx)+len(lpipss))
                log_fun('mean lpips benchmark combined',float(mean_lpips_combined), on_step=False, on_epoch=True)
                mean_ssim_combined = (sum(ssims_ctx)+sum(ssims))/(len(ssims_ctx)+len(ssims))
                log_fun('mean ssim benchmark combined',float(mean_ssim_combined), on_step=False, on_epoch=True)
                
                # if not Mode.cholesky in self.mode:  
                #     log_fun('avg_rotational_distance',float(sum(rotational_distances)/len(rotational_distances)), on_step=False, on_epoch=True)
                return mean_lpips_combined