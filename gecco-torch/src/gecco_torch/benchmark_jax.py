import os
import math
from typing import Iterable, Optional, Callable, Union,List,Tuple
from functools import partial
import wandb
from scipy.spatial import cKDTree

import pathlib
import jax
import datetime
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
import torch
from tqdm.auto import tqdm
# from torch.utils.tensorboard import SummaryWriter

from gecco_torch.additional_metrics.distance_pointclouds import chamfer_distance_naive_occupany_networks, chamfer_distance_kdtree_occupancy_network,  \
                                                                chamfer_occupancy_networks_correct, chamfer_formula_smart_l1_np

from gecco_torch.metrics import (
    chamfer_distance,
    sinkhorn_emd,
    chamfer_distance_squared,
)

code_to_categoryname = {
    '03046257' : 'clock',
    '03513137' : 'helmet',
    '02992529' : 'smartphone',
    '03991062' : 'pot',
    '03761084' : 'microwave',
    '04379243': 'table',
    '03593526': 'jar',
    '04225987': 'skateboard',
    '02958343': 'car',
    '02876657': 'bottle',
    '04460130': 'tower',
    '03001627': 'chair',
    '02871439': 'bookshelf',
    '02942699': 'camera',
    '02691156': 'airplane',
    '03642806': 'laptop',
    '02801938': 'basket',
    '04256520': 'sofa',
    '03624134': 'knife',
    '02946921': 'can',
    '04090263': 'rifle',
    '04468005': 'train',
    '03938244': 'pillow',
    '03636649': 'lamp',
    '02747177': 'trash bin',
    '03710193': 'mailbox',
    '04530566': 'watercraft',
    '03790512': 'motorbike',
    '03207941': 'dishwasher',
    '02828884': 'bench',
    '03948459': 'pistol',
    '04099429': 'rocket',
    '03691459': 'loudspeaker',
    '03337140': 'file cabinet',
    '02773838': 'bag',
    '02933112': 'cabinet',
    '02818832': 'bed',
    '02843684': 'birdhouse',
    '03211117': 'display',
    '03928116': 'piano',
    '03261776': 'earphone',
    '04401088': 'telephone',
    '04330267': 'stove',
    '03759954': 'microphone',
    '02924116': 'bus',
    '03797390': 'mug',
    '04074963': 'remote',
    '02808440': 'bathtub',
    '02880940': 'bowl',
    '03085013': 'keyboard',
    '03467517': 'guitar',
    '04554684': 'washer',
    '02834778': 'bicycle',
    '03325088': 'faucet',
    '04004475': 'printer',
    '02954340': 'cap'
}
# from gecco_torch.diffusion import Diffusion


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
        batch_size: int = 64,
        tag_prefix: str = "benchmark",
        rng_seed: int = 42,
        block_size: int = 16,
        distance_fn: Union[str, Callable] = chamfer_distance_squared,
        save_path: Optional[str] = None,
    ):

        self.data = [d[0] for d in list_data_and_ctx]
        self.data = torch.cat(self.data,dim=0).cpu().numpy()
        print(f"Create benchmark with data shape {self.data.shape} and batch size {batch_size}")
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
        # key = jax.random.PRNGKey(self.rng_seed)

        # samples = []
        # print(f"sample from model. Number of times we are going to sample: {jax.random.split(key, self.n_batches)}")
        # a = 0
        # for key in tqdm(
        #     jax.random.split(key, self.n_batches), desc="Sampling for benchmark"
        # ):
        #     print(a)
        #     print(key)
        #     a+=1
            
        # for key in tqdm(
        #     jax.random.split(key, self.n_batches), desc="Sampling for benchmark"
        # ):
        #     sample = model.sample_stochastic(
        #         (1,self.n_points, 3),
        #         n=self.batch_size,
        #     )
        #     samples.append(np.asarray(sample))
        # return np.concatenate(samples, axis=0)[: self.data.shape[0]]
        print(f"sample {self.number_of_examples} times from model")
        samples = []
        num_full_batches = int(np.floor(self.number_of_examples/self.batch_size))
        # which_batch = np.random.choice(num_full_batches)
        for i in range(num_full_batches):
            print(f"sample batch {i} from model")
            sample = model.sample_stochastic(
                shape = (self.batch_size,self.n_points, 3),
                context=None if self.contexts is None else self.contexts[i],
                # context=None if self.contexts is None else self.contexts[i*self.batch_size:(i+1)*self.batch_size],
                rng=None,
            )
            samples.append(sample.cpu().numpy())
        if self.number_of_examples-num_full_batches*self.batch_size == 0:
            return np.concatenate(samples, axis=0)
        print(f"sample last batch from model (with shape {(self.number_of_examples-num_full_batches*self.batch_size,self.n_points, 3)})")
        sample = model.sample_stochastic(
            shape = (self.number_of_examples-num_full_batches*self.batch_size,self.n_points, 3),
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
        # samples = np.concatenate(samples, axis=0)
        path_samples = pathlib.Path(self.save_path,f"epoch_{epoch}")
        path_samples.mkdir(parents=True,exist_ok=True)
        with open(pathlib.Path(path_samples,"vertices.npz"), "wb") as f:
            np.savez(f, vertices=samples)
        with open(pathlib.Path(path_samples,"gt_vertices.npz"), "wb") as f:
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
                all_categories.extend(ctx.category)
            # only L1 chamfer distance
            distances_chamfer = []
            dict_category_chamfer_l1 = {}
            dict_category_chamfer_l2 = {}
            distances_chamfer_squared = []
            stupid_cds_smart_l1 = []
            dict_stupid_cds_smart_l1 = {}
            distances_chamfer_naive_l1 = chamfer_distance_naive_occupany_networks(samples,self.data,norm=1)
            distances_chamfer_naive_l2 = chamfer_distance_naive_occupany_networks(samples,self.data,norm=2)
            distances_chamfer_kdtree_l1 = chamfer_distance_kdtree_occupancy_network(samples,self.data,norm=1)
            distances_chamfer_kdtree_l2 = chamfer_distance_kdtree_occupancy_network(samples,self.data,norm=2)
            distances_chamfer_occupancy_original_l1 = []
            distances_chamfer_occupancy_original_l2 = [] 
            for i,(sample,gt) in enumerate(zip(samples,self.data)):
                original_l1, original_l2 = chamfer_occupancy_networks_correct(sample,gt)
                distances_chamfer_occupancy_original_l1.append(original_l1)
                distances_chamfer_occupancy_original_l2.append(original_l2)
                print(f"cond. chamf distance shape sample {sample.shape}")
                d_scd_smart = chamfer_formula_smart_l1_np(sample,gt)
                stupid_cds_smart_l1.append(d_scd_smart)

                distances_chamfer.append(chamfer_distance(sample,gt))
                distances_chamfer_squared.append(chamfer_distance_squared(sample,gt))
                category = all_categories[i]
                if category not in dict_category_chamfer_l1:
                    dict_category_chamfer_l1[category] = []
                if category not in dict_category_chamfer_l2:
                    dict_category_chamfer_l2[category] = []
                if category not in dict_stupid_cds_smart_l1:
                    dict_stupid_cds_smart_l1[category] = []
                dict_category_chamfer_l1[category].append(distances_chamfer_naive_l1[i])
                dict_category_chamfer_l2[category].append(distances_chamfer_naive_l2[i])
                dict_stupid_cds_smart_l1[category].append(d_scd_smart)

            # calculate average per categeroy
            for key in dict_category_chamfer_l1:
                dict_category_chamfer_l1[key] = sum(dict_category_chamfer_l1[key])/len(dict_category_chamfer_l1[key])
            for key in dict_category_chamfer_l2:
                dict_category_chamfer_l2[key] = sum(dict_category_chamfer_l2[key])/len(dict_category_chamfer_l2[key])
            for key in dict_stupid_cds_smart_l1:
                dict_stupid_cds_smart_l1[key] = sum(dict_stupid_cds_smart_l1[key])/len(dict_stupid_cds_smart_l1[key])
            print("Chamfer per category l1:")
            for key in dict_category_chamfer_l1:
                print(f"{code_to_categoryname[key]}: {dict_category_chamfer_l1[key]}")
            print("Chamfer per category l2:")
            for key in dict_category_chamfer_l2:
                print(f"{code_to_categoryname[key]}: {dict_category_chamfer_l2[key]}")
            print("Stupid CD per category l1:")
            for key in dict_stupid_cds_smart_l1:
                print(f"{code_to_categoryname[key]}: {dict_stupid_cds_smart_l1[key]}")
            print(f"Eval conditional, chamfer distances: {distances_chamfer}")
            print(f"Eval conditional, chamfer distances square: {distances_chamfer_squared}")
            # print(f"eval naive: {distances_chamfer_naive}")
            mean_distance = sum(distances_chamfer)/len(distances_chamfer)
            mean_distance_squared = sum(distances_chamfer_squared)/len(distances_chamfer_squared)
            # mean_distance_naive = sum(distances_chamfer_naive)/len(distances_chamfer_naive)
            mean_distance_kdtree_l2 = sum(distances_chamfer_kdtree_l2)/len(distances_chamfer_kdtree_l2)
            mean_distance_kdtree_l1 = sum(distances_chamfer_kdtree_l1)/len(distances_chamfer_kdtree_l1)
            mean_distance_naive_l1 = sum(distances_chamfer_naive_l1)/len(distances_chamfer_naive_l1)
            mean_distance_naive_l2 = sum(distances_chamfer_naive_l2)/len(distances_chamfer_naive_l2)
            mean_stupid_cds_smart_l1 = sum(stupid_cds_smart_l1)/len(stupid_cds_smart_l1)
            log_fun('meanchamferdistancestupid_l1',float(mean_stupid_cds_smart_l1), on_step=False, on_epoch=True)
            distance_chamfer_occupancy_original_l1 = sum(distances_chamfer_occupancy_original_l1)/len(distances_chamfer_occupancy_original_l1)
            log_fun('meanchamferdistancecorrect_l1',float(distance_chamfer_occupancy_original_l1), on_step=False, on_epoch=True)
            distance_chamfer_occupancy_original_l2 = sum(distances_chamfer_occupancy_original_l2)/len(distances_chamfer_occupancy_original_l2)
            log_fun('meanchamferdistancecorrect_l2',float(distance_chamfer_occupancy_original_l2), on_step=False, on_epoch=True)

            print(f"mean chamfer: {mean_distance}") # -> ich glaube, das ist die chamfer distanz, die sie im paper benutzen, aber nicht ganz klar, wieso
                                                    # meine höher ist 
                                                    # Im Paper haben sie als Durchschnitt 0.122 angegeben (mit 2048 points per pointcloud). 
                                                    # Wir haben hier aktuell einen Durchschnitt von 



            print(f"mean chamfer squared: {mean_distance_squared}")
            print(f"mean chamfer kdtree: {distances_chamfer_kdtree_l2}")
            # print(f"mean chamfer naive: {mean_distance_naive}")
            # logger.log('mean chamfer distance',np.asarray(mean_distance), on_step=False, on_epoch=True)
            log_fun('mean chamfer distance',float(mean_distance), on_step=False, on_epoch=True)
            log_fun('mean chamfer distance squared',float(mean_distance_squared), on_step=False, on_epoch=True)
            # log_fun('mean-chamfer-distance-naive',float(mean_distance_naive), on_step=False, on_epoch=True)
            log_fun('meanchamferdistancekdtree_l2',float(mean_distance_kdtree_l2), on_step=False, on_epoch=True)
            log_fun('meanchamferdistancekdtree_l1',float(mean_distance_kdtree_l1), on_step=False, on_epoch=True)
            log_fun('meanchamferdistancenaive_l1',float(mean_distance_naive_l1), on_step=False, on_epoch=True)
            log_fun('meanchamferdistancenaive_l2',float(mean_distance_naive_l2), on_step=False, on_epoch=True)
            # logger.experiment.add_scalar('mean chamfer distance', scalar_value=np.asarray(mean_distance), global_step=epoch)

        if self.save_path is not None:
            eqx.tree_serialise_leaves(f"{self.save_path}/{epoch}.eqx", model)

if __name__ == "__main__":
    gt = np.load("/globalwork/giese/gecco_shapenet/logs/version_449389/meshes/gt_verticies_20.npz")['vertices']
    sample = np.load("/globalwork/giese/gecco_shapenet/logs/version_449389/meshes/verticies_20.npz")['vertices']
    chamfer_distance_squared(sample,gt)
    # pass
    # dataset_path = (pathlib.Path(pathlib.Path.home(),"..","..","globalwork","giese","gecco_shapenet","ShapeNetCore.v2.PC15k"))
    # # pointclouds1 = load_all_pointclouds(pathlib.Path(dataset_path,"02691156","val"))
    # # pointclouds2 = load_all_pointclouds(pathlib.Path(dataset_path,"02691156","test"))
    # limit = 6
    # pointclouds1 = pointclouds1[100:100+limit]
    # pointclouds2 = pointclouds2[100:100+limit]
    # chamfers = np.zeros((limit,limit))
    # for i,p1 in enumerate(pointclouds1):
    #     p1 = p1[:100]
    #     for j,p2 in enumerate(pointclouds2):
    #         p2 = p2[:100]
    #         chamfers[i,j] = chamfer_distance_squared(p1,p2)
    # print(chamfers)
    # # distance_fn = partial(
    # #         batched_pairwise_distance,
    # #         distance_fn=chamfer_distance_squared,
    # #         block_size=1,
    # #     )
    # # print(distance_fn(pointclouds1,pointclouds2))

    # # p1 = np.array([[1,3],[3,3],[4,1]])
    # # p2 = np.array([[2,2],[3,1],[5,1]])

    # p1 = np.array([[0,1,2],[1,-1,1]])
    # p2 = np.array([[2,2,1],[1,0,1]])
    # # p2 = p1 + 1

    # print(chamfer_distance_squared(p1,p2))
    # print(chamfer_distance(p1,p2))

    # p1 = np.array([[0,1,2],[1,-1,1],[2,3,4]])
    # p2 = np.array([[2,2,1],[1,0,1],[0,0,0]]) #-> distance 8
    # print(chamfer_distance_squared(p1,p2))
    # print(chamfer_distance(p1,p2))