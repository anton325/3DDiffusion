import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import pathlib

import os
import math
from typing import Iterable, Optional, Callable, Union
from functools import partial

def load_all_pointclouds(folder):
    pointclouds =[]
    for file in folder.glob("*.npy"):
        pointclouds.append(np.load(file))
    return np.array(pointclouds)


def distance_matrix_from_jax_implementation(a,b,squared: bool = False):
    aa = np.einsum("nd,nd->n", a, a)
    bb = np.einsum("md,md->m", b, b)
    ab = np.einsum("nd,md->nm", a, b)

    dist_sqr = aa[:, None] + bb[None, :] - 2 * ab
    # protection against numerical errors resulting in NaN
    dist_sqr = np.maximum(dist_sqr, 0.0)

    if squared:
        return dist_sqr
    else:
        return np.sqrt(dist_sqr)

def chamfer_distance_from_jax_implementation(
    a,
    b,
    squared: bool = False):
    dist_m = distance_matrix_from_jax_implementation(a, b, squared=squared)
    min_a = dist_m.min(axis=0).mean()
    min_b = dist_m.min(axis=1).mean()

    return (min_a + min_b) / 2

def pairwise_distance(a, b, distance_fn):
    """Compute pairwise distances between points in a and b using the specified distance function."""
    assert a.shape == b.shape, "Shapes of a and b must match."
    distances = np.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            distances[i, j] = distance_fn(a[i], b[j])
    return distances


def batched_pairwise_distance(a, b, distance_fn, block_size=32):
    """Compute pairwise distances in batches to manage memory usage."""
    assert a.shape == b.shape, "Shapes of a and b must match."
    n_batches = int(math.ceil(a.shape[0] / block_size))
    distances = []
    for i in tqdm(range(n_batches), desc="Computing pairwise distances"):
        a_batch = a[i*block_size:(i+1)*block_size]
        for j in range(n_batches):
            b_batch = b[j*block_size:(j+1)*block_size]
            distances.append(pairwise_distance(a_batch, b_batch, distance_fn))
    return np.block(distances)


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
        data: np.array,
        batch_size: int = 64,
        tag_prefix: str = "benchmark",
        rng_seed: int = 42,
        block_size: int = 32,
        distance_fn: Union[str, Callable] = chamfer_distance_from_jax_implementation,
        save_path: Optional[str] = None,
    ):
        self.data = data
        self.n_points = self.data.shape[1]
        self.batch_size = batch_size
        self.tag_prefix = tag_prefix
        self.n_batches = int(math.ceil(self.data.shape[0] / self.batch_size))
        self.rng_seed = rng_seed
        self.block_size = block_size

        if isinstance(distance_fn, str):
            distance_fn = {
                "chamfer": chamfer_distance_from_jax_implementation,
                "chamfer_squared": chamfer_distance_from_jax_implementation,
                "emd": None,
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


def load_all_pointclouds(folder):
    pointclouds = []
    for file in folder.glob("*.npy"):
        pointclouds.append(np.load(file))
    return np.array(pointclouds)


# Example usage
if __name__ == "__main__":
    # Example usage remains largely the same, with adjustments for any JAX-specific operations removed
    dataset_path = (pathlib.Path(pathlib.Path.home(),"..","..","globalwork","giese","gecco_shapenet","ShapeNetCore.v2.PC15k"))

    pointclouds1 = load_all_pointclouds(pathlib.Path(dataset_path,"02691156","val"))
    pointclouds2 = load_all_pointclouds(pathlib.Path(dataset_path,"02691156","test"))
    pointclouds2 = pointclouds2[:pointclouds1.shape[0]]
    limit = 10
    pointclouds1 = pointclouds1[:limit]
    pointclouds2 = pointclouds2[:limit]

    bm = BenchmarkCallback(pointclouds1)
    print(bm.call_without_logging(pointclouds2))