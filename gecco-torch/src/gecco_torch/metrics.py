from typing import Callable, Dict, Sequence, Tuple
from functools import partial
import jax
import torch
from scipy.spatial import cKDTree

import jax.numpy as jnp
# import equinox as eqx
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_dimcheck import dimchecked, A
# try:
# from gecco_torch.benchmark_jax import Diffusion#, LogpDetails
# except ImportError:
    # pass
from gecco_torch.geometry import distance_matrix
from gecco_torch.gecco_types import PyTree, PRNGKey


class Metric:
    name: str

    def __call__(
        self,
        model,
        data: PyTree,
        raw_ctx: PyTree,
        key: PRNGKey,
    ):
        raise NotImplementedError()


class LossMetric(Metric, eqx.Module):
    loss_scale: float
    name: str = "loss"

    @eqx.filter_jit
    @dimchecked
    def __call__(
        self,
        model,
        data: A["B X*"],
        raw_ctx: PyTree,
        key: A["2"],
    ) -> Dict[str, A[""]]:
        loss = type(model).batch_loss_fn(
            model,
            data,
            raw_ctx,
            key=key,
            loss_scale=self.loss_scale,
        )

        return {"loss": loss}


class LogpMetric(Metric, eqx.Module):
    name: str = "logp"
    n_log_det_jac_samples: int = 1

    @eqx.filter_jit
    @dimchecked
    def __call__(
        self,
        model,
        data: A["B X*"],
        raw_ctx: PyTree,
        key: A["2"],
    ) -> Dict[str, A[""]]:
        @eqx.filter_vmap
        def v_sample_fn(data, raw_ctx, key):
            return model.evaluate_logp(
                data=data,
                raw_ctx=raw_ctx,
                ctx=None,
                return_details=True,
                n_log_det_jac_samples=self.n_log_det_jac_samples,
                key=key,
            )

        keys = jax.random.split(key, data.shape[0])
        details: LogpDetails = v_sample_fn(
            data,
            raw_ctx,
            keys,
        )

        return {
            "total": details.logp,
            "prior": details.prior_logp,
            "det-jac": details.delta_jacobian,
            "reparam": details.delta_reparam,
        }


@dimchecked
def chamfer_distance(
    a: A["N D"],
    b: A["N D"],
    squared: bool = False,
) -> A[""]:
    dist_m = distance_matrix(a, b, squared=squared)
    # print(dist_m.min(axis=0))
    min_a = sum(dist_m.min(axis=0)).mean() # /2
    min_b = sum(dist_m.min(axis=1)).mean() # /2
    # print(min_a)
    # print(min_b)
    return (min_a + min_b)/2 # a.shape[0] # / 2 das geteilt durch zwei muss weg, das steckt schon im mean drin


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        # kdtree = KDTree(p2)
        kdtree = cKDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances

def chamfer_distance_kdtree_occupancy_network(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    # points1 = torch.from_numpy(points1)
    # points2 = torch.from_numpy(points2)
    batch_size = points1.size(0)
    # batch_size = points1.shape[0]
    # points1_np = points1
    # points2_np = points2
    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


def chamfer_distance_naive_occupany_networks(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    if type(points1) == type(np.array([1,2])):
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        points1 = torch.from_numpy(points1)
        points2 = torch.from_numpy(points2)
    elif type(points1) == type(np.array([1.0,2.0])):
        points1 = torch.from_numpy(points1)
        points2 = torch.from_numpy(points2)

    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer

def chamfer_distance_naive_occupany_networks_np(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    if type(points1) == type(torch.tensor([1,2])):
        points1 = points1.cpu().numpy()
        points2 = points2.cpu().numpy()
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
    elif type(points1) == type(torch.tensor([1.0,2.0])):
        points1 = points1.cpu().numpy()
        points2 = points2.cpu().numpy()

    assert(points1.shape == points2.shape)
    batch_size, T, _ = points1.shape

    points1 = points1.reshape(batch_size, T, 1, 3)
    points2 = points2.reshape(batch_size, 1, T, 3)

    distances = np.power((points1 - points2),2).sum(axis=-1)

    chamfer1 = distances.min(axis=1)[0].mean(axis=1)
    chamfer2 = distances.min(axis=2)[0].mean(axis=1)

    chamfer = chamfer1 + chamfer2
    return chamfer

@dimchecked
def chamfer_distance_squared(
    a: A["N D"],
    b: A["N D"],
) -> A[""]:
    return chamfer_distance(a, b, squared=True)


@dimchecked
def _scipy_lsa(cost_matrix: A["N N"]) -> Tuple[A["N"], A["N"]]:
    shape = jnp.zeros(cost_matrix.shape[0], dtype=np.int32)

    def inner(cost_matrix):
        rows, cols = linear_sum_assignment(cost_matrix)
        return rows.astype(np.int32), cols.astype(np.int32)

    return jax.pure_callback(
        inner,
        (shape, shape),
        jax.lax.stop_gradient(cost_matrix),
        vectorized=False,
    )


@dimchecked
def scipy_emd(p1: A["N D"], p2: A["N D"], match="l1", average="l1") -> A[""]:
    match_squared = {"l1": False, "l2": True}[match]
    match_dist = distance_matrix(p1, p2, squared=match_squared)
    rows, cols = _scipy_lsa(match_dist)

    average_squared = {"l1": False, "l2": True}[average]
    if average_squared == match_squared:
        average_dist = match_dist
    else:
        average_dist = distance_matrix(p1, p2, squared=average_squared)

    return average_dist[rows, cols].mean()


@dimchecked
def sinkhorn_emd(
    p1: A["N D"],
    p2: A["N D"],
    epsilon: float = 0.01,
) -> A[""]:
    import ott

    cloud = ott.geometry.pointcloud.PointCloud(p1, p2, epsilon=epsilon)
    ot_prob = ott.problems.linear.linear_problem.LinearProblem(cloud)
    solver = ott.solvers.linear.sinkhorn.Sinkhorn()
    solution = solver(ot_prob)
    return jnp.einsum("ab,ab->", solution.matrix, cloud.cost_matrix)


class _SinkhornEMDMetric:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.__name__ = f"sinkhorn_emd_epsilon_{epsilon}"

    def __call__(self, p1, p2):
        return sinkhorn_emd(p1, p2, epsilon=self.epsilon)


class SupervisedMetric(Metric, eqx.Module):
    name: str = "supervised"
    metrics: Sequence[Callable] = (
        chamfer_distance,
        # _SinkhornEMDMetric(epsilon=0.01),
        # _SinkhornEMDMetric(epsilon=1.0),
    )

    @eqx.filter_jit
    @dimchecked
    def __call__(
        self,
        model,
        data: A["B X*"],
        raw_ctx: PyTree,
        key: A["2"],
    ) -> Dict[str, A[""]]:
        @eqx.filter_vmap
        def v_sample_fn(raw_ctx, key) -> jnp.ndarray:
            return model.sample(x_shape=data.shape[-2:], raw_ctx=raw_ctx, n=1, key=key)

        # v_sample_fn = eqx.filter_vmap(partial(
        #    model.sample,
        #    x_shape=data.shape[-2:],
        #    raw_ctx=raw_ctx,
        #    n=1,
        # ))
        keys = jax.random.split(key, data.shape[0])
        samples = v_sample_fn(raw_ctx, keys)
        samples = samples.squeeze(1)  # n_samples dimension

        results = {}
        for metric in self.metrics:
            results[metric.__name__] = jax.vmap(metric)(samples, data)

        return results


class MetricPmapWrapper(Metric):
    def __init__(self, inner):
        self.inner = inner

    @property
    def name(self):
        return self.inner.name

    def __call__(self, model, xyz, raw_ctx, key):
        keys = jax.random.split(key, jax.device_count())
        keys = jax.device_put_sharded(list(keys), jax.devices())
        values = eqx.filter_pmap(self.inner)(model, xyz, raw_ctx, keys)
        return jax.tree_map(
            lambda array: array.mean(axis=0),
            values,
        )
