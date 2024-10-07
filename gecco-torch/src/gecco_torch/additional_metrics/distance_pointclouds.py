"""
Implements different ways to calculate the distance between two point clouds.
"""
import pathlib
import numpy as np
from scipy.spatial import cKDTree
# from im2mesh.utils.libkdtree import KDTree
# import point_cloud_utils as pcu

import numpy as np
import torch
from tqdm import tqdm
# from gecco_torch.utils.libkdtree import KDTree

def chamfer_formula_smart_l1_np(cloud1, cloud2):
    n = cloud1.shape[0]
    repeated_cloud1 = np.repeat(cloud1, n, axis=0)
    tiled_cloud2 = np.tile(cloud2, (n,1))

    abs_distance = np.sum(np.abs(repeated_cloud1 - tiled_cloud2), axis = -1) # manhatten distance, L1
    dis_matrix = abs_distance.reshape(n,n)
    accuracy = np.sum(dis_matrix.min(axis=0))
    completeness = np.sum(dis_matrix.min(axis=1))
    result = 2 * (accuracy + completeness) / n

    return result

def chamfer_occupancy_networks_correct(sampled_pcs, target_pcs):
    accuracy = chamfer_kdtree(sampled_pcs, target_pcs)
    accuracy2 = accuracy ** 2
    completeness = chamfer_kdtree(target_pcs, sampled_pcs)
    completeness2 = completeness ** 2
    accuracy = accuracy.mean()
    completeness = completeness.mean()

    accuracy2 = accuracy2.mean()
    completeness2 = completeness2.mean()
    chamfer_l1 = (completeness + accuracy) / 2
    chamfer_l2 = (completeness2 + accuracy2) / 2
    return chamfer_l1, chamfer_l2
    
def chamfer_kdtree(pcs1, pcs2):
    kdtree = cKDTree(pcs1,leafsize=16) # 16 ist die von occpancy trees
    dist, idx = kdtree.query(pcs2,p=2)
    return dist

def chamfer_distance_naive_occupany_networks(points1, points2,norm):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    points1 = torch.from_numpy(points1)
    points2 = torch.from_numpy(points2)

    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    if norm == 2:
        distances = (points1 - points2).pow(2).sum(-1)
    elif norm == 1:
        distances = (points1 - points2).abs().sum(-1)
    else:
        raise Exception(f"weder 1 noch 2 war die norm ({norm})")


    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer

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

def chamfer_distance_kdtree_occupancy_network(points1, points2, norm, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    points1 = torch.from_numpy(points1)
    points2 = torch.from_numpy(points2)
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
    if norm == 2:
        # normal
        chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
        chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)
    elif norm == 1:
        chamfer1 = (points1 - points_12).abs().sum(2).mean(1)
        chamfer2 = (points2 - points_21).abs().sum(2).mean(1)
    else:
        raise Exception(f"weder 1 noch 2 war die norm ({norm})")
    

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer
