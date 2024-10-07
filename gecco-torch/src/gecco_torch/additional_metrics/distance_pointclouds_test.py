"""
Implements different ways to calculate the distance between two point clouds.
"""
import pathlib
import numpy as np
from scipy.spatial import cKDTree
# from im2mesh.utils.libkdtree import KDTree
import time
from functools import partial
# import point_cloud_utils as pcu

import numpy as np
import torch
from tqdm import tqdm
import math
from gecco_torch.metrics import (
    chamfer_distance,
    sinkhorn_emd,
    chamfer_distance_squared,
)

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

def load_all_pointclouds(folder):
    pointclouds =[]
    for file in folder.glob("*.npy"):
        pointclouds.append(np.load(file))
    return np.array(pointclouds)

def batched_pairwise_distance_np(a, b, distance_fn, block_size):
    """
    Returns an N-by-N array of the distance between a[i] and b[j] using only NumPy.
    """
    assert a.shape == b.shape, "Shapes of a and b must be the same."

    n_batches = int(math.ceil(a.shape[0] / block_size))
    distances = []
    for a_block in np.array_split(a, n_batches):
        row = []
        for b_block in np.array_split(b, n_batches):
            # Compute pairwise distances for the current blocks
            # We need to reshape the blocks for broadcasting
            # Reshape a_block to (-1, 1, D) and b_block to (1, -1, D) where D is the dimensionality
            d = distance_fn(a_block[:, np.newaxis, :], b_block[np.newaxis, :, :])
            row.append(d)
        distances.append(np.concatenate(row, axis=1))
    return np.concatenate(distances, axis=0)


def coverage_from_jax_implementation(reference_pointclouds, generated_pointclouds,distance_fn):
    sd_dist = distance_fn(generated_pointclouds, reference_pointclouds)
    return np.unique(sd_dist.argmin(axis=1)).size / sd_dist.shape[1]

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


def coverage(reference_pointclouds, generated_pointclouds,distance_function):
    assert reference_pointclouds.shape[0] == generated_pointclouds.shape[0], "The number of reference and generated pointclouds must be the same."
    list_unique_closest_reference_pointclouds = []
    for i,g in enumerate(generated_pointclouds):
        min_distance = 10000
        min_neigbor_index = 0
        for j,r in enumerate(reference_pointclouds):
            distance = distance_function(g,r)
            if distance < min_distance:
                min_distance = distance
                min_neigbor_index = j
        list_unique_closest_reference_pointclouds.append(min_neigbor_index)
    
    return len(set(list_unique_closest_reference_pointclouds))/reference_pointclouds.shape[0]


def mean_matching_distance(reference_pointclouds, generated_pointclouds,distance_function):
    assert reference_pointclouds.shape[0] == generated_pointclouds.shape[0], "The number of reference and generated pointclouds must be the same."
    list_closest_distances = []
    for i,r in enumerate(reference_pointclouds):
        min_distance = 10000
        for j,g in enumerate(generated_pointclouds):
            distance = distance_function(g,r)
            if distance < min_distance:
                min_distance = distance
        list_closest_distances.append(min_distance)
    
    return sum(list_closest_distances)/reference_pointclouds.shape[0]
    
def H_1nn(Sa,Sb,distance_function):
    number_of_correct_assignments = 0
    for i,u in enumerate(Sa):
        min_distance = 1000
        min_class = None
        union = []
        for j,a in enumerate(Sa):
            if i == j:
                continue
            union.append((a,"a"))
        for j,b in enumerate(Sb):
            union.append((b,"b"))

        for a,cl in union:
            distance = distance_function(u,a)
            if distance < min_distance:
                min_distance = distance
                min_class = cl
        if min_class == "a":
            number_of_correct_assignments += 1
    return number_of_correct_assignments
        
def oneNN_accuracy(reference_pointclouds,generated_pointclouds,distance_function):
    assert reference_pointclouds.shape[0] == generated_pointclouds.shape[0], "The number of reference and generated pointclouds must be the same."
    Hrg = H_1nn(reference_pointclouds,generated_pointclouds,distance_function)
    Hgr = H_1nn(generated_pointclouds,reference_pointclouds,distance_function)

    return (Hrg + Hgr)/(2*reference_pointclouds.shape[0])

def dis_points(p1,p2):
    # dist = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2+ (p1[2]-p2[2])**2
    # dist = np.linalg.norm(p1- p2, axis=-1, ord=2)
    # dist = np.sum((p1-p2)**2)
    dist = np.sum(np.abs((p1-p2))) # L1 chamfer
    return dist

def stupid_cd(cloud1,cloud2):
    num_points = cloud1.shape[0]
    distances1 = []
    for p in tqdm(cloud1):
        min_distance = 1000000
        for p2 in cloud2:
            distance = dis_points(p,p2)
            # print(f"dis {p} {p2} {distance}")
            if distance < min_distance:
                min_distance = distance
        distances1.append(min_distance)

    distances2 = []
    for p2 in tqdm(cloud2):
        min_distance = 1000000
        for p in cloud1:
            distance = dis_points(p,p2)
            if distance < min_distance:
                min_distance = distance
        distances2.append(min_distance)
    # print(distances1)
    # print(distances2)
    ret = (sum(distances2)+sum(distances1))/(num_points)
    # ret = (sum(distances2)+sum(distances1)/len(distances2))/(num_points) # wrong zu viel averagen
    # print(f"ret {ret}")
    return ret

def stupid_cd_smart(cloud1, cloud2):
    n = cloud1.shape[0]
    repeated_cloud1 = np.repeat(cloud1, n, axis=0)
    tiled_cloud2 = np.tile(cloud2, (n,1))

    abs_distance = np.sum(np.abs(repeated_cloud1 - tiled_cloud2), axis = -1)
    dis_matrix = abs_distance.reshape(n,n)
    accuracy = np.sum(dis_matrix.min(axis=0))
    completeness = np.sum(dis_matrix.min(axis=1))
    result = 2 * (accuracy + completeness) / n

    return result
    



# def chamfer_distance(cloud1, cloud2):
#     """
#     Compute the nearest neighbor distance from each point in one cloud to all points in the other cloud and then average these distances. 
#     -> Es gibt auch noch die squared variante
#     """
#     # Create KD-Trees for each cloud
#     tree1 = cKDTree(cloud1)
#     tree2 = cKDTree(cloud2)

#     # For each point in cloud1, find the closest point in cloud2 (and vice versa)
#     dist1, _ = tree1.query(cloud2, k=1)
#     dist2, _ = tree2.query(cloud1, k=1)

#     # Calculate the average of these minimum distances
#     chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2
#     return chamfer_dist

# dataset_path = (pathlib.Path(pathlib.Path.home(),"..","..","globalwork","giese","gecco_shapenet","ShapeNetCore.v2.PC15k"))

# pointcloud1 = np.load(pathlib.Path(dataset_path,"02691156","val","10db820f0e20396a492c7ca609cb0182.npy"))
# pointcloud2 = np.load(pathlib.Path(dataset_path,"02691156","val","72537a008bf0c5e09d1a1149c8c36258.npy"))


# pointclouds1 = load_all_pointclouds(pathlib.Path(dataset_path,"02691156","val"))
# pointclouds2 = load_all_pointclouds(pathlib.Path(dataset_path,"02691156","test"))

# pointclouds2 = pointclouds2[:pointclouds1.shape[0]]

# limit = 5
# pointclouds1 = pointclouds1[100:100+limit]
# pointclouds2 = pointclouds2[100:100+limit]
# chamfers = np.zeros((limit,limit))
# for i,p1 in enumerate(pointclouds1):
#     p1 = p1[:100]
#     for j,p2 in enumerate(pointclouds2):
#         p2 = p2[:100]
#         chamfers[i,j] = stupid_cd(p1,p2)
# print(chamfers)

# pc = pointclouds1[0]
# pc = pc[:5000]
# pc2 = pc.copy()
# for p in pc2:
#     p[0] -= 100000

# # p1 = np.array([[1,3],[3,3],[4,1]])
# # p2 = np.array([[2,2],[3,1],[5,1]])

# p1 = np.array([[0,1,2],[1,-1,1]])
# p2 = np.array([[2,2,1],[1,0,1]]) #-> distance 5.5
# p2 = p1+1

# # print(chamfer_distance(p1,p2))
# print(stupid_cd(p1,p2))

# print(chamfer_distance(pointcloud2, pointcloud1))
# print(chamfer_distance_from_jax_implementation(pointcloud2, pointcloud1))

# print(coverage(pointclouds1,pointclouds2,chamfer_distance))

# print(mean_matching_distance(pointclouds1,pointclouds2,chamfer_distance))

# distance_fn = lambda a,b: batched_pairwise_distance_np(a,b,chamfer_distance_from_jax_implementation,10)

# print(coverage_from_jax_implementation(pointclouds1,pointclouds2,distance_fn))
# print(oneNN_accuracy(pointclouds1,pointclouds2,chamfer_distance))

# Define the point clouds
# A = np.array([[1, 0, 0], [0, 1, 0]])
# B = np.array([[0, 0, 0], [1, 1, 0]]) 

# Calculate Chamfer Distance
# def chamfer_distance(A, B):
#     # For each point in A, find the minimum squared distance to points in B
#     dist_A_to_B = np.sum([np.min([dis_points(a, b) for b in B]) for a in A]) #/ len(A)
    
#     # For each point in B, find the minimum squared distance to points in A
#     dist_B_to_A = np.sum([np.min([dis_points(b, a) for a in A]) for b in B]) #/ len(B)
    
#     # Average these distances
    # return (dist_A_to_B + dist_B_to_A) / 2

# Calculate Chamfer Distance for the example point clouds
# chamfer_dist = chamfer_distance(A, B)
# print(chamfer_dist)
# chamfer_dist = chamfer_distance(p1, p2)
# print(chamfer_dist)
# print(pcu.chamfer_distance(A.astype(np.float32),B.astype(np.float32) ))
# print(pcu.chamfer_distance(p1.astype(np.float32),p2.astype(np.float32),p_norm=2))

# p1 = np.array([[0,1,2],[1,-1,1],[2,3,4]])
# p2 = np.array([[2,2,1],[1,0,1],[0,0,0]]) #-> distance 8
# print(p1.shape)
# print(stupid_cd(p1,p2))

# print(chamfer_distance_naive_occupany_networks(p1[None,...],p2[None,...],1))
# print(chamfer_distance_naive_occupany_networks(p1[None,...],p2[None,...],2))
# print(chamfer_distance_kdtree_occupancy_network(p1[None,...],p2[None,...],1))
# print(chamfer_distance_kdtree_occupancy_network(p1[None,...],p2[None,...],2))

# print(chamfer_distance(p1,p2,squared=False))
# print(chamfer_distance(p1,p2,squared=True))
# gt = np.load("logs/version_444364/meshes/gt_verticies_2.npz")['vertices']
# samples = np.load("logs/version_444364/meshes/verticies_2.npz")['vertices'][:,2048:,:]
# print(chamfer_distance_naive_occupany_networks(gt,samples))
# for g,s in zip(gt,samples):
#     print(stupid_cd(g,s))


# umrechnung chamfer zu gecco chamfer squared: * 2/numpoints
gt = np.load("logs/version_449794/benchmark-checkpoints/chamfer_distance_squared/epoch_30/gt_vertices.npz")['vertices']
samples = np.load("logs/version_449794/benchmark-checkpoints/chamfer_distance_squared/epoch_30/vertices.npz")['vertices']
# print(stupid_cd(gt[0],samples[0]))
# gt = gt[:,:100,:]
# samples = samples[:,:100,:]
# print(chamfer_distance_naive_occupany_networks(gt,samples,1))
# print(chamfer_distance_kdtree_occupancy_network(gt,samples,1))
# print(chamfer_distance_naive_occupany_networks(gt,samples,2))
# print(chamfer_distance_kdtree_occupancy_network(gt,samples,2))
stupids = []
smarts = []
for g,s in zip(gt,samples):
    print(stupid_cd_smart(g,s))
    print(chamfer_distance(g,s,squared=False))
    print(chamfer_distance(g,s,squared=True))
    print(chamfer_distance(g,s,squared=True)*2/gt.shape[1])
    scd =stupid_cd(g,s)
    print(scd)
    stupids.append(scd)
    smart_cd = stupid_cd_smart(g,s)
    print(smart_cd)
    smarts.append(smart_cd)

    print(sum(stupids)/len(stupids))
    print(sum(smarts)/len(smarts))
print(sum(stupids)/len(stupids))

