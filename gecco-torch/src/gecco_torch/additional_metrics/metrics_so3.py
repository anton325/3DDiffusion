from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

import numpy as np
import jax 
import jaxlie 
import jax.numpy as jnp
import lietorch
from scipy.optimize import linear_sum_assignment
import torch


def geodesic_distance(L,K):
    lower_triangle_dif = L[:,3:6] - K[:,3:6]
    frobenius_lower_triangle_dif = torch.square(lower_triangle_dif).sum(dim = -1) # frobenius norm einer jeden einzelnen L matrix

    log_diag_dif = torch.log(L[:,0:3]) - torch.log(K[:,0:3])
    frobenius_lower_triangle_diff = torch.square(log_diag_dif).sum(dim = -1)

    distance = torch.sqrt(frobenius_lower_triangle_diff + frobenius_lower_triangle_dif)
    return distance

def geodesic_distance_log_L(L,K):
    lower_triangle_dif = L[:,3:6] - K[:,3:6]
    frobenius_lower_triangle_dif = torch.square(lower_triangle_dif).sum(dim = -1) # frobenius norm einer jeden einzelnen L matrix

    log_diag_dif = L[:,0:3] - K[:,0:3]
    frobenius_lower_triangle_diff = torch.square(log_diag_dif).sum(dim = -1)

    distance = torch.sqrt(frobenius_lower_triangle_diff + frobenius_lower_triangle_dif)
    return distance


def geodesic_distance_matrix(log_L_1,log_L_2):
    """
    log_L_1 and log_L_2 shape (batch,6)
    """
    assert log_L_1.shape[0] == log_L_2.shape[0], "batch size must be the same"
    n = log_L_1.shape[0]
    log_L_1 = log_L_1.repeat_interleave(log_L_1.shape[0],dim=0) # es wird erst die erste zeile so oft wie nötig wiederholt, dann die zweite, etc.
    log_L_2 = log_L_2.repeat(log_L_2.shape[0],1) # das ganze ding wird wiederholt 
    """ ich habe bewusst bestimmt welches zeilenweise und welches ganz repeated wird, damit wir dann die entstehende matrix nicht transponieren müssen"""

    distances = geodesic_distance(log_L_1,log_L_2)

    d_matrix = distances.view(n,n)

    return d_matrix

def euclidean_distance_matrix(data_1, data_2):
    """
    data_1 and data_2 shape (batch,3)
    """
    assert data_1.shape[0] == data_2.shape[0], "batch size must be the same"
    n = data_1.shape[0]
    data_1 = data_1.repeat_interleave(data_1.shape[0],dim=0) # es wird erst die erste zeile so oft wie nötig wiederholt, dann die zweite, etc.
    data_2 = data_2.repeat(data_2.shape[0],1) # das ganze ding wird wiederholt 
    """ ich habe bewusst bestimmt welches zeilenweise und welches ganz repeated wird, damit wir dann die entstehende matrix nicht transponieren müssen"""

    distances = torch.norm(data_1 - data_2, dim = -1)

    d_matrix = distances.view(n,n)

    return d_matrix

"""
Berechne geodesic distance for die beiden log L's (einmal gesampled, einmal gt). Also wir berechnen paarweise alle, und dann matchen wir
die beiden, die jeweils die geringste Distanz haben. Aber es ist ein bijektives matching
"""
def best_fit_geodesic_distance(sampled_L, gt_L, return_indices = False):
    if sampled_L.shape[0] < gt_L.shape[0]:
        gt_L = gt_L[:sampled_L.shape[0]]

    elif gt_L.shape[0] < sampled_L.shape[0]:
        sampled_L = sampled_L[:gt_L.shape[0]]
    d_mat = geodesic_distance_matrix(gt_L, sampled_L)

    row_ind, col_ind = linear_sum_assignment(d_mat.cpu().numpy(),maximize=False)
    sum_distance = d_mat[row_ind, col_ind].sum()
    if return_indices:
        return sum_distance, row_ind, col_ind   
    else:
        return sum_distance

def best_fit_euclidean_distance(sampled_euclid, gt_euclid, return_indices = False):
    if sampled_euclid.shape[0] < gt_euclid.shape[0]:
        gt_euclid = gt_euclid[:sampled_euclid.shape[0]]

    elif gt_euclid.shape[0] < sampled_euclid.shape[0]:
        sampled_euclid = sampled_euclid[:gt_euclid.shape[0]]
    d_mat = euclidean_distance_matrix(gt_euclid, sampled_euclid)

    row_ind, col_ind = linear_sum_assignment(d_mat.cpu().numpy(),maximize=False)
    sum_distance = d_mat[row_ind, col_ind].sum()
    if return_indices:
        return sum_distance, row_ind, col_ind   
    else:
        return sum_distance

def c2st(X,Y,seed,n_folds, down_sample = True, down_sample_len = 5_000 ):
    """Binary classifier with 2 hidden layers of 10x dim each, 
    following the architecture of Benchmarking Simulation-Based Inference 
    https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py
    Parameters
        ----------
        X: First sample.
        Y: Second sample.
        seed: Seed for sklearn.
        n_folds: Number of folds. 
    Returns
    ----------
        Score
    """

    if X.shape[0] > down_sample_len:
        rand_idx = np.random.randint(0, high=min(X.shape[0],Y.shape[1]), size = min(min(down_sample_len,X.shape[0]),Y.shape[1]))
        X = X[rand_idx]
        Y = Y[rand_idx]
    
    X = jax.vmap(lambda m: jaxlie.SO3(m).log()  )(X) # print(X.shape)
    Y = jax.vmap(lambda m: jaxlie.SO3(m).log()  )(Y)
 
    ndim = X.shape[1]
 
    
    clf = MLPClassifier(
    activation="relu",
    hidden_layer_sizes=(10 * ndim, 10 * ndim),
    max_iter=1000,
    solver="adam",
    random_state=seed,
                       )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring="accuracy")
    print(scores)
    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return scores

def c2st_gaussian(X,Y,seed,n_folds, down_sample = True, down_sample_len = 5_000 ):
    """Binary classifier with 2 hidden layers of 10x dim each, 
    following the architecture of Benchmarking Simulation-Based Inference 
    https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py
    Parameters
        ----------
        X: First sample.
        Y: Second sample.
        seed: Seed for sklearn.
        n_folds: Number of folds. 
    Returns
    ----------
        Score
    """

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()

    if X.shape[0] > down_sample_len:
        rand_idx = np.random.randint(0, high=min(X.shape[0],Y.shape[1]), size = min(min(down_sample_len,X.shape[0]),Y.shape[1]))
        X = X[rand_idx]
        Y = Y[rand_idx]

    ndim = X.shape[1]
 
    
    clf = MLPClassifier(
    activation="relu",
    hidden_layer_sizes=(10 * ndim, 10 * ndim),
    max_iter=1000,
    solver="adam",
    random_state=seed,
                       )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring="accuracy")
    print(f"scores in c2st : {scores}")
    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return scores


def rotational_distance_between_pairs(rotations1, rotations2):
    if type(rotations1) != lietorch.groups.SO3 and type(rotations2) != lietorch.groups.SO3:
        rotations1 = lietorch.SO3(rotations1)
        rotations2 = lietorch.SO3(rotations2)
    dif = rotations1.inv() * rotations2
    trace = dif.matrix()[:,:3,:3].diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1),-1,1) / 2)
    return angle

def rotational_distance_between_pairs_dot_product(rotations1, rotations2):
    dot_prod = torch.sum(rotations1*rotations2,dim=-1)
    clamped = torch.clamp(dot_prod.abs(),-1,1)
    angle = 2 * torch.acos(clamped)
    return angle

def distance_matrix_wrong(rotations1,rotations2):
    """
    rotations1 and rotations2 shape (batch,4), in xyzw quaternion convention
    """
    assert rotations1.shape[0] == rotations2.shape[0], "batch size must be the same"
    n = rotations1.shape[0]
    rotations1 = rotations1.repeat_interleave(rotations1.shape[0],dim=0) # es wird erst die erste zeile so oft wie nötig wiederholt, dann die zweite, etc.
    rotations2 = rotations2.repeat(rotations2.shape[0],1) # das ganze ding wird wiederholt 
    """ ich habe bewusst bestimmt welches zeilenweise und welches ganz repeated wird, damit wir dann die entstehende matrix nicht transponieren müssen"""
    angle = rotational_distance_between_pairs(rotations1,rotations2)
    d_matrix = angle.view(n,n)

    return d_matrix

def distance_matrix(rotations1,rotations2,distance_function):
    """
    rotations1 and rotations2 shape (batch,4), in xyzw quaternion convention
    """
    assert rotations1.shape[0] == rotations2.shape[0], "batch size must be the same"
    n = rotations1.shape[0]
    rotations1 = rotations1.repeat_interleave(rotations1.shape[0],dim=0) # es wird erst die erste zeile so oft wie nötig wiederholt, dann die zweite, etc.
    rotations2 = rotations2.repeat(rotations2.shape[0],1) # das ganze ding wird wiederholt 
    """ ich habe bewusst bestimmt welches zeilenweise und welches ganz repeated wird, damit wir dann die entstehende matrix nicht transponieren müssen"""
    angle = distance_function(rotations1,rotations2)

    d_matrix = angle.view(n,n)
    return d_matrix

"""
compute rotational distance for rotations in xyzw batch format (not in lietorch format, but in vec format)
somit ist es nur wichtig, dass beide die gleiche quaternion convention haben 
"""
def minimum_distance(x_t, gt_rotations,distance_function):
    if x_t.shape[0] < gt_rotations.shape[0]:
        gt_rotations = gt_rotations[:x_t.shape[0]]
    elif gt_rotations.shape[0] < x_t.shape[0]:
        x_t = x_t[:gt_rotations.shape[0]]
    d_mat = distance_matrix(gt_rotations, x_t, distance_function)

    row_ind, col_ind = linear_sum_assignment(d_mat.cpu().numpy(),maximize=False)
    sum_distance = d_mat[row_ind, col_ind].sum()
    return sum_distance


def sanity_rotational_distance():
    unit_rot = torch.tensor([0.,0.,0.,1.]).reshape(1,4)
    rot1 = lietorch.SO3(unit_rot)
    rot2 = lietorch.SO3(unit_rot)
    assert minimum_distance(rot1.vec(),rot2.vec()) == 0, "every distance between itsself should be 0"

    ninty = jaxlie.SO3.from_rpy_radians(pitch=jnp.pi/2, yaw=0, roll=0).wxyz # 90 degree rotation -> pi/2
    forty5 = jaxlie.SO3.from_rpy_radians(pitch=jnp.pi/4, yaw=0, roll=0).wxyz # 45 degree rotation -> pi/4
    ninty = torch.tensor(np.array(ninty)).reshape(1,4)[:,[1,2,3,0]]
    forty5 = torch.tensor(np.array(forty5)).reshape(1,4)[:,[1,2,3,0]]
    assert minimum_distance(rot1.vec(),ninty) == torch.pi/2, "90 degree rotation should be pi/2"
    assert torch.isclose(minimum_distance(rot1.vec(),forty5),torch.tensor([torch.pi/4])), "45 degree rotation should be pi/2"
    assert torch.isclose(minimum_distance(ninty,forty5),torch.tensor([torch.pi/4])), "45 degree rotation difference should be pi/2"

    negforty5 = jaxlie.SO3.from_rpy_radians(pitch=-jnp.pi/4, yaw=0, roll=0).wxyz # 45 degree rotation -> pi/4
    negforty5 = torch.tensor(np.array(negforty5)).reshape(1,4)[:,[1,2,3,0]]
    assert minimum_distance(negforty5,forty5) == torch.pi/2, "two 45 degree rotations in opposite directions should be pi/2"


if __name__ == "__main__":
    # sanity_rotational_distance()
    two_rotations = lietorch.SO3([],from_uniform_sampled=4).vec()
    print(rotational_distance_between_pairs(two_rotations[:1],two_rotations[1:]))
    print(rotational_distance_between_pairs_dot_product(two_rotations[:2],two_rotations[2:]))
