import torch
import numpy as np


def change_all_axis_viewing_direction_single():
    matrix = np.array([[-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    return matrix

def change_x_axis_viewing_direction_single():
    matrix = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    return matrix

def switch_y_z_axis_single():
    matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    return matrix

def switch_x_z_axis_single():
    matrix = [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    
    return matrix

def switch_x_y_axis_single():
    matrix = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    return matrix


def rotate_x_axis_single(angle):
    angle = angle * np.pi/2
    matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return matrix

def rotate_y_axi_single(angle):
    angle = angle * np.pi/2
    matrix = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return matrix

def rotate_z_axis_single(angle):
    angle = angle * np.pi/2
    matrix =  np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return matrix

def change_all_axis_viewing_direction(world_matrices):
    matrix = np.array([[-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])

def switch_y_z_axis(world_matrices):
    matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])

def switch_x_z_axis(world_matrices):
    matrix = [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])

def switch_x_y_axis(world_matrices):
    matrix = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])


def rotate_x_axis(world_matrices,angle):
    angle = angle * np.pi/2
    matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])

def rotate_y_axis(world_matrices,angle):
    angle = angle * np.pi/2
    matrix = np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])

def rotate_z_axis(world_matrices,angle):
    angle = angle * np.pi/2
    matrix =  np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.stack([np.matmul(matrix,cam_mat) for cam_mat in world_matrices])


def get_translation(tx,ty,tz):
    """Get the translation matrix for movement in all 3 directions."""
    """
    | 1 0 0 tx |
    | 0 1 0 ty |
    | 0 0 1 tz |
    | 0 0 0 1  |

    """
    matrix = [
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ]
    return np.array(matrix,dtype=np.float32)

def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    # only movement along z axis
    """
    | 1 0 0 tx |
    | 0 1 0 ty |
    | 0 0 1 tz |
    | 0 0 0 1  |

    """
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return torch.tensor(np.array(matrix,dtype=np.float32))

def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(np.array(matrix,dtype=np.float32))

def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(np.array(matrix,dtype=np.float32))


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.from_numpy(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],dtype=np.float32)) @ c2w
    return c2w