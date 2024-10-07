import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_quaternion


def rotation_matrix_to_quaternion_pytorch3d(R):
    # input of shape (N,3,3)
    q = matrix_to_quaternion(R)
    return q

    # pytorch3d ist der real part vorne (Also so wie wir es brauchen)

def rotation_matrix_to_quaternion_scipy(R):
    t = R.dtype
    batch_size = R.shape[0]
    dev = R.device
    R = R.reshape(-1,3,3)
    r = Rotation.from_matrix(R.cpu().numpy())
    quats = r.as_quat()
    quats = torch.from_numpy(quats).to(dev).type(t)
    quats = quats.reshape(batch_size,4)
    # scipy stellt quaternion als x,y,z,w dar, aber gaussian splatting braucht es als w,x,y,z
    # quats = quats[:,[3,0,1,2]]
    quats = torch.roll(quats,shifts=1,dims=-1)
    return quats

def quaternion_to_rotation_matrix_scipy(q: np.array):
    # scipy hat quaternion standard x,y,z,w, aber wir haben die rotations als w,x,y,z
    q = q[:,[1,2,3,0]]
    r = Rotation.from_quat(q)
    return r.as_matrix()

def rotation_matrix_to_quaternion_numpy(R):
    T = np.trace(R)
    if T > 0:
        w = 0.5 * np.sqrt(1 + T)
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        x = 0.5 * np.sqrt(1 + 2 * R[0, 0] - T)
        w = (R[2, 1] - R[1, 2]) / (4 * x)
        y = (R[0, 1] + R[1, 0]) / (4 * x)
        z = (R[0, 2] + R[2, 0]) / (4 * x)
    elif R[1, 1] > R[2, 2]:
        y = 0.5 * np.sqrt(1 + 2 * R[1, 1] - T)
        w = (R[0, 2] - R[2, 0]) / (4 * y)
        x = (R[0, 1] + R[1, 0]) / (4 * y)
        z = (R[1, 2] + R[2, 1]) / (4 * y)
    else:
        z = 0.5 * np.sqrt(1 + 2 * R[2, 2] - T)
        w = (R[1, 0] - R[0, 1]) / (4 * z)
        x = (R[0, 2] + R[2, 0]) / (4 * z)
        y = (R[1, 2] + R[2, 1]) / (4 * z)
    return np.array([w, x, y, z])

def mult_quats_vectorized_numpy(q1,q2):
    a0, a1, a2, a3 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    b0, b1, b2, b3 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    # Compute the product of quaternions element-wise
    w = a0*b0 - a1*b1 - a2*b2 - a3*b3
    x = a0*b1 + a1*b0 + a2*b3 - a3*b2
    y = a0*b2 - a1*b3 + a2*b0 + a3*b1
    z = a0*b3 + a1*b2 - a2*b1 + a3*b0
    
    # Stack the components together to form the output quaternions
    result = np.stack((w, x, y, z), axis=1)
    return result



def rotation_matrix_to_quaternion_torch_batched(R):
    """
    R is of shape [batch size, 3, 3]
    """
    batch_size = R.shape[0]
    qw = torch.empty(batch_size,device=R.device)
    qx = torch.empty(batch_size,device=R.device)
    qy = torch.empty(batch_size,device=R.device)
    qz = torch.empty(batch_size,device=R.device)
    
    T = torch.einsum('bii->b', R)  # Compute the trace for each matrix in the batch
    greater_than_zero = T > 0
    
    # Case 1: T > 0
    qw[greater_than_zero] = 0.5 * torch.sqrt(1 + T[greater_than_zero])
    denom = 4 * qw[greater_than_zero]
    qx[greater_than_zero] = (R[greater_than_zero, 2, 1] - R[greater_than_zero, 1, 2]) / denom
    qy[greater_than_zero] = (R[greater_than_zero, 0, 2] - R[greater_than_zero, 2, 0]) / denom
    qz[greater_than_zero] = (R[greater_than_zero, 1, 0] - R[greater_than_zero, 0, 1]) / denom
    
    # Case 2: R[0, 0] is the max diagonal element
    cond = (~greater_than_zero) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    qx[cond] = 0.5 * torch.sqrt(1 + 2 * R[cond, 0, 0] - T[cond])
    denom = 4 * qx[cond]
    qw[cond] = (R[cond, 2, 1] - R[cond, 1, 2]) / denom
    qy[cond] = (R[cond, 0, 1] + R[cond, 1, 0]) / denom
    qz[cond] = (R[cond, 0, 2] + R[cond, 2, 0]) / denom
    
    # Case 3: R[1, 1] is the max diagonal element
    cond = (~greater_than_zero) & (R[:, 1, 1] > R[:, 2, 2])
    qy[cond] = 0.5 * torch.sqrt(1 + 2 * R[cond, 1, 1] - T[cond])
    denom = 4 * qy[cond]
    qw[cond] = (R[cond, 0, 2] - R[cond, 2, 0]) / denom
    qx[cond] = (R[cond, 0, 1] + R[cond, 1, 0]) / denom
    qz[cond] = (R[cond, 1, 2] + R[cond, 2, 1]) / denom
    
    # Case 4: R[2, 2] is the max diagonal element
    cond = (~greater_than_zero) & (~cond)
    qz[cond] = 0.5 * torch.sqrt(1 + 2 * R[cond, 2, 2] - T[cond])
    denom = 4 * qz[cond]
    qw[cond] = (R[cond, 1, 0] - R[cond, 0, 1]) / denom
    qx[cond] = (R[cond, 0, 2] + R[cond, 2, 0]) / denom
    qy[cond] = (R[cond, 1, 2] + R[cond, 2, 1]) / denom
    
    return torch.stack([qw, qx, qy, qz], dim=1)

def mult_quats_vectorized_torch_batched(q1, q2):
    # Assuming q1 and q2 have shapes (batch_size, num_quaternions, 4), num_quaternions ist normal 4000
    a0, a1, a2, a3 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    b0, b1, b2, b3 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Compute the product of the quaternions element-wise across the batches
    w = a0*b0 - a1*b1 - a2*b2 - a3*b3
    x = a0*b1 + a1*b0 + a2*b3 - a3*b2
    y = a0*b2 - a1*b3 + a2*b0 + a3*b1
    z = a0*b3 + a1*b2 - a2*b1 + a3*b0
    
    # Stack the components together to form the output quaternions
    result = torch.stack((w, x, y, z), dim=-1)
    return result

if __name__ == "__main__":
    q1 = np.random.rand(4000,4)
    q2 = np.random.rand(4000,4)
    q3 = np.random.rand(4000,4)
    q4 = np.random.rand(4000,4)

    q1q2 = mult_quats_vectorized_numpy(q1, q2)
    print(q1q2)

    q1_torch = torch.from_numpy(q1).unsqueeze(0)
    q2_torch = torch.from_numpy(q2).unsqueeze(0)
    q3_torch = torch.from_numpy(q3).unsqueeze(0)
    q4_torch = torch.from_numpy(q4).unsqueeze(0)
    q12 = torch.concat((q1_torch,q2_torch))
    q34 = torch.concat((q3_torch,q4_torch))
    
    q1q2_torch = mult_quats_vectorized_torch_batched(q1_torch, q2_torch)
    print(q1q2_torch)
    

    rot = torch.randn((48,4000,3,3))
    a = rotation_matrix_to_quaternion_scipy(rot)

    b = torch.tensor([1,2,3,4])
    print(b[[3,0,1,2]])