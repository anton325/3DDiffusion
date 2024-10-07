import lietorch
import torch


def quaternion_to_lietorch_tangential(q: torch.Tensor) -> torch.Tensor:
    # ändere reinfolge des quaternion -> wir stellen es als (w,x,y,z) dar, aber lietorch als (x,y,z,w)
    q = q[:, [1, 2, 3, 0]]
    # SO3 object
    R = lietorch.SO3.InitFromVec(torch.from_numpy(q))

    # so3 representation (tangential space) -> aber keine lietorch.SO3 objekt mehr

    t = R.log()

    return t.cpu().numpy()

def batched_lietorch_tangential_to_quaternion(t: torch.Tensor) -> torch.Tensor:
    """
    Wir nehmen an, dass die Eingabe t batched ist, also sowas wie (batch, 4000, 3)
    """
    # zurück von tangentialraum zu SO3
    R = lietorch.SO3.exp(t.double())

    # zurück zu quaternion
    q = R.vec()
    # ändere reinfolge des quaternion -> wir stellen es als (w,x,y,z) dar, aber lietorch als (x,y,z,w)
    q = q[:, :, [3, 0, 1, 2]]
    return q

def batched_lietorch_tangential_to_rotation_matrix(t: torch.Tensor) -> torch.Tensor:
    """
    Wir nehmen an, dass die Eingabe t batched ist, also sowas wie (batch, 4000, 3)
    """
    # zurück von tangentialraum zu SO3
    R = lietorch.SO3.exp(t.float())

    # zurück zu rotation matrix
    mat = R.matrix()
    # translation abschneiden
    mat = mat[:,:,:3,:3]
    return mat

if __name__ == "__main__":
    # quats = torch.randn(10, 4)
    # quats2 = quats.clone()
    # lie = [0 for i in range(10)]
    # for i in range(quats.shape[0]):
    #     # quats[i] = quats[i] / torch.norm(quats[i])
    #     lie[i] = quaternion_to_lietorch_tangential(quats[i])
    # quats2 = quaternion_to_lietorch_tangential(quats2)
    # print(lie)
    # quats2 = quats2 + torch.ones_like(quats2)
    # print(quats2)
    rot_tang = torch.tensor([-1000,-1000,-1000]).reshape(1,1,3)
    print(batched_lietorch_tangential_to_quaternion(rot_tang))

