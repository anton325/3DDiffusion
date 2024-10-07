import lietorch
import numpy as np
import torch

rotations_plane_quats = np.load("/home/giese/Documents/SO3DiffusionModels/rotations_plane.npy")
rotations_plane_quats = torch.from_numpy(rotations_plane_quats) # wxyz convention, für lietorch brauchen wir xyzw
rotations_plane_quats_xyzw = rotations_plane_quats[:,[1,2,3,0]]

matrices = lietorch.SO3.InitFromVec(rotations_plane_quats_xyzw).matrix()[:,:3,:3]

quats_recreated = lietorch.SO3(matrices,from_rotation_matrix=True).data # vec()

close_rows, close_cols = torch.where(~torch.isclose(rotations_plane_quats_xyzw,quats_recreated,rtol=1e-3))

for i in close_rows:
    print(rotations_plane_quats_xyzw[i])
    print(quats_recreated[i])
    print("\n")

samples = lietorch.SO3([],from_uniform_sampled=100)