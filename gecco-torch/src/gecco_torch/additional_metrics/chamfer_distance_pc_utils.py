import point_cloud_utils as pcu
import numpy as np

# a and b are arrays where each row contains a point
# Note that the point sets can have different sizes (e.g [100, 3], [111, 3])
a = np.random.rand(100, 3)
b = np.random.rand(100, 3)

A = np.array([[1, 0, 0], [0, 1, 0]],dtype=np.float32)
B = np.array([[0, 0, 0], [1, 1, 0]],dtype=np.float32)

chamfer_dist = pcu.chamfer_distance(A,B)
print(chamfer_dist)