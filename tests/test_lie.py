import numpy as np

class SO3:
    """Special orthogonal group for 3D rotations.

    Internal parameterization is `(qw, qx, qy, qz)`. Tangent parameterization is
    `(omega_x, omega_y, omega_z)`.
    """

    def __init__(self, wxyz: np.ndarray):
        self.wxyz = wxyz

    def __repr__(self) -> str:
        wxyz = np.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    @staticmethod
    def from_quaternion_xyzw(xyzw: np.ndarray):
        """Construct a rotation from an `xyzw` quaternion.

        Note that `wxyz` quaternions can be constructed using the default constructor.

        Args:
            xyzw: xyzw quaternion. Shape should be (4,).

        Returns:
            Output.
        """
        assert xyzw.shape == (4,)
        return SO3(np.roll(xyzw, shift=1))

    def as_quaternion_xyzw(self) -> np.ndarray:
        """Grab parameters as xyzw quaternion."""
        return np.roll(self.wxyz, shift=-1)

    
    def as_matrix(self) -> np.ndarray:
        norm = np.dot(self.wxyz, self.wxyz)
        q = self.wxyz * np.sqrt(2.0 / norm)
        q = np.outer(q, q)
        return np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )
    
    @staticmethod
    def from_matrix(matrix: np.ndarray):
        assert matrix.shape == (3, 3), "Matrix shape must be (3, 3)"

        def case0(m):
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = np.array([
                m[2, 1] - m[1, 2],
                t,
                m[1, 0] + m[0, 1],
                m[0, 2] + m[2, 0],
            ])
            return t, q

        def case1(m):
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = np.array([
                m[0, 2] - m[2, 0],
                m[1, 0] + m[0, 1],
                t,
                m[2, 1] + m[1, 2],
            ])
            return t, q

        def case2(m):
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = np.array([
                m[1, 0] - m[0, 1],
                m[0, 2] + m[2, 0],
                m[2, 1] + m[1, 2],
                t,
            ])
            return t, q

        def case3(m):
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = np.array([
                t,
                m[2, 1] - m[1, 2],
                m[0, 2] - m[2, 0],
                m[1, 0] - m[0, 1],
            ])
            return t, q

        # Compute four cases, then pick the most precise one.
        case0_t, case0_q = case0(matrix)
        case1_t, case1_q = case1(matrix)
        case2_t, case2_q = case2(matrix)
        case3_t, case3_q = case3(matrix)

        cond0 = matrix[2, 2] < 0
        cond1 = matrix[0, 0] > matrix[1, 1]
        cond2 = matrix[0, 0] < -matrix[1, 1]

        t, q = np.select(
            [cond0 & cond1, cond0 & ~cond1, ~cond0 & cond2, ~cond0 & ~cond2],
            [case0_t, case1_t, case2_t, case3_t],
            default=case3_t
        ), np.select(
            [cond0 & cond1, cond0 & ~cond1, ~cond0 & cond2, ~cond0 & ~cond2],
            [case0_q, case1_q, case2_q, case3_q],
            default=case3_q
        )

        q = q * 0.5 / np.sqrt(t)

        return SO3(wxyz=q)


q = np.array([1.3478957, -0.11576728, 0.026394425, 0.024374295])
rot = SO3.from_quaternion_xyzw(q)
print(rot.as_matrix())
rot_matrix = rot.as_matrix()

print(SO3.from_matrix(rot_matrix))