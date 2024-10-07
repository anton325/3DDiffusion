"""
the lietorch library https://github.com/princeton-vl/lietorch needs to 
be slightly modified in order to work with this project. 
The SO3 class in site-packages/lietorch{...}/lietorch/groups.py
needs to be adjusted as:
"""

class SO3(LieGroup):
    group_name = 'SO3'
    group_id = 1
    manifold_dim = 3
    embedded_dim = 4
    
    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0]) # Darstellung: xyzw

    def __init__(self, data, from_rotation_matrix = False, from_uniform_sampled=0):
        assert not (from_uniform_sampled > 0 and from_rotation_matrix is True), "es geht nicht gleichzeitig von rotation matrix UND von uniform random"                 
        if isinstance(data, SE3):
            data = data.data[..., 3:7]

        if from_rotation_matrix:
            data = self.from_matrix(data)
        
        if from_uniform_sampled > 0:
            # print("Sample uniformly...")
            data = self.from_uniform_sampled(from_uniform_sampled)

        super(SO3, self).__init__(data)

    def from_uniform_sampled(self,batch_size):
        # Uniformly sample over S^3 in batch.
        u1 = torch.rand(batch_size)

        u2 = 2.0 * torch.pi * torch.rand(batch_size)
        u3 = 2.0 * torch.pi * torch.rand(batch_size)
        
        a = torch.sqrt(1.0 - u1)
        b = torch.sqrt(u1)
        
        """
        Das ist der jax code (wxyz standart):
        w = a * torch.sin(u2)
        x = a * torch.cos(u2)
        y = b * torch.sin(u3)
        z = b * torch.cos(u3)
        """

        w = a * torch.sin(u2)
        x = a * torch.cos(u2)
        y = b * torch.sin(u3)
        z = b * torch.cos(u3)

        """
        old
        w = b * torch.cos(u3)
        x = a * torch.cos(u2)
        y = b * torch.sin(u3)
        z = a * torch.sin(u2)
        """

        quats = torch.stack([x, y, z, w], dim=-1) 
        return quats
    
    def from_matrix(self,matrices):
        assert len(matrices.shape) == 3, "Diese Funktion funktioniert nur für batched rotation matrices"
        assert matrices.shape[1] == 3, "Es müssen 3x3 matrizen sein"
        assert matrices.shape[2] == 3, "Es müssen 3x3 matrizen sein"
        
        def case0(m):
            t = 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]
            q = torch.stack([
                m[:, 2, 1] - m[:, 1, 2],
                t,
                m[:, 1, 0] + m[:, 0, 1],
                m[:, 0, 2] + m[:, 2, 0],
            ], dim=-1)
            return t, q

        def case1(m):
            t = 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2]
            q = torch.stack([
                m[:, 0, 2] - m[:, 2, 0],
                m[:, 1, 0] + m[:, 0, 1],
                t,
                m[:, 2, 1] + m[:, 1, 2],
            ], dim=-1)
            return t, q

        def case2(m):
            t = 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]
            q = torch.stack([
                m[:, 1, 0] - m[:, 0, 1],
                m[:, 0, 2] + m[:, 2, 0],
                m[:, 2, 1] + m[:, 1, 2],
                t,
            ], dim=-1)
            return t, q

        def case3(m):
            t = 1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
            q = torch.stack([
                t,
                m[:, 2, 1] - m[:, 1, 2],
                m[:, 0, 2] - m[:, 2, 0],
                m[:, 1, 0] - m[:, 0, 1],
            ], dim=-1)
            return t, q

        # Compute four cases for all batches
        case0_t, case0_q = case0(matrices)
        case1_t, case1_q = case1(matrices)
        case2_t, case2_q = case2(matrices)
        case3_t, case3_q = case3(matrices)

        cond0 = matrices[:, 2, 2] < 0
        cond1 = matrices[:, 0, 0] > matrices[:, 1, 1]
        cond2 = matrices[:, 0, 0] < -matrices[:, 1, 1]

        t = torch.where(
            cond0,
            torch.where(cond1, case0_t, case1_t),
            torch.where(cond2, case2_t, case3_t)
        )
        q = torch.where(
            cond0.unsqueeze(-1),
            torch.where(cond1.unsqueeze(-1), case0_q, case1_q),
            torch.where(cond2.unsqueeze(-1), case2_q, case3_q)
        )

        # Normalize quaternion
        q = q * 0.5 / torch.sqrt(t).unsqueeze(-1)

        # die funktion ist geklaut von jaxlie -> da ist wxyz darstellung. Das wird jetzt korrigiert
        q_xyzw = q[:,[1,2,3,0]]
        return q_xyzw