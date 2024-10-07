import torch
import torch.distributions as dist
from lietorch import SO3
import numpy as np
import math

import jax
import jax.numpy as jnp

def torch_interp(x,xp,fp):
    denom = (xp[:,1:] - xp[:,:-1])
    nom = (fp[1:] - fp[:-1])
    # nom = (fp[:,1:] - fp[:,:-1])
    m = nom / denom
    
    isinf = torch.isinf(m)
    
    # Use cumsum to find the first invalid in each row
    invalid_cumsum = isinf.cumsum(dim=1)
    
    # Find the last valid values in each row before the first invalid
    last_valid_values = torch.where(invalid_cumsum == 0, m, torch.tensor(float('nan')).to(m.device))
    last_valid_values = torch.nan_to_num(last_valid_values[:, -1:], nan=0, posinf=0, neginf=0)  # Handle rows with all invalids
    
    # Broadcast the last valid values to the invalid positions
    m = torch.where(isinf, last_valid_values, m)

    b = fp[:-1] - m * xp[:,:-1]
    idx = torch.sum(x.reshape(x.shape[0],1) >= xp,dim=-1) - 1
    idx = torch.clamp(idx, 0, m.shape[1] - 1)
    selected_bs = b[[torch.arange(b.shape[0]), idx]] # b[:,idx]
    selected_ms = m[[torch.arange(m.shape[0]), idx]]
    res = selected_ms * x + selected_bs
    max_values = torch.max(fp)
    min_values = torch.min(fp)
    clamped_tensor = torch.max(torch.min(res, max_values), min_values)
    return clamped_tensor

# def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
#     """One-dimensional linear interpolation for monotonically increasing sample
#     points.

#     Returns the one-dimensional piecewise linear interpolant to a function with
#     given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

#     Args:
#         x: the :math:`x`-coordinates at which to evaluate the interpolated
#             values.
#         xp: the :math:`x`-coordinates of the data points, must be increasing.
#         fp: the :math:`y`-coordinates of the data points, same length as `xp`.

#     Returns:
#         the interpolated values, same size as `x`.
#     """
#     denom = (xp[:,1:] - xp[:,:-1])
#     denom += 1e-7
#     m = (fp[1:] - fp[:-1]) / denom
#     b = fp[:-1] - (m * xp[:,:-1])

#     indicies = torch.sum(torch.ge(x[:, None], xp), 1) - 1 # ge = greater equal >=
#     indicies = torch.clamp(indicies, 0, m.shape[1] - 1)
#     selected_bs = torch.gather(b, 1, indicies.unsqueeze(1)).squeeze(1)[0]
#     selected_ms = torch.gather(m, 1, indicies.unsqueeze(1)).squeeze(1)[0]

#     res = selected_ms * x + selected_bs
#     return res


def _isotropic_gaussian_so3_small(omg, scale):
    # if omg.shape[0] == 512: # das ist nur wenn wir die prob density berechnen so -> dann müssen wir die scale squueezen
    #     scale = scale.squeeze()
    eps = torch.clamp(scale**2,max=47.45)
    small_number = 1e-9
    small_num = small_number / 2
    small_dnm = (1 - torch.exp(-1. * torch.pi**2 / eps) * (2 - 4 * (torch.pi**2) / eps)) * small_number

    prob = 0.5 * torch.sqrt(torch.tensor([torch.pi],device=omg.device)) * (eps ** -1.5) * torch.exp((eps - (omg**2 / eps)) / 4) / (torch.sin(omg/2) + small_num) \
           * (small_dnm + omg - ((omg - 2*torch.pi) * torch.exp(torch.pi * (omg - torch.pi) / eps) + (omg + 2*torch.pi) * torch.exp(-torch.pi * (omg + torch.pi) / eps)))
    return prob

def _isotropic_gaussian_so3(omg, scale, lmax=None):
    eps = scale**2
    if lmax is None:
        lmax = max(int(3. / math.sqrt(eps)), 2)

    small_number = 1e-9
    sum = 0.
    for l in range(lmax + 1):
        sum += (2*l+1) * torch.exp(-l*(l+1) * eps) * (torch.sin((l+0.5)*omg) + (l+0.5)*small_number) / (torch.sin(omg/2) + 0.5*small_number)
    return sum

class IsotropicGaussianSO3(dist.Distribution):
    def __init__(self, loc, scale, force_small_scale=False, validate_args=False):
        super(IsotropicGaussianSO3, self).__init__()    
        if isinstance(loc,torch.Tensor):
            loc = SO3(loc)
            # try:
            #     loc = SO3(torch.tensor(loc))
            # except:
            #     loc = SO3(torch.tensor(loc))
        self._loc = loc.cuda()
        
        if not isinstance(scale, torch.Tensor):
            try:
                scale = torch.from_numpy(np.array(scale.val))
            except:
                scale = torch.from_numpy(np.array(scale))

        self._scale = scale.cuda()
        self._force_small_scale = force_small_scale
        self._x = torch.linspace(0, torch.pi, 1024).cuda()
        y = (1 - torch.cos(self._x)) / torch.pi * self._f(self._x)
        y = torch.cumsum(y, dim=1) * torch.pi / 1024
        self._y = y / (y.max()+1e-7)
        self._y = self._y.cuda()

    def _f(self, angles): # im samplen gehen die angles von 0-pi (mit nem vorfaktor)
        if self._force_small_scale:
            return _isotropic_gaussian_so3_small(angles.reshape(1,1024), self._scale.reshape(-1,1) / torch.sqrt(torch.tensor([2],device = angles.device)))
        mask = (self._scale < 1).reshape(-1)
        prob_dens_angles = torch.zeros((mask.shape[0],1024),device=self._scale.device)
        prob_dens_angles[mask] = _isotropic_gaussian_so3_small(angles.reshape(1,1024), self._scale[mask].reshape((mask>0).sum(),1) / torch.sqrt(torch.tensor([2],device = angles.device)))
        prob_dens_angles[~mask] = _isotropic_gaussian_so3(angles.reshape(1,1024), self._scale[~mask].reshape(((~mask)>0).sum(),1) / torch.sqrt(torch.tensor([2],device = angles.device)),lmax=3)
        return prob_dens_angles
    

    def _prob_density(self, angles):
        """
        Die gleiche Funktion wie _f, aber für den Fall, dass wir nur für die 512 eingangs angles die prob density haben wollen, und nicht
        für die 1024 verschiedenen angles
        """
        if self._force_small_scale or (self._scale < 1).all():
            prob_dens = _isotropic_gaussian_so3_small(angles, (self._scale / torch.sqrt(torch.tensor([2],device = angles.device))).squeeze())
            return prob_dens
        else:
            mask = (self._scale < 1).reshape(-1)
            prob_dens_angles = torch.zeros((mask.shape[0]),device=self._scale.device)
            prob_dens_angles[mask] = _isotropic_gaussian_so3_small(angles[mask], self._scale[mask].reshape(-1) / torch.sqrt(torch.tensor([2],device = angles.device)))
            prob_dens_angles[~mask] = _isotropic_gaussian_so3(angles[~mask], self._scale[~mask].reshape(-1) / torch.sqrt(torch.tensor([2],device = angles.device)),lmax=5)
            return prob_dens_angles
        # else:
        #     raise NotImplementedError("_prob_density für nicht small scale nicht implementiert ")
        # mask = self._scale < 1
        # prob_dens_angles = torch.zeros((mask.shape[0]),device=self._scale.device)
        # prob_dens_angles[mask] = _isotropic_gaussian_so3_small(angles[mask], self._scale[mask] / torch.sqrt(torch.tensor([2],device = angles.device)))
        # print(f"prob dens angles nan 1 {torch.isnan(prob_dens_angles).sum()}")
        # prob_dens_angles[~mask] = _isotropic_gaussian_so3(angles[~mask], self._scale[~mask] / torch.sqrt(torch.tensor([2],device = angles.device)),lmax=3)
        # print(f"prob dens angles nan 2 {torch.isnan(prob_dens_angles).sum()}")
        # return prob_dens_angles

    def prob(self, q):
        angles = self._get_angles(q)
        prob_dens = self._prob_density(angles)
        return prob_dens

    def log_prob(self, q):
        angles = self._get_angles(q)
        prob_dens = self._prob_density(angles)
        o = torch.log(prob_dens+1e-9)
        # print(f"o nan: {torch.isnan(o).sum()}")
        return o

    def _get_angles(self, q):
        # q = q.unsqueeze(0) if q.ndim == 1 else q
        angles = torch.norm((self._loc.inv() * SO3(q)).log(), dim=-1)
        return angles

    # def sample(self,rand_angle,axis):
    #     angle = torch_interp(rand_angle,self._y,self._x)
    #     axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    #     axis_angle = angle[..., None] * axis
    #     quats = (self._loc * SO3.exp(axis_angle)).vec() # als quaternion xyzw
    #     # quats = quats[:,[3,0,1,2]]
    #     return quats
    
    # def sample(self,seed):
    #     n = self._scale.shape[0]
    #     key1, key2 = jax.random.split(seed)
    #     rand_angle = jax.random.uniform(shape=[n], key=key1) # zwischen 0 und 1
    #     rand_angle = torch.from_numpy(np.array(rand_angle)).cuda()
    #     angle = torch_interp(rand_angle,self._y,self._x)
    #     axis = jax.random.normal(shape=[n,3], key=key2)
    #     axis = torch.from_numpy(np.array(axis)).cuda()
    #     axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    #     axis_angle = angle[..., None] * axis
    #     quats = (self._loc * SO3.exp(axis_angle)).vec() # als quaternion xyzw
    #     # quats = quats[:,[3,0,1,2]]
    #     return quats
    
    # def sample(self,seed):
    #     n = self._scale.shape[0]
    #     key1, key2 = jax.random.split(seed)
    #     rand_angle = jax.random.uniform(shape=[n], key=key1) # zwischen 0 und 1
    #     rand_angle = torch.from_numpy(np.array(rand_angle)).cuda()
    #     angle = torch_interp(rand_angle,self._y,self._x)
    #     axis = jax.random.normal(shape=[n,3], key=key2)
    #     axis = torch.from_numpy(np.array(axis)).cuda()
    #     axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    #     axis_angle = angle[..., None] * axis
    #     quats = (self._loc * SO3.exp(axis_angle)).vec() # als quaternion xyzw
    #     # quats = quats[:,[3,0,1,2]]
    #     return quats
    

    def sample(self):
        n = self._scale.shape[0]
        rand_angle = torch.rand(n).cuda() # uniform sample
        rand_angle = jnp.array(rand_angle.cpu().numpy())
        angle = jax.vmap(lambda x, xp: jnp.interp(x,xp,jnp.array(self._x.cpu().numpy())))(rand_angle,jnp.array(self._y.cpu().numpy()))
        # angle = torch_interp(rand_angle,self._y,self._x)
        angle = torch.from_numpy(np.array(angle)).cuda()
        axis = torch.randn(n, 3).cuda()
        axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        axis_angle = angle[..., None] * axis
        quats = (self._loc * SO3.exp(axis_angle)).vec() # als quaternion xyzw
        # quats = quats[:,[3,0,1,2]]
        return quats
    
    def sample_jaxseed(self,seed):
        n = self._scale.shape[0]
        key1, key2 = jax.random.split(seed)
        rand_angle = jax.random.uniform(shape=[n], key=key1) # zwischen 0 und 1
        rand_angle = torch.from_numpy(np.array(rand_angle)).cuda()

        angle = torch_interp(rand_angle,self._y,self._x)

        axis = jax.random.normal(shape=[n,3], key=key2) # sample axis from S2 -> indem man 3 normalverteilte werte normiert,
                                                        # punkt ist auf sphere if x^2 + y^2 + z^2 = 1
        axis = torch.from_numpy(np.array(axis)).cuda()
        axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        axis_angle = angle[..., None] * axis
        quats = (self._loc * SO3.exp(axis_angle)).vec() # als quaternion xyzw
        # quats = quats[:,[3,0,1,2]]
        return quats
    
    def sample_one(self,seed):
        key1, key2 = jax.random.split(seed)
    

    # def sample_jaxseed(self,seed):
    #     n = self._scale.shape[0]
    #     key1, key2 = jax.random.split(seed)
    #     rand_angle = jax.random.uniform(shape=[n], key=key1) # zwischen 0 und 1
    #     rand_angle = torch.from_numpy(np.array(rand_angle)).cuda()

    #     angle = torch_interp(rand_angle,self._y,self._x)

    #     axis = jax.random.normal(shape=[n,3], key=key2) # sample axis from S2 -> indem man 3 normalverteilte werte normiert,
    #                                                     # punkt ist auf sphere if x^2 + y^2 + z^2 = 1
    #     axis = torch.from_numpy(np.array(axis)).cuda()
    #     axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    #     axis_angle = angle[..., None] * axis
    #     quats = (self._loc * SO3.exp(axis_angle)).vec() # als quaternion xyzw
    #     # quats = quats[:,[3,0,1,2]]
    #     return quats

if __name__ == "__main__":
    # SO3.exp(torch.tensor([0., 0., 0.]))
    # ig = IsotropicGaussianSO3(SO3.exp(torch.tensor([0., 0., 0.])), torch.tensor([0.1]))
    unit_quats = torch.zeros((1024,4))
    unit_quats[:,3] = 1

    scales = torch.ones(1024)*0.1 # geht nur bis 0.001
    ig = IsotropicGaussianSO3(unit_quats,scales)
    axis_angles = torch.linspace(0,1,1025).cuda()[:-1] # 1 darf nicht drin sein
    # axis = torch.randn(1024,3).cuda()
    axis = torch.linspace(-2,2,1024).repeat_interleave((3)).reshape(1024,3).cuda()
    x = ig.sample()
    # ig.log_prob(unit_quats.cuda())

    scales = torch.ones(2)*0.3
    bsp = torch.tensor([[0.0425, 0.3292, 0.0716,0.9406],[0.0425, 0.3292, 0.0716, 0.9406]]).reshape(2,4)
    ig = IsotropicGaussianSO3(bsp,scales,force_small_scale=True)
    ig.sample()
    # print(ig.log_prob(unit_quats.cuda()))
    # print(ig.log_prob(bsp.cuda()))

    # ig = IsotropicGaussianSO3(torch.tensor([[-0.02953,0.1977,0.1447,0.969],[-0.02953,0.1977,0.1447,0.969]]),torch.tensor([[0.3],[0.3]]))
    # c = torch.tensor([[-0.00934,0.021,0.41,0.91],[-0.00934,0.021,0.41,0.91]])
    # ig.log_prob(c)
    a = 3