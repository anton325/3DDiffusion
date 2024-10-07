
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
# import tensorflow_probability as tfp; tfp = tfp.substrates.jax
# tfd = tfp.distributions
# tfb = tfp.bijectors

import torch
import jaxlie
import lietorch 
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_plotting import visualize_so3_probabilities

# @jax.jit
@jax.vmap
def sample_checkerboard(seed, size=np.pi):
    key1, key2, key3 = jax.random.split(seed, 3)
    x1 = jax.random.uniform(key=key1) * size - size/2
    x2_ = jax.random.uniform(key=key2) - jax.random.randint(minval=0, maxval=2,shape=[], key=key3) * 2
    x2 = x2_ + (jnp.floor(x1) % 2)
    data = jnp.stack([x1, x2]) * size/2
    Rs = jaxlie.SO3.from_rpy_radians(pitch=data[0]/2, yaw=data[1], roll=0).as_matrix()
    return Rs

# Create a checkerboard grid
Rs = sample_checkerboard(jax.random.split(jax.random.PRNGKey(0), 1024))

# This will decide our noise schedule, to start with, here I use a linear noise schedule in 
# variance, based on the Variance Exploding SDE (note, slightly diffrent from the choices of Ho et al.)
# This is not necessarily the best choice, this is just for testing
delta = 0.01 # delta in noise variance
noise_schedule = jnp.arange(0.01,0.9, delta) # Noise variance
# @jax.jit
@jax.vmap
def sample(R, scale, seed):
    key1, key2 = jax.random.split(seed)
    x = jaxlie.SO3.from_matrix(R)
    x = x.wxyz.val[:,[1,2,3,0]]
    # print(x.wxyz.val.shape)
    x = torch.from_numpy(np.array(x))
    
    print(f"rotations x shape {x.shape}")
    x = lietorch.SO3.InitFromVec(x)
    print(x)
    print(f"scale shape {scale.val.shape}")
    # Sampling from current temperature
    dist = IsotropicGaussianSO3(x, scale)
    qn = dist.sample()
    qn_wxyz = jnp.array(qn[:,[3,0,1,2]].cpu().numpy())

    
    # Sampling from next temperature step 
    scale_torch = torch.sqrt(delta * torch.ones(512,device=qn.device))
    dist2 = IsotropicGaussianSO3(qn, scale_torch)
    qnplus1 = dist2.sample()
    qnplus1_wxyz = jnp.array(qnplus1[:,[3,0,1,2]].cpu().numpy())
    
    x_wxyz = jnp.array(x.vec()[:,[3,0,1,2]].cpu().numpy())
    return {'x': x_wxyz, 'yn': qn_wxyz, 'yn+1': qnplus1_wxyz, 
            'sn':scale, 'sn+1':jnp.sqrt(scale**2 + delta)}

# @jax.jit
def get_batch(seed, batch_size=512):
    key1, key2, key3 = jax.random.split(seed,3)
    # Sample from the target distribution
    Rs = sample_checkerboard(jax.random.split(key1, batch_size))
    
    # Sample random noise levels from the schedule
    s = jax.random.choice(key3, noise_schedule, shape=[batch_size])
    s = jnp.sqrt(s)
    
    # Sample random rotations
    samp = sample(Rs, s, jax.random.split(key2, batch_size))
    return samp

batch = get_batch(jax.random.PRNGKey(0))
batch['yn'] = batch['yn'][0]
batch['yn+1'] = batch['yn+1'][0]
batch['x'] = batch['x'][0]
a = 34