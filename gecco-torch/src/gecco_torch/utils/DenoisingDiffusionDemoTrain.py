
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import copy
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import torch
import jaxlie
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_plotting import visualize_so3_probabilities, visualize_so3_density

from torch import nn, optim

@jax.jit
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
# Rs = sample_checkerboard(jax.random.split(jax.random.PRNGKey(0), 1024))
# Rs = torch.from_numpy(np.array(Rs))

# This will decide our noise schedule, to start with, here I use a linear noise schedule in 
# variance, based on the Variance Exploding SDE (note, slightly diffrent from the choices of Ho et al.)
# This is not necessarily the best choice, this is just for testing
delta = 0.01 # delta in noise variance
noise_schedule = jnp.arange(0.01,0.9, delta) # Noise variance


def batch_convert_rotations_to_quaternions(R):
    # R is expected to be a batch of rotation matrices with shape (batch_size, 3, 3)
    
    # Initialize the Rotation object from the batch of matrices
    rotation = Rotation.from_matrix(R)
    
    # Convert all rotations to quaternions (xyzw convention)
    quaternions = rotation.as_quat()  # This will have shape (batch_size, 4)
    
    # Convert the numpy array of quaternions to a PyTorch tensor
    quaternions_tensor = torch.from_numpy(quaternions).type(torch.float32)
    
    return quaternions_tensor

def sample(R, scale):
    # r shape 512,3,3
    rots = batch_convert_rotations_to_quaternions(R)
    
    # print(f"rotations x shape {rots.shape}")
    # x = lietorch.SO3.InitFromVec(rots)
    # print(x)
    # print(f"scale shape {scale.val.shape}")
    # Sampling from current temperature
    dist = IsotropicGaussianSO3(rots, scale)
    qn = dist.sample()

    # zurück zur wxyz darstellung
    # qn_wxyz = qn[:,[3,0,1,2]]

    
    # Sampling from next temperature step 
    scale_torch = torch.sqrt(delta * torch.ones(R.shape[0],device=qn.device))
    dist2 = IsotropicGaussianSO3(qn, scale_torch)
    qnplus1 = dist2.sample()

    # zurück zur wxyz darstellung
    # qnplus1_wxyz = qnplus1[:,[3,0,1,2]]
    
    # zurück zur wxyz darstellung
    # x_wxyz = x.vec()[:,[3,0,1,2]]

    res = {'x': rots, 'yn': qn, 'yn+1': qnplus1,
            'sn':scale, 'sn+1':torch.sqrt(scale**2 + delta)}
    return res

def get_batch(seed, batch_size=512):
    key1, key2, key3 = jax.random.split(seed,3)
    # Sample from the target distribution
    Rs = sample_checkerboard(jax.random.split(key1, batch_size))
    # Rs = torch.from_numpy(np.array(Rs))
    
    # Sample random noise levels from the schedule
    s = jax.random.choice(key3, noise_schedule, shape=[batch_size])
    s = torch.from_numpy(np.array(s)).sqrt()
    
    # Sample random rotations
    Rs = np.array(Rs)
    samp = sample(Rs, s)
    # samp['yn'] = samp['yn'][-1]
    # samp['yn+1'] = samp['yn+1'][-1]
    # samp['x'] = samp['x'][-1]
    return samp

losses = []

class SimpleMLP(nn.Module):
    def __init__(self,input_size=5):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.layer_mu = nn.Linear(256, 4)
        self.layer_scale = nn.Linear(256, 1)

    def forward(self, x, s):
        concat = torch.cat([x, s], dim=-1)
        out = self.mlp(concat)
        mu = self.layer_mu(out) + x
        mu_normed = mu / (torch.norm(mu, dim=-1, keepdim=True)+1e-7)

        scale = self.layer_scale(out)
        scale = nn.functional.softplus(scale) + 0.001
        return mu_normed, scale

model = SimpleMLP().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

rng_seq = hk.PRNGSequence(42)

def train(model, batch):
    model.train()
    optimizer.zero_grad()
    # yn1 = torch.from_numpy(np.array(batch['yn+1']))
    # sn1 = torch.from_numpy(np.array(batch['sn+1']))
    yn1 = batch['yn+1'].cuda()
    sn1 = batch['sn+1'].cuda()

    mu, scale = model(yn1, sn1.reshape(-1,1))

    def fn(x, mu, scale):

        # von der wxyz zur lietorch xyzw darstellung
        # mu = mu[:,[1,2,3,0]]
        dist = IsotropicGaussianSO3(mu, scale, 
                                    force_small_scale=True)

        # logprob auch in der lietorch xyzw darstellung
        # prob_dist = dist.log_prob(x[:,[1,2,3,0]]) 
        prob_dist = dist.log_prob(x) 
        return prob_dist # shape 512
    
    yn = batch['yn']
    loss = (-fn(yn, mu, scale)).mean()
    loss.backward()
    optimizer.step()
    return loss.item()

# for step in tqdm(range(100000)):
for step in tqdm(range(20000)):
    batch = get_batch(seed=next(rng_seq))
    loss = train(model, batch)
    losses.append(loss)
    # if step%100 == 0:
    print(loss)
    if loss == min(losses):
        print(f"best model with loss {loss}")
        best_model_sd = copy.deepcopy(model.state_dict())
# model.load_state_dict(best_model_sd)

save_path = Path("gecco-torch","src","gecco_torch","utils",str(datetime.datetime.now()).replace(" ","_"))
save_path.mkdir(exist_ok=True,parents=False)

fig,ax = plt.subplots()
ax.plot(losses)
plt.savefig(save_path / "losses.png")

# Try to sample with the trained model
nsamps = 10000
points = sample_checkerboard(jax.random.split(jax.random.PRNGKey(0), nsamps))
points = sample(points, torch.from_numpy(np.array(noise_schedule[-1])).sqrt()*torch.ones(nsamps))

p = visualize_so3_probabilities(
    jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in points['x']]), 
    0.001)
p.savefig(save_path / 'target_dist.png')

# Samples at T=~1 # Nur noise
p = visualize_so3_probabilities(
    jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in points['yn']]),
    0.001);
p.savefig(save_path / 'noisy_dist.png')

def fn_sample(mu,s):
    ig = IsotropicGaussianSO3(mu, s, force_small_scale=True)
    samp = ig.sample()
    return samp

# Running the denoising loop
x = points['yn']

reverse_noise_schedule = torch.from_numpy(np.array(noise_schedule[::-1])).cuda()
with torch.no_grad():
    for i,variance in enumerate(reverse_noise_schedule):
        var = variance.sqrt() * torch.ones((nsamps,1),device=x.device)
        mu, s = model(x, var)
        print(f"{i}, mu shape {mu.shape}, s shape {s.shape}, {mu[0]}")
        x = fn_sample(mu, s)


p = visualize_so3_probabilities(
    jnp.array([jaxlie.SO3(y[[3,0,1,2]].cpu().numpy()).as_matrix() for y in x]),
    0.001);
p.savefig(save_path / 'denoised_dist.png')

p = visualize_so3_density(
    jnp.array([jaxlie.SO3(y[[3,0,1,2]].cpu().numpy()).as_matrix() for y in x]),
    32);
p.savefig(save_path / 'denoised_density.png')