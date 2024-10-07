
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from gecco_torch.scene.gaussian_model import GaussianModel
import haiku as hk
import torch
import jaxlie
import lietorch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_plotting import visualize_so3_probabilities, visualize_so3_density

from torch import nn, optim

def eval(rotations_plane,noise_schedule,model):
    gt_rots,x = sample_from_model(rotations_plane,noise_schedule,model)
    d_mat = distance_matrix(gt_rots ,x)
    row_ind, col_ind = linear_sum_assignment(d_mat.cpu().numpy(),maximize=False)
    sum_distance = d_mat[row_ind, col_ind].sum()
    return sum_distance

def sample_from_model(rotations_plane,noise_schedule,model):
    # Try to sample with the trained model
    perm = torch.randperm(rotations_plane.size(0))
    idx = perm[:4000]
    rotations_plane = rotations_plane[idx]
    points = rotations_plane
    nsamps = points.shape[0]
    points = sample(rotations_plane, torch.from_numpy(np.array(noise_schedule[-1])).sqrt()*torch.ones(nsamps))
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
            # print(f"{i}, mu shape {mu.shape}, s shape {s.shape}, {mu[0]}")
            x = fn_sample(mu, s)
    return rotations_plane, x

def distance_matrix(rotations1: torch.Tensor,rotations2: torch.Tensor):
    """
    rotations1 and rotations2 shape (batch,4), in xyzw quaternion convention
    """
    assert rotations1.shape[0] == rotations2.shape[0], "batch size must be the same"
    n = rotations1.shape[0]
    rotations1 = rotations1.repeat_interleave(rotations1.shape[0],dim=0) # es wird erst die erste zeile so oft wie nötig wiederholt, dann die zweite, etc.
    rotations2 = rotations2.repeat(rotations2.shape[0],1) # das ganze ding wird wiederholt 
    """ ich habe bewusst bestimmt welches zeilenweise und welches ganz repeated wird, damit wir dann die entstehende matrix nicht transponieren müssen"""
    rotations1_lie = lietorch.SO3(rotations1)
    rotations2_lie = lietorch.SO3(rotations2)
    dif = rotations1_lie.inv() * rotations2_lie
    trace = dif.matrix().diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1),-1,1) / 2)

    d_matrix = angle.view(n,n)

    return d_matrix



gc = GaussianModel(3)
gc.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
rotations_plane = gc.get_rotation.detach() # quats in wxyz
# bringe sie in xyzw, weil das ist womit lietorch später arbeitet
rotations_plane = rotations_plane[:,[1,2,3,0]]


# This will decide our noise schedule, to start with, here I use a linear noise schedule in 
# variance, based on the Variance Exploding SDE (note, slightly diffrent from the choices of Ho et al.)
# This is not necessarily the best choice, this is just for testing
delta = 0.01 # delta in noise variance
noise_schedule = jnp.arange(0.01,0.9, delta) # Noise variance


def sample(R, scale):
    # print(f"rotations x shape {rots.shape}")
    # x = lietorch.SO3.InitFromVec(rots)
    # print(x)
    # print(f"scale shape {scale.val.shape}")
    # Sampling from current temperature
    dist = IsotropicGaussianSO3(R, scale)
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

    res = {'x': R, 'yn': qn, 'yn+1': qnplus1,
            'sn':scale, 'sn+1':torch.sqrt(scale**2 + delta)}
    return res

def get_batch(seed, batch_size=512):
    key1, key2, key3 = jax.random.split(seed,3)
    # Sample from the target distribution
    perm = torch.randperm(rotations_plane.size(0))
    idx = perm[:batch_size]
    Rs = rotations_plane[idx]
    
    # Sample random noise levels from the schedule
    s = jax.random.choice(key3, noise_schedule, shape=[batch_size])
    s = torch.from_numpy(np.array(s)).sqrt()
    
    # Sample random rotations
    samp = sample(Rs, s)
    # samp['yn'] = samp['yn'][-1]
    # samp['yn+1'] = samp['yn+1'][-1]
    # samp['x'] = samp['x'][-1]
    return samp

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
        mu = mu / torch.norm(mu, dim=-1, keepdim=True)

        scale = self.layer_scale(out)
        scale = nn.functional.softplus(scale) + 0.001
        return mu, scale

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
    # print(loss)
    loss.backward()
    optimizer.step()
    return loss.item()

losses = []
distances_all = []
# for step in tqdm(range(100000)):
for i,step in tqdm(enumerate(range(100000))):
    batch = get_batch(seed=next(rng_seq))
    loss = train(model, batch)
    losses.append(loss)
    print(loss)
    if i%999==0:
        distances_all.append(eval(rotations_plane,noise_schedule,model).cpu().numpy())


save_path = Path(str(datetime.datetime.now()).replace(" ","_"))
save_path.mkdir(exist_ok=True,parents=False)

fig,ax = plt.subplots()
ax.plot(losses)
plt.savefig(save_path / "losses.png")

fig,ax = plt.subplots()
ax.plot(distances_all)
plt.savefig(save_path / "distances_all.png")




# Try to sample with the trained model
points = rotations_plane
nsamps = points.shape[0]
points = sample(rotations_plane, torch.from_numpy(np.array(noise_schedule[-1])).sqrt()*torch.ones(nsamps))

p = visualize_so3_probabilities(
    jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in points['x']]), 
    0.001)
p.savefig(save_path / 'target_dist.png')

# Samples at T=~1 # Nur noise
p = visualize_so3_probabilities(
    jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in points['yn']]),
    0.001);
p.savefig(save_path / 'noisy_dist.png')

_ ,x = sample_from_model(rotations_plane,noise_schedule,model)

p = visualize_so3_probabilities(
    jnp.array([jaxlie.SO3(y[[3,0,1,2]].cpu().numpy()).as_matrix() for y in x]),
    0.001);
p.savefig(save_path / 'denoised_dist.png')

p = visualize_so3_density(
    jnp.array([jaxlie.SO3(y[[3,0,1,2]].cpu().numpy()).as_matrix() for y in x]),
    32);

p.savefig(save_path / 'denoised_density.png')