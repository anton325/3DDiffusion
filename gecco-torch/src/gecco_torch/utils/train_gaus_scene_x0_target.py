# Main script used to train a particular model on a particular dataset.
import shutil
from absl import app
from absl import flags

import sys
sys.path.append('../')

import haiku as hk
from torch import nn, optim
from pathlib import Path
import datetime
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as onp
import jaxlie
import lietorch
from torch import nn
import math
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_gaussian_no_vmap import IsotropicGaussianSO3 as IsotropicGaussianSO3_no_vmap
from gecco_torch.utils.isotropic_plotting import visualize_so3_density, visualize_so3_probabilities
from gecco_torch.utils.sh_utils import SH2RGB, RGB2SH
import matplotlib.pyplot as plt

import torch

from gecco_torch.additional_metrics.metrics_so3 import c2st_gaussian, minimum_distance
from gecco_torch.scene.gaussian_model import GaussianModel
import json
from torchvision.utils import save_image
from gecco_torch.gaussian_renderer import render
from typing import NamedTuple

flags.DEFINE_string("dataset", "gaus_rot", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "so3models/so3ddpm/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 1024, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate for the optimizer.")
# flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 4000, "Number of samples to draw at testing time.")
# flags.DEFINE_integer("test_nsamples", 200_000, "Number of samples to draw at testing time.")
flags.DEFINE_string("input_rotation_param", "matrix", "Parameterisation of the rotation at the input of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("output_rotation_param", "matrix", "Parameterisation of the rotation at the output of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("diffusion_type", "vexp", "Variance preserving or variance exploding diffusion 'vexp' or 'vpres'") 

flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")

flags.DEFINE_string("sampler", "ddpm_sampling", "Sampler to use. Can be 'original' or 'heun'.") # heun original_sample ddpm_sampling

flags.DEFINE_integer("n_steps", 128, "Number of steps in noise schedule")

flags.DEFINE_integer("n_folds", 5, "Number of folds in c2st")
    
FLAGS = flags.FLAGS

FORCE_SMALL_SCALE = False

class LogUniformSchedule(nn.Module):
    """
    LogUniform noise schedule which seems to work better in our (GECCO) context.

    alle schedules returnen einfach nur für jedes n ein sigma, 
    sie werden gecalled mit schedule(samples) und samples hat shape (batchsize, num_points, 3)
    und dann gibt er für jedes element im batch ein sigma
    """

    def __init__(self, max: float, min: float = 0.002, low_discrepancy: bool = True):
        super().__init__()

        self.sigma_min = min
        self.sigma_max = max
        self.log_sigma_min = math.log(min)
        self.log_sigma_max = math.log(max)
        self.low_discrepancy = low_discrepancy

    # def forward(self, data):
    #     def ones(n: int):
    #         return (1,) * n
    #     u = torch.rand(data.shape[0], device=data.device) # uniform distribution between 0,1

    #     if self.low_discrepancy:
    #         div = 1 / data.shape[0]
    #         u = div * u
    #         u = u + div * torch.arange(data.shape[0], device=data.device)

    #     sigma = (
    #         u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
    #     ).exp()
    #     return sigma.reshape(-1, *ones(data.ndim - 1))
    
    def return_schedule(self,n):
        u = torch.linspace(0,1,n).cuda()
        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma
    


def original_sample(model,gt_rotations,noise_schedule):
    # Starting sampling from the trained model
    print("original sampling...")
    if FLAGS.diffusion_type == "vexp":
        X0 = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec()
    else:
        raise NotImplementedError

    def fn_sample(x, delta_mu, s):
        rotated_mu = (lietorch.SO3(delta_mu) * lietorch.SO3(x)).vec()
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(rotated_mu, s)
        samples = (lietorch.SO3(rotated_mu) * lietorch.SO3.exp(axis_angles)).vec()
        return samples

    
    x_t = X0.cuda()
    with torch.no_grad():
        for sn in torch.flip(noise_schedule,[0]):
            noise_level = sn*torch.ones([FLAGS.test_nsamples,1],device=x_t.device)
            noise_level = noise_level[:x_t.shape[0]]
            mu, s = model(x_t, noise_level)

            x_t = fn_sample(x_t, mu, s)

    # Remove nans if we accidentally sampled any
    x_t = x_t[~torch.isnan(x_t.sum(axis=-1))]
    return x_t

@torch.no_grad()
def sample_heun_so3(model,gt_rotations,noise_schedule):
    print("heun sampling ...")
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec().to(gt_rotations.device)

    def fn_sample(x, s):
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(x, s)
        samples = (lietorch.SO3(x) * lietorch.SO3.exp(axis_angles)).vec()
        return samples

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t = noisy_rotation
    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        mu, scale = model(x_t, noise_level)

        denoised_rotation = fn_sample(mu, scale)

        scaled_x_t = lietorch.SO3.exp(lietorch.SO3(x_t).log()/t_i)

        scaled_denoised_rotation = lietorch.SO3.exp(lietorch.SO3(denoised_rotation).log()/t_i).inv()

        dif_rotation = scaled_x_t * scaled_denoised_rotation

        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0
        x_next = (lietorch.SO3(x_t) * lietorch.SO3.exp((t_iminus-t_i) * dif_rotation.log())).vec()

        if t_iminus != 0:
            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            mu , scale = model(x_next, noise_level)
            denoised_rotation = fn_sample(mu, scale)


            scaled_x_next = lietorch.SO3.exp(lietorch.SO3(x_next).log()/t_iminus)

            scaled_denoised_rotation_2nd = lietorch.SO3.exp(lietorch.SO3(denoised_rotation).log()/t_iminus).inv()
            dif_rotation_2nd = scaled_x_next * scaled_denoised_rotation_2nd

            x_next = (lietorch.SO3(x_t) * lietorch.SO3.exp((t_iminus-t_i) * 
                                            (lietorch.SO3.exp(dif_rotation.log()/2) * lietorch.SO3.exp(dif_rotation_2nd.log()/2)).log()
                                            )).vec()
        x_t = x_next

    return x_t

def get_noisy_rotation(n, noise_level):
    unit_rotation = torch.zeros([n,4],device='cuda')
    unit_rotation[:,3] = 1
    unit_rotation = lietorch.SO3(unit_rotation).vec()
    noise = noise_level * torch.ones([n,1],device=unit_rotation.device)
    axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                s_single,
                                                                force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                            randomness="different")(unit_rotation, noise)
    samples = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles))
    return samples

@torch.no_grad()
def ddpm_sampling(model,gt_rotations,noise_schedule):
    print("ddpm sampling")
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_rotations.shape[0]).vec().to(gt_rotations.device)
    noisy_rest = torch.randn([noisy_rotation.shape[0],10],device=noisy_rotation.device) * noise_schedule[-1]
    
    def fn_sample(x, s):
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(x, s)
        samples = (lietorch.SO3(x) * lietorch.SO3.exp(axis_angles))
        return samples

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t_rotation = noisy_rotation
    x_t_rest = noisy_rest

    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t_rotation.shape[0],1],device=x_t_rotation.device)
        mu, scale, denoised_rest = model(x_t_rest, x_t_rotation, noise_level)

        denoised_rotation = fn_sample(mu, scale).vec()

        """
        füge bisschen noise hinzu
        """
        if i < len(noise_schedule) - 1:
            t_iminus = noise_schedule[i+1]
            noisy_rotation = get_noisy_rotation(x_t_rotation.shape[0], t_iminus)
            x_t_rotation = (lietorch.SO3(denoised_rotation) * noisy_rotation.inv()).vec()

            noisy_rest = torch.randn([x_t_rotation.shape[0],10],device=x_t_rotation.device) * t_iminus
            x_t_rest = denoised_rest + noisy_rest

        else:
            x_t_rotation = denoised_rotation
            x_t_rest = denoised_rest

    return x_t_rest, x_t_rotation


    
def eval(rng_seq, noise_schedule, model, output_dir, step, gt_rotations):
    print("eval...")
    
    if FLAGS.sampler == "heun":
        x_t_rotation = sample_heun_so3(model,gt_rotations,noise_schedule)
    elif FLAGS.sampler == "original_sample":
        x_t_rotation = original_sample(model,gt_rotations,noise_schedule)
    elif FLAGS.sampler == "ddpm_sampling":
        x_t_rest, x_t_rotation = ddpm_sampling(model,gt_rotations,noise_schedule)
    # with open(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_' + str(FLAGS.test_nsamples) + ".npy", "wb") as f:
    #     onp.save(f, x_t.detach().cpu().numpy())

    visualize_so3_density(jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in x_t_rotation]), 100);
    plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_density_{step}_' + str(FLAGS.test_nsamples) + ".png")

    visualize_so3_probabilities(jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in x_t_rotation]), 0.001);
    plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_probability_{step}_' + str(FLAGS.test_nsamples) + ".png")

    visualize_so3_density(jnp.array([jaxlie.SO3(x.cpu().numpy()).as_matrix() for x in gt_rotations]), 100);
    plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_gt_density_' + str(FLAGS.test_nsamples) + ".png")

    visualize_so3_probabilities(jnp.array([jaxlie.SO3(x.cpu().numpy()).as_matrix() for x in gt_rotations]), 0.001);
    plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_gt_probability_' + str(FLAGS.test_nsamples) + ".png")


    """
    render (obwohl es wahrscheinlich keinen Sinn macht)
    """
    with open("/home/giese/Documents/gaussian-splatting/circle_cams.json","r") as f:
        circle_cams = json.load(f)  

    class Camera(NamedTuple):
        world_view_transform: torch.Tensor
        projection_matrix: torch.Tensor
        tanfovx: float
        tanfovy: float
        imsize: int

    img_path = Path(output_dir ,'vis', str(step))
    img_path.mkdir(parents=True, exist_ok=True)
    gm = GaussianModel(3)
    gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
    gm._scaling = x_t_rest[:,6:9]
    gm._opacity = x_t_rest[:,9].unsqueeze(1)
    gm._features_dc = RGB2SH(x_t_rest[:,3:6]).unsqueeze(1)
    gm._xyz = x_t_rest[:,:3]
    gm._rotation = x_t_rotation
    for i in circle_cams.keys():
        cam = circle_cams[i]
        camera = Camera(
            world_view_transform = torch.tensor(cam["world_view_transform"]).cuda(),#.unsqueeze(0),
            projection_matrix = torch.tensor(cam['projection_matrix']).cuda(),#.unsqueeze(0),
            tanfovy = 0.45714,
            tanfovx = 0.45714,
            imsize=400,
        )
        bg = torch.tensor([1.0,1.0,1.0], device=torch.device('cuda')) # weiß, weil das originale referenz bild auch schwarz ist, aber wirs auf weiß ändern
        with torch.no_grad():
            render_dict = render(gm,bg,camera = camera)
            img = render_dict['render']
            save_image(img, img_path / f'_render_{i}.png')

    """
    compute c2st in wxyz format 
    Wir versuchen einen Classifier zu trainieren, der zwischen gt rotations und generierten rotations unterscheidet. Wenn als 
    score 0.5 rauskommt, heißt es, dass das Trainieren fehlschlägt, was darauf hindeutet, dass die generierten und gt rotations
    nicht zu unterscheiden sind
    """

    print("Calculating c2st ... ")

    true_samp = gt_rotations # true samp ist in wxyz

    seed = 1

    x_t_wxyz = x_t_rotation[:,[3,0,1,2]]
    # true_samp_wxyz = true_samp[:,[3,0,1,2]]
    c2_score = c2st_gaussian(true_samp, x_t_wxyz, seed, FLAGS.n_folds)

    with open(output_dir+"output_"+ FLAGS.diffusion_type + ".txt", "a") as f:
        print( f"step {step} C2ST score: "+ str(c2_score), file=f)

    print("C2ST score: "+ str(c2_score))

    print("compute rotational distance")

    sum_distance = minimum_distance(x_t_wxyz, gt_rotations)

    return sum_distance

# def lr_schedule(step):
#     """Step learning rate schedule rule."""
#     lr = (1.0 * FLAGS.batch_size) / 1024
#     boundaries = torch.tensor([0.2, 0.7]) * FLAGS.training_steps
#     values = torch.tensor([1.0, 0.1, 0.01]) * lr
#     index = torch.sum(boundaries < step).item()
#     return values[index]

def quat_power(quat, a):
    quat = lietorch.SO3(quat)
    return lietorch.SO3.exp(quat.log()*a)

def get_batch(batch, key, noise_schedule):

    def sample_vexp(q, rest, temperature):
        """
        bei checkerboard dataset kommt hier das quaternion auch rein (keine matrix)
        """
        scale = noise_schedule[temperature]
        # scale = scale[0].repeat(scale.shape[0])

        # von wxyz zu xyzw # -> das ist das format von lietorch und scipy, wxyz ist das format von gaussian splatting und jaxlie
        q = q[:,[1,2,3,0]]

        q = lietorch.SO3(q).vec()
        # Sampling from current temperature
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(q, scale)
        qn = (lietorch.SO3(q) * lietorch.SO3.exp(axis_angles)).vec()

        noise = torch.randn_like(rest) * scale[:,None]
        noisy_rest = rest + noise

        return {'x': q, 'yn': qn, 'sn':scale, 'rest': rest, 'noisy_rest': noisy_rest}

    # Sample random noise levels from the schedule
    temp_list = torch.arange(len(noise_schedule)-1, dtype=torch.float32,device='cuda')
    probs = torch.ones_like(temp_list)  # Example probabilities for each item -> wenn jedes item 1 hat, wird es gleich häufig gesampled
    num_samples = FLAGS.batch_size # Number of samples

    # Sample with replacement
    temperature = torch.multinomial(probs, num_samples, replacement=True)

    if FLAGS.diffusion_type == "vexp":
        sample = sample_vexp
    else:
        raise NotImplementedError

    # Sample random rotations
    pos_quat = batch['pos_quat'].to(temperature.device)
    rest = batch['rest']
    sampled = sample(pos_quat, rest, temperature)
    return sampled

 
class SimpleMLP(nn.Module):
    def __init__(self,input_size=20): # input size: 6 von der rotationsmatrix, 1 von variance
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.layer_mu = nn.Linear(256, 6)
        self.layer_scale = nn.Linear(6, 1)
        self.layer_rest = nn.Linear(256, 10)

    def forward(self, rest, rotation, s):
        # konvertiere quaternion zu rotation matrix
        rot_mat = lietorch.SO3(rotation).matrix()[:,:3,:3]
        rot_mat = rot_mat.reshape(-1,9)

        concat = torch.cat([rest, rot_mat, s], dim=-1)
        out = self.mlp(concat)

        mu = self.layer_mu(out)
        scale = self.layer_scale(mu)
        scale = nn.functional.softplus(scale) + 0.0001
        
        R1 = mu[:,0:3] / torch.norm(mu[:,0:3], dim=-1, keepdim=True)
        R3 = torch.cross(R1, mu[:, 3:], dim=-1)
        R3 = R3 / torch.norm(R3, dim=-1, keepdim=True)
        R2 = torch.cross(R3, R1, dim = -1)

        rotation_matrix = torch.stack([R1,R2,R3],dim=-1)

        quat = lietorch.SO3(rotation_matrix,from_rotation_matrix=True).vec()

        rest_out = self.layer_rest(out)

        # scalings in range -10/10 so use sigmoid mapping for that
        rest_out[:,6:9] = torch.sigmoid(rest_out[:,6:9]) * 25 - 13

        return quat, scale, rest_out

def train(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    # yn1 = torch.from_numpy(np.array(batch['yn+1']))
    # sn1 = torch.from_numpy(np.array(batch['sn+1']))
    yn = batch['yn'].cuda()
    sn = batch['sn'].cuda()
    sn = sn[:yn.shape[0]]

    rest_x0 = batch['rest'].cuda()
    noisy_rest = batch['noisy_rest'].cuda()

    mu, scale, denoised_rest = model(noisy_rest, yn, sn.reshape(-1,1))

    # visualize_so3_probabilities(jnp.array([jaxlie.SO3(x[[3,0,1,2]].detach().cpu().numpy()).as_matrix() for x in mu]), 0.001);
    # plt.savefig("/home/giese/Documents/gecco/temp.png")

    def fn(x, mu, scale):
        # mu = lietorch.SO3(mu) * lietorch.SO3(y) # apply residual rotation
        # dist = IsotropicGaussianSO3(mu, scale, 
        #                             force_small_scale=FORCE_SMALL_SCALE)

        # prob_dist = dist.log_prob(x)
        # scale = torch.ones((x.shape[0],1), device=x.device) * 0.1
        dist = IsotropicGaussianSO3_no_vmap(mu, scale,
                                    force_small_scale=FORCE_SMALL_SCALE)

        prob_dist = dist.log_prob(x)
        return prob_dist # shape 512
    
    q = batch['x']
    loss_rot = (-fn(q, mu, scale)).mean() + (scale.mean()/2)
    loss_rest = (denoised_rest - rest_x0).pow(2).mean() * 100
    loss_scale = (denoised_rest[:,6:9] - rest_x0[:,6:9]).pow(2).mean() * 100
    print(loss_rest)
    loss = loss_rot + loss_rest + loss_scale
    loss.backward()
    optimizer.step()
    return loss_rest.item(), loss_rot.item(), loss.item()

def main(_):
    run = datetime.datetime.now()
    run = str(run).replace(" ","_")
    run = run.replace(run.split(".")[-1],"")
    run = FLAGS.dataset + "_" + FLAGS.sampler + "_" + run + "_" + FLAGS.diffusion_type
    output_dir = FLAGS.output_dir  + run + "/"

    Path(output_dir +'/model').mkdir(parents=True, exist_ok=True)
    Path(output_dir +'/vis').mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, output_dir)
    # Instantiate the network
    model = SimpleMLP().cuda()
    
    rng_seq = hk.PRNGSequence(42)
    
    if FLAGS.diffusion_type == "vexp":
        noise_schedule = torch.linspace(0.05, 1.25, FLAGS.n_steps).cuda()
        noise_schedule = noise_schedule**2 + 0.0001

        schedule = LogUniformSchedule(165)
        noise_schedule = schedule.return_schedule(FLAGS.n_steps)

        # schedule = torch.linspace(0.0)

    else:
        raise NotImplementedError     
    
    fig,ax = plt.subplots()
    x = onp.arange(0,len(noise_schedule))
    ax.plot(x,noise_schedule.cpu().numpy())
    plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_noise_schedule.png')
 
    if FLAGS.train:
        distances_all = []
        steps = []
        losses = []
        losses_rot = []
        losses_rest = []
        # Open the dataset
        gm = GaussianModel(3)
        gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
        rotations_plane = gm.get_rotation.detach() # quats in wxyz -> das ist ok, wir rechnen das in get_batch um auf xyzw
        sh = gm.get_features.detach()
        rgb = SH2RGB(sh)[:,0]
        scaling = gm._scaling.detach()
        xyz = gm._xyz.detach()
        opactiy = gm._opacity.detach()
        rest = torch.cat([xyz, rgb, scaling, opactiy], dim=-1)

        # Initialize weights
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)


        print('training begins')
        for step in tqdm(range(FLAGS.training_steps)):
            rand_perm_i = torch.randperm(rotations_plane.shape[0])
            rand_perm = rotations_plane[rand_perm_i]
            rand_perm_rest = rest[rand_perm_i]
            dset = {
                'pos_quat':rand_perm[:FLAGS.batch_size],
                'rest' : rand_perm_rest[:FLAGS.batch_size],
                }
            batch = get_batch(dset, next(rng_seq), noise_schedule)
            # Sampling another batch if the current one had a NaN
            while torch.isnan(batch['yn']).sum()>0:
                print("WARNING: Skipped a batch because it had a NaN")
                rand_perm_i = torch.randperm(rotations_plane.shape[0])
                rand_perm = rotations_plane[rand_perm_i]
                rand_perm_rest = rest[rand_perm_i]
                dset = {
                    'pos_quat':rand_perm[:FLAGS.batch_size],
                    'rest' : rand_perm_rest[:FLAGS.batch_size],
                    }
                batch = get_batch(dset, next(rng_seq), noise_schedule)
            
            # adjust learning rate
            if step == 80_000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FLAGS.learning_rate * 0.1
                    print(f"set to {FLAGS.learning_rate * 0.1}")
                    
            elif step == 280_000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FLAGS.learning_rate * 0.01

            loss_rest, loss_rot, loss = train(model,batch,optimizer)
            losses.append(loss)#.detach().cpu().numpy())
            losses_rot.append(loss_rot)#.detach().cpu().numpy())
            losses_rest.append(loss_rest)#.detach().cpu().numpy())
 
            if step%50==0:
                print(loss)

            if step in [1000] or (step>1000 and step % 20000 == 0):
                distances = eval(rng_seq, noise_schedule, model, output_dir, step, rotations_plane)
                distances_all.append(distances.cpu().numpy())
                steps.append(step)
                fig,ax = plt.subplots()
                ax.plot(steps,distances_all)
                ax.set_title(f'Rotational distances ({distances.cpu().numpy()})')  # Add this line to set the title
                plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_distances.png')

            if step %500 == 0:
                fig,ax = plt.subplots()
                ax.plot(losses)
                ax.set_title(f'Loss: ({losses[-1]})')  # Add this line to set the title
                plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_losses.png')

                fig,ax = plt.subplots()
                ax.plot(losses_rest)
                ax.set_title(f'Loss: ({losses_rest[-1]})')  # Add this line to set the title
                plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_losses_rest.png')

                fig,ax = plt.subplots()
                ax.plot(losses_rot)
                ax.set_title(f'Loss: ({losses_rot[-1]})')  # Add this line to set the title
                plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_losses_rot.png')


            if step%10000 ==0:
                state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_model-{step}.pckl'
                torch.save(model.state_dict(),state_dict_path)

        state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_model-final.pckl'
        torch.save(model.state_dict(),state_dict_path)

    state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_model-final.pckl'
    torch.save(model.state_dict(),state_dict_path)


if __name__ == "__main__":
    app.run(main)
