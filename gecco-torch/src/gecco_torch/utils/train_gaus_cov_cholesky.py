# Main script used to train a particular model on a particular dataset.
from absl import app
from absl import flags

import sys
sys.path.append('../')

from torchvision.utils import save_image

import shutil
import haiku as hk
from typing import NamedTuple
from gecco_torch.gaussian_renderer import render
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
from gecco_torch.utils.build_cov_matrix_torch import build_covariance_from_activated_scaling_rotation, build_covariance_from_scaling_rotation_xyzw, strip_lowerdiag
import matplotlib.pyplot as plt
import json

import torch
from gecco_torch.utils.riemannian_helper_functions import lower_triangle_to_3x3, upper_triangle_to_cov_3x3, find_cholesky_L

from gecco_torch.additional_metrics.metrics_so3 import c2st_gaussian, best_fit_geodesic_distance, geodesic_distance, geodesic_distance_log_L
from gecco_torch.scene.gaussian_model import GaussianModel

flags.DEFINE_string("dataset", "gaus_rot_cholesky_small", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "so3models/so3ddpm/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate for the optimizer.")
# flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 500, "Number of samples to draw at testing time.")
# flags.DEFINE_integer("test_nsamples", 200_000, "Number of samples to draw at testing time.")
flags.DEFINE_string("input_rotation_param", "matrix", "Parameterisation of the rotation at the input of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("output_rotation_param", "matrix", "Parameterisation of the rotation at the output of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("diffusion_type", "vexp", "Variance preserving or variance exploding diffusion 'vexp' or 'vpres'") 

flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")

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

    def return_schedule(self,n):
        u = torch.linspace(0,1,n).cuda()
        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma
    

@torch.no_grad()
def L_sample_heun_so3(model,gt_log_L,noise_schedule):
    print("heun sampling ...")
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_log_L.shape[0]).vec().to(gt_log_L.device)
    noisy_scale = torch.randn(gt_log_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)
    

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t = noisy_start_L.cuda()
    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        denoised_log_L = model(x_t, noise_level)
        denoised_cov = log_L_to_3x3cov(denoised_log_L)
        x_t_cov = log_L_to_3x3cov(x_t)
        scaled_log_L = lietorch.SO3.exp(lietorch.SO3(x_t_cov,from_rotation_matrix=True).log()/t_i)

        scaled_denoised_log_L = lietorch.SO3.exp(lietorch.SO3(denoised_cov,from_rotation_matrix=True).log()/t_i).inv()

        dif_rotation = scaled_log_L * scaled_denoised_log_L

        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0
        x_next = (lietorch.SO3(x_t_cov,from_rotation_matrix=True) * lietorch.SO3.exp((t_iminus-t_i) * dif_rotation.log())).matrix()[:,:3,:3]
        x_next = find_cholesky_L(strip_lowerdiag(x_next))

        if t_iminus != 0:
            x_next_cov = log_L_to_3x3cov(x_next)
            scaled_log_L_second = lietorch.SO3.exp(lietorch.SO3(x_next_cov,from_rotation_matrix=True).log()/t_iminus)
            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            denoised_log_L = model(x_next, noise_level)
            denoised_log_L_cov = log_L_to_3x3cov(denoised_log_L)
            scaled_denoised_log_L_2nd = lietorch.SO3.exp(lietorch.SO3(denoised_log_L_cov,from_rotation_matrix=True).log()/t_iminus).inv()
            dif_rotation_2nd = scaled_log_L_second * scaled_denoised_log_L_2nd
            x_next = (lietorch.SO3(x_t_cov,from_rotation_matrix=True) * lietorch.SO3.exp((t_iminus-t_i) * 
                                            (lietorch.SO3.exp(dif_rotation_2nd.log()/2) * lietorch.SO3.exp(dif_rotation.log()/2)).log()
                                            )).matrix()[:,:3,:3]
            x_t = find_cholesky_L(strip_lowerdiag(x_next))
        else:
            x_t = x_next

    return x_t


@torch.no_grad()
def L_sample_heun(model,gt_log_L,noise_schedule):
    print("heun sampling ...")
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_log_L.shape[0]).vec().to(gt_log_L.device)
    noisy_scale = torch.randn(gt_log_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t = noisy_start_L.cuda()
    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        denoised_log_L = model(x_t, noise_level)
        denoised_L = unlog_L(denoised_log_L)

        scaled_L = denoised_L/t_i

        scaled_x_t = unlog_L(x_t)/t_i

        dif = scaled_x_t - scaled_L

        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0

        x_next = x_t + (t_iminus-t_i) * dif
        x_next = log_L(x_next)

        if t_iminus != 0:

            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            denoised_log_L = model(x_next, noise_level)
            denoised_L = unlog_L(denoised_log_L)

            scaled_x_next = unlog_L(x_next) / t_iminus
            scaled_denoised_L = denoised_L / t_iminus

            dif_2nd = scaled_x_next - scaled_denoised_L
            x_next = unlog_L(x_t) + (t_iminus-t_i) * (dif_2nd/2 + dif/2)
            x_next = log_L(x_next)

    return x_t

@torch.no_grad()
def L_sample_heun_rand_start(model,gt_log_L,noise_schedule):
    print("heun sampling ...")
    noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_log_L.shape[0]).vec().to(gt_log_L.device)
    noisy_scale = torch.randn(gt_log_L.shape[0],3).cuda() * noise_schedule[-1]
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)
    noisy_start_L = torch.randn_like(noisy_start_L) * noise_schedule[-1]

    noise_schedule = torch.flip(noise_schedule,[0])
    x_t = noisy_start_L.cuda()
    for i in range(len(noise_schedule)):
        t_i = noise_schedule[i]

        noise_level = t_i * torch.ones([x_t.shape[0],1],device=x_t.device)
        denoised_log_L = model(x_t, noise_level)

        scaled_L = denoised_log_L/t_i

        scaled_x_t = x_t/t_i

        dif = scaled_x_t - scaled_L

        t_iminus = noise_schedule[i+1] if i+1 < len(noise_schedule) else 0

        x_next = x_t + (t_iminus-t_i) * dif
        x_next = x_next

        if t_iminus != 0:

            noise_level = t_iminus * torch.ones([x_t.shape[0],1],device=x_t.device)
            denoised_log_L = model(x_next, noise_level)
            denoised_L = denoised_log_L

            scaled_x_next = x_next / t_iminus
            scaled_denoised_L = denoised_L / t_iminus

            dif_2nd = scaled_x_next - scaled_denoised_L
            x_next = x_t + (t_iminus-t_i) * (dif_2nd/2 + dif/2)
            x_next = x_next

    return x_t

def L_sampling(model,gt_log_L,noise_schedule):
    # Starting sampling from the trained model
    print("original sampling...")
    if FLAGS.diffusion_type == "vexp":
        noisy_rotation = lietorch.SO3([],from_uniform_sampled=gt_log_L.shape[0]).vec().to(gt_log_L.device)
        noisy_scale = torch.randn(gt_log_L.shape[0],3).cuda() * noise_schedule[-1]
        noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
        noisy_start_L = find_cholesky_L(noisy_cov)

    else:
        raise NotImplementedError

    
    x_t = noisy_start_L.cuda()

    def get_noisy_log_L(noise_level):
        unit_rotation = torch.tensor([0.,0.,0.,1.]).reshape(1,4).cuda().repeat(noise_level.shape[0],1) # xyzw
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,s_single).sample_one_vmap(),randomness="different")(unit_rotation, noise_level)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(noise_level.shape[0],3).cuda() * noise_level
        noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
        noisy_L = find_cholesky_L(noisy_cov)
        return noisy_L
    
    def get_noisy_rotation(noise_level):
        unit_rotation = torch.tensor([0.,0.,0.,1.]).reshape(1,4).cuda().repeat(noise_level.shape[0],1) # xyzw
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation, noise_level)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles))
        return noisy_rotation
    
    def add_noisy_log_L(log_L_1,log_L_2):
        summation = torch.zeros_like(log_L_1).to(log_L_1.device)
        summation[:,0:3] = torch.log(torch.exp(log_L_1[:, 0:3]) + torch.exp(log_L_2[:, 0:3]))
        summation[:,3:6] = log_L_1[:, 3:6] + log_L_2[:, 3:6]
        return summation

    with torch.no_grad():
        for sn in torch.flip(noise_schedule,[0]):
            noise_level = sn * torch.ones([FLAGS.test_nsamples,1],device=x_t.device)
            noise_level = noise_level[:x_t.shape[0]]
            denoised_L = model(x_t, noise_level)
            if sn != torch.flip(noise_schedule,[0])[-1]:
                # x_t = add_noisy_log_L(denoised_L, get_noisy_log_L(noise_level))
                x_t = denoised_L + torch.randn_like(x_t) * sn
            else:
                x_t = denoised_L

            # noisy_rotation = get_noisy_rotation(noise_level)
            # x_t = (lietorch.SO3(lower_triangle_to_3x3(denoised_L),from_rotation_matrix=True) * noisy_rotation).matrix()
            # x_t = strip_lowerdiag(x_t)


    # Remove nans if we accidentally sampled any
    x_t = x_t[~torch.isnan(x_t.sum(axis=-1))]
    return x_t

    
def eval(rng_seq, noise_schedule, model, output_dir, step, gt_L):
    print("eval...")
    
    sampled_log_L = L_sample_heun_rand_start(model,gt_L, noise_schedule)
    # sampled_log_L = L_sampling(model,gt_log_L, noise_schedule)
    # with open(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_' + str(FLAGS.test_nsamples) + ".npy", "wb") as f:
    #     onp.save(f, x_t.detach().cpu().numpy())
    print("compute rotational distance")

    sum_distance, row_ind, col_ind = best_fit_geodesic_distance(gt_L, sampled_log_L, return_indices=True)

    closest_sampled_log_L = sampled_log_L[col_ind]
    closest_sampled_covs_3x3 = L_to_cov(closest_sampled_log_L)
    closest_sampled_covs = strip_lowerdiag(closest_sampled_covs_3x3)

    with open("/home/giese/Documents/gaussian-splatting/circle_cams.json","r") as f:
        circle_cams = json.load(f)  

    class Camera(NamedTuple):
        world_view_transform: torch.Tensor
        projection_matrix: torch.Tensor
        tanfovx: float
        tanfovy: float
        imsize: int

    img_path = Path(output_dir ,'vis' ,'imgs',f'imgs_{step}')
    img_path.mkdir(parents=True, exist_ok=True)
    gm = GaussianModel(3)
    gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
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
        # print(f"shape {context.camera.world_view_transform.shape}") # (1,4,4)
        kwargs = {
            'use_cov3D' : True,
            'covs' : closest_sampled_covs #closest_sampled_covs
        }
        with torch.no_grad():
            render_dict = render(gm,bg,camera = camera, **kwargs)
            img = render_dict['render']
            save_image(img, img_path / f'_render_{i}.png')

    """
    compute c2st in wxyz format 
    Wir versuchen einen Classifier zu trainieren, der zwischen gt rotations und generierten rotations unterscheidet. Wenn als 
    score 0.5 rauskommt, heißt es, dass das Trainieren fehlschlägt, was darauf hindeutet, dass die generierten und gt rotations
    nicht zu unterscheiden sind
    """

    print("Calculating c2st ... ")

    seed = 1

    # true_samp_wxyz = true_samp[:,[3,0,1,2]]
    c2_score = c2st_gaussian(gt_L, sampled_log_L, seed, FLAGS.n_folds)

    with open(output_dir+"output_"+ FLAGS.diffusion_type + ".txt", "a") as f:
        print( f"step {step} C2ST score: "+ str(c2_score), file=f)

    print("C2ST score: "+ str(c2_score))



    return sum_distance


def get_batch(batch, key, noise_schedule):

    def sample_vexp(rotation, scaling, temperature):
        scale = noise_schedule[temperature]
        # scale = torch.tensor([0.001]).repeat(scale.shape[0]).cuda()

        gt_cov = build_covariance_from_activated_scaling_rotation(scaling, rotation)
        gt_L = find_cholesky_L(gt_cov)

        """
        find noisy L
        """
        rotation_xyzw = rotation[:,[1,2,3,0]] # von wxyz zu xyzw

        # Sampling from current temperature
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                                    s_single,
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(rotation_xyzw, scale)
        noisy_rotation = (lietorch.SO3(rotation_xyzw) * lietorch.SO3.exp(axis_angles)).vec()

        noisy_rotation = noisy_rotation[:,[3,0,1,2]] # von xyzw zu wxyz

        noise = torch.randn_like(scaling) * scale[:, None]

        noisy_scaling = scaling + noise

        noisy_cov = build_covariance_from_activated_scaling_rotation(noisy_scaling, noisy_rotation)
        noisy_L = find_cholesky_L(noisy_cov)

        return {'rotation': rotation, 'scaling': scaling, 'noisy_scaling': noisy_scaling, 'noisy_rotation': noisy_rotation,
                'noisy_L': noisy_L, 'gt_L': gt_L, 'sn' : scale}

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
    rotation = batch['rotation'].to(temperature.device)
    scale = batch['scale'].to(temperature.device)
    sampled = sample(rotation,scale, temperature)
    return sampled

 
class SimpleMLP(nn.Module):
    def __init__(self,input_size=7): # input size: 6 von der rotationsmatrix, 1 von variance
        super(SimpleMLP, self).__init__()
        neurons = 512
        self.mlp = nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
        )
        self.layer_L = nn.Linear(neurons, 6)

    def forward(self, x, s):
        concat = torch.cat([x, s], dim=-1)
        out = self.mlp(concat)

        L = self.layer_L(out)

        # exponentieren, damit die diagonale safe positiv ist
        L[:,:3] = torch.exp(L[:,:3])
        
        return L

def train(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()

    noisy_L = batch['noisy_L'].cuda()
    scale = batch['sn'].cuda()

    denoised_L = model(noisy_L, scale.reshape(-1,1))

    gt_L = batch['gt_L'].cuda()
    if not torch.isinf(denoised_L).any():
        loss = geodesic_distance(denoised_L, gt_L).mean()
        print(loss)
        loss.backward()
        optimizer.step()
        return loss.item()
    else:
        return 10
    # if not torch.isinf(geodesic_distance(denoised_L, gt_L).mean()):
    #     loss = ((gt_L - denoised_L)**2).mean()*100
    # else:
    #     loss = ((gt_L - denoised_L)**2).mean()*100 #+ geodesic_distance(denoised_L, gt_L).mean()

def main(_):
    run = datetime.datetime.now()
    run = str(run).replace(" ","_")
    run = run.replace(run.split(".")[-1],"")
    run = FLAGS.dataset + "_" + run + "_" + FLAGS.diffusion_type
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

    else:
        raise NotImplementedError     
    
    fig,ax = plt.subplots()
    x = onp.arange(0,len(noise_schedule))
    ax.plot(x,noise_schedule.cpu().numpy())
    plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_noise_schedule.png')
 
    if FLAGS.train:
        distances_all = []
        distance_losses = []
        # Open the dataset
        gm = GaussianModel(3)
        gm.load_ply("/globalwork/giese/gaussians/02691156/1a04e3eab45ca15dd86060f189eb133/point_cloud/iteration_10000/point_cloud.ply")
        rotations_plane = gm.get_rotation.detach()[:FLAGS.test_nsamples] # quats in wxyz -> das ist ok, wir rechnen das in get_batch um auf xyzw
        scale_plane = gm.get_scaling.detach()[:FLAGS.test_nsamples]

        # Initialize weights
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=0.01)

        print('training begins')
        for step in tqdm(range(FLAGS.training_steps)):
            rand_perm_i = torch.randperm(rotations_plane.shape[0])
            rand_perm_rotations = rotations_plane[rand_perm_i]
            rand_perm_scale = scale_plane[rand_perm_i]
            dset = {'rotation':rand_perm_rotations[:FLAGS.batch_size],'scale':rand_perm_scale[:FLAGS.batch_size]}
            batch = get_batch(dset, next(rng_seq), noise_schedule)
            # Sampling another batch if the current one had a NaN
            while torch.isnan(batch['noisy_L']).sum()>0:
                print("WARNING: Skipped a batch because it had a NaN")
                rand_perm_i = torch.randperm(rotations_plane.shape[0])
                rand_perm_rotations = rotations_plane[rand_perm_i]
                rand_perm_scale = scale_plane[rand_perm_i]
                dset = {'rotation':rand_perm_rotations[:FLAGS.batch_size],'scale':rand_perm_scale[:FLAGS.batch_size]}
                batch = get_batch(dset, next(rng_seq), noise_schedule)
            
            # adjust learning rate
            if step == 80_000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FLAGS.learning_rate * 0.1
                    print(f"set to {FLAGS.learning_rate * 0.1}")
                    
            elif step == 280_000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FLAGS.learning_rate * 0.01

            loss = train(model,batch,optimizer)
            distance_losses.append(loss)#.detach().cpu().numpy())
 
            if step%50==0:
                print(loss)

            if step in [1000,10000] or (step>1000 and step % 20000 == 0):
                pass
                # # distances = eval(rng_seq, noise_schedule, model, output_dir, step, gt_log_L)
                # # distances_all.append(distances.cpu().numpy())
                # # fig,ax = plt.subplots()
                # # ax.plot(distances_all)
                # ax.set_title(f'cholesky distances ({distances.cpu().numpy()})')  # Add this line to set the title
                # plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_choleksy_distance_eval.png')

            if step % 1000 == 0:    
                fig,ax = plt.subplots()
                ax.plot(distance_losses)
                ax.set_title(f'mean cholesky distance loss: ({distance_losses[-1]})')  # Add this line to set the title
                plt.savefig(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_cholesky_distance_loss.png')

            if step%10000 ==0:
                state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_model-{step}.pckl'
                torch.save(model.state_dict(),state_dict_path)

        state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_model-final.pckl'
        torch.save(model.state_dict(),state_dict_path)

    state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_model-final.pckl'
    torch.save(model.state_dict(),state_dict_path)


if __name__ == "__main__":
    app.run(main)
