# Main script used to train a particular model on a particular dataset.
import datetime
from absl import app
from absl import flags

import sys
sys.path.append('../')

import tensorflow as tf
import tensorflow_datasets as tfds
import haiku as hk
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
import numpy as onp
import jax
import jax.numpy as jnp
import jaxlie
import lietorch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_gaussian_no_vmap import IsotropicGaussianSO3 as IsotropicGaussianSO3_no_vmap
from gecco_torch.utils.isotropic_plotting import visualize_so3_density, visualize_so3_probabilities
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from gecco_torch.additional_metrics.metrics_so3 import c2st, minimum_distance

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "so3models/so3ddpm/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 1024, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate for the optimizer.")
# flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
# flags.DEFINE_integer("test_nsamples", 100_000, "Number of samples to draw at testing time.")
flags.DEFINE_integer("test_nsamples", 50_000, "Number of samples to draw at testing time.")
flags.DEFINE_string("input_rotation_param", "matrix", "Parameterisation of the rotation at the input of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("output_rotation_param", "matrix", "Parameterisation of the rotation at the output of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("diffusion_type", "vexp", "Variance preserving or variance exploding diffusion 'vexp' or 'vpres'") 

flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")

flags.DEFINE_integer("n_steps", 256, "Number of steps in noise schedule")

flags.DEFINE_integer("n_folds", 5, "Number of folds in c2st")
    
FLAGS = flags.FLAGS

def eval(rng_seq,noise_schedule, model, output_dir,step):
    print("eval...")
    # Starting sampling from the trained model
    if FLAGS.diffusion_type == "vexp":
        X0 = lietorch.SO3([],from_uniform_sampled=FLAGS.test_nsamples).vec()

    elif FLAGS.diffusion_type == "vpres":
        locs = torch.zeros((FLAGS.test_nsamples, 4), device='cuda')
        locs[:,3] = 1
        scales = torch.ones(FLAGS.test_nsamples, device='cuda')
        axis_angles = torch.vmap(lambda l,s: IsotropicGaussianSO3(l,s,force_small_scale=True).sample_one_vmap(),
                                randomness="different")(locs, scales)
        X0 = (lietorch.SO3(locs) * lietorch.SO3.exp(axis_angles)).vec()
        # X0 = IsotropicGaussianSO3(locs, scales).sample()
    else:
        raise NotImplementedError

    def fn_sample(x, delta_mu, s):
        rotated_mu = (delta_mu * lietorch.SO3(x)).vec()
        axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                    s_single,
                                                                    force_small_scale=True).sample_one_vmap(),
                                randomness="different")(rotated_mu, s)
        samples = (lietorch.SO3(rotated_mu) * lietorch.SO3.exp(axis_angles)).vec()
        return samples

    x_t = X0.cuda()

    print("sampling...")
    # dir = Path(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_sampling_{step}')
    # print(dir)
    # dir.mkdir(exist_ok=True)
    with torch.no_grad():
        every = 16
        for j,sn in enumerate(torch.flip(noise_schedule,[0])):
            mu, s = model(x_t, sn*torch.ones([FLAGS.test_nsamples,1]).cuda())
            # x_t_new = torch.zeros_like(x_t)
            # batch_size = 10000
            # for i in range(0,mu.shape[0],batch_size):
            #     x_t_new[i:i+batch_size] = fn_sample(x_t[i:i+batch_size], mu[i:i+batch_size], s[i:i+batch_size])
            x_t = fn_sample(x_t, mu, s)
            # if j % every == 0:
            #     visualize_so3_density(jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in x_t[~torch.isnan(x_t.sum(axis=-1))]]), 100);
            #     plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_sampling_{step}/' + f"density_{j}.png")

            #     visualize_so3_probabilities(jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in x_t[~torch.isnan(x_t.sum(axis=-1))]]), 0.001 );
            #     plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_sampling_{step}/' + f"prob_{j}.png")

    # Remove nans if we accidentally sampled any
    x_t = x_t[~torch.isnan(x_t.sum(axis=-1))] # wenn ein quaternion ein nan hat, entferne das ganze

    with open(output_dir + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_' + str(FLAGS.test_nsamples) + ".npy", "wb") as f:
        onp.save(f, x_t.detach().cpu().numpy())

    visualize_so3_density(jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in x_t]), 100);
    plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_density_{step}_' + str(FLAGS.test_nsamples) + ".png")

    visualize_so3_probabilities(jnp.array([jaxlie.SO3(x[[3,0,1,2]].cpu().numpy()).as_matrix() for x in x_t]), 0.001 );
    plt.savefig(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_probability_{step}_' + str(FLAGS.test_nsamples) + ".png")


    """
    compute c2st in wxyz format
    """
    if FLAGS.compute_c2st:    
        true_samp_loc = '/home/giese/Documents/SO3DiffusionModels/scripts/reference_distribution/' + FLAGS.dataset + '_true_200_000.npy'


        with open(true_samp_loc , 'rb') as file:
            true_samp = onp.load(file)

        seed = 1
        if true_samp.shape[1] == 3:
            true_samp = jax.vmap(lambda m: jaxlie.SO3.from_matrix(m).wxyz )(true_samp) # print(X.shape)

        print("Calculating c2st ... ")
        x_t_wxyz = x_t[:,[3,0,1,2]].detach().cpu().numpy()
        c2_score = c2st(true_samp, x_t_wxyz, seed, FLAGS.n_folds)

        # with open(output_dir+"output_"+ FLAGS.diffusion_type + f"{step}.txt", "a") as f:
        #   print( "C2ST score: "+ str(c2_score), file=f)

        print(true_samp.shape[1])
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")

        print("C2ST score: "+ str(c2_score))
        
        # limit number of rotations
        gt_rots = torch.from_numpy(np.array(true_samp))[:10000]
        sampled = x_t[:10000,[3,0,1,2]].cpu()
        sum_distance = minimum_distance(sampled, gt_rots)
        
    return sum_distance, c2_score

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

    def sample_vexp(q, temperature):
        """
        bei checkerboard dataset kommt hier das quaternion auch rein (keine matrix)
        """
        scale = noise_schedule[temperature]
        scalenplus1 = noise_schedule[temperature+1]
        delta = scalenplus1**2 - scale**2

        # von wxyz zu xyzw # -> das ist das format von lietorch und scipy
        q = q[:,[1,2,3,0]]

        q = lietorch.SO3(q).vec()
        # Sampling from current temperature
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                                    s_single,
                                                                    force_small_scale=True).sample_one_vmap(),
                                randomness="different")(q, scale)
        qn = (lietorch.SO3(q) * lietorch.SO3.exp(axis_angles)).vec()
        # dist = IsotropicGaussianSO3(q, scale)
        # qn = dist.sample()

        # Sampling from next temperature step 
        scale_torch = torch.sqrt(delta * torch.ones(q.shape[0],device=qn.device))
        # dist2 = IsotropicGaussianSO3(qn, scale_torch)
        # qnplus1 = dist2.sample()
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                                    s_single,
                                                                    force_small_scale=True).sample_one_vmap(),
                                randomness="different")(qn, scale_torch)
        qnplus1 = (lietorch.SO3(qn) * lietorch.SO3.exp(axis_angles)).vec()

        return {'x': q, 'yn': qn, 'yn+1': qnplus1, 
                'sn':scale, 'sn+1':scalenplus1}  

    
    def sample_vpres(quaternion, temperature):
        alpha = noise_schedule[temperature]
        beta  = 1 - (noise_schedule[temperature+1] / alpha)
        quaternion = quaternion[:,[1,2,3,0]]
        
        # Sampling from the current temperature
        scale_torch = torch.sqrt((1-alpha) * torch.ones(quaternion.shape[0],device=quaternion.device))

        q = quat_power(quaternion, torch.sqrt(alpha).reshape(-1,1)).vec()
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                            s_single,
                                                            force_small_scale=True).sample_one_vmap(),
                        randomness="different")(q, scale_torch)
        qn = (lietorch.SO3(q) * lietorch.SO3.exp(axis_angles)).vec()

        # Sampling from the next temperature step
        scale_torch = torch.sqrt(beta * torch.ones(quaternion.shape[0],device=quaternion.device))
        qn2 = quat_power(qn, torch.sqrt(1-beta).reshape(-1,1)).vec()
        axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                    s_single,
                                                    force_small_scale=True).sample_one_vmap(),
                                                    randomness="different")(qn2, scale_torch)
        qnplus1 = (lietorch.SO3(qn2) * lietorch.SO3.exp(axis_angles)).vec()
    
        return {'x': quaternion, 'yn': qn, 'yn+1': qnplus1, 
                'sn':alpha, 'sn+1':noise_schedule[temperature+1]}  

    # Sample random noise levels from the schedule
    temp_list = torch.arange(len(noise_schedule)-1, dtype=torch.float32,device='cuda')
    probs = torch.ones_like(temp_list)  # Example probabilities for each item -> wenn jedes item 1 hat, wird es gleich häufig gesampled
    num_samples = FLAGS.batch_size # Number of samples

    # Sample with replacement
    temperature = torch.multinomial(probs, num_samples, replacement=True)

    if FLAGS.diffusion_type == "vexp":
        sample = sample_vexp
    elif FLAGS.diffusion_type == "vpres":
        sample = sample_vpres
    else:
        raise NotImplementedError

    # Sample random rotations
    pos_quat = torch.from_numpy(onp.copy(batch['pos_quat'])).to(temperature.device)
    sampled = sample(pos_quat, temperature)
    return sampled

 
class SimpleMLP(nn.Module):
    def __init__(self,input_size=10): # input size: 9 von der rotationsmatrix, 1 von variance
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

    def forward(self, x, s):

        # konvertiere quaternion zu rotation matrix
        rot_mat = lietorch.SO3(x).matrix()[:,:3,:3]
        rot_mat = rot_mat.reshape(-1,9)

        concat = torch.cat([rot_mat, s], dim=-1)
        out = self.mlp(concat)

        mu = self.layer_mu(out)
        R1 = mu[:,0:3] / torch.norm(mu[:,0:3], dim=-1, keepdim=True)
        R3 = torch.cross(R1, mu[:, 3:], dim=-1)
        R3 = R3 / torch.norm(R3, dim=-1, keepdim=True)
        R2 = torch.cross(R3, R1, dim = -1)

        rotation_matrix = torch.stack([R1,R2,R3],dim=-1)

        quat = lietorch.SO3(rotation_matrix,from_rotation_matrix=True)

        scale = self.layer_scale(mu)
        scale = nn.functional.softplus(scale) + 0.0001
        return quat, scale

def train(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    # yn1 = torch.from_numpy(np.array(batch['yn+1']))
    # sn1 = torch.from_numpy(np.array(batch['sn+1']))
    yn1 = batch['yn+1'].cuda()
    sn1 = batch['sn+1'].cuda()

    mu, scale = model(yn1, sn1.reshape(-1,1))

    def fn(x, y, mu, scale):
        mu = mu * lietorch.SO3(y)# apply residual rotation
        dist = IsotropicGaussianSO3_no_vmap(mu, scale, 
                                    force_small_scale=True)

        prob_dist = dist.log_prob(x)
        # prob_dist = torch.vmap(lambda m,s,q: IsotropicGaussianSO3(m,s,force_small_scale=True).log_prob(q),
        #                         randomness="different")(mu, scale, x)
        return prob_dist # shape 512
    
    yn = batch['yn']
    ynp1 = batch['yn+1']
    loss = (-fn(yn, ynp1, mu, scale)).mean()
    # print(loss)
    loss.backward()
    optimizer.step()
    return loss.item()

def main(_):
    run = datetime.datetime.now()
    run = str(run).replace(" ","_")
    run = run.replace(run.split(".")[-1],"")
    run = FLAGS.dataset + "_" + run + "_" + FLAGS.diffusion_type
    output_dir = FLAGS.output_dir  + run + "/"

    Path(output_dir +'/model').mkdir(parents=True, exist_ok=True)
    Path(output_dir +'/vis').mkdir(parents=True, exist_ok=True)

    # Instantiate the network
    model = SimpleMLP().cuda()
    
    rng_seq = hk.PRNGSequence(42)
    
    if FLAGS.diffusion_type == "vexp":
        noise_schedule = torch.linspace(0.05, 1.25, FLAGS.n_steps).cuda()
        noise_schedule = noise_schedule**2 + 0.0001
    elif FLAGS.diffusion_type == "vpres":
        beta = torch.linspace(0.0001, 0.04, FLAGS.n_steps).cuda()
        # This corresponds to alpha, it starts from 1, i.e. almost no change in the image
        noise_schedule = torch.cumprod(1 - beta, dim=0).cuda()
    else:
        raise NotImplementedError
 
    if FLAGS.train:
        distances = []
        c2_scores = []
        losses = []
        eval_steps = []
        # Open the dataset
        print("load dataset")
        dset = tfds.load(FLAGS.dataset, split="train")
        print("Dataset loaded")
        dset = dset.repeat()
        dset = dset.shuffle(buffer_size=10000)
        dset = dset.batch(FLAGS.batch_size)
        # dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dset = dset.prefetch(buffer_size=-1)
        dset = dset.as_numpy_iterator()
        _ = next(dset)
        # old tensorflow 2.12.0
        
        # Initialize weights
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

        print('training begins')
        for step in tqdm(range(FLAGS.training_steps)):
            batch = get_batch(next(dset), next(rng_seq), noise_schedule)
            # Sampling another batch if the current one had a NaN
            while torch.isnan(batch['yn+1']).sum()>0:
                print("WARNING: Skipped a batch because it had a NaN")
                batch = get_batch(next(dset), next(rng_seq), noise_schedule)

            # adjust learning rate
            if step == 80_000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FLAGS.learning_rate * 0.1
                    print(f"set to {FLAGS.learning_rate * 0.1}")
                    
            elif step == 280_000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FLAGS.learning_rate * 0.01

            loss = train(model,batch,optimizer)
            losses.append(loss)
            if step%50==0:
                print(loss)

            if step in [1000] or (step>1000 and step % 20000 == 0):
                model.eval()
                dis, c2_score = eval(rng_seq,noise_schedule, model, output_dir,step)
                c2_scores.append(c2_score)
                distances.append(dis.cpu().numpy())
                eval_steps.append(step)
                fig,ax = plt.subplots()
                ax.plot(eval_steps,distances)
                plt.savefig(output_dir+ '/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_distances.png')

                fig,ax = plt.subplots()
                ax.plot(eval_steps,c2_scores)
                plt.savefig(output_dir+ '/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_c2sts.png')

                fig,ax = plt.subplots()
                ax.plot(losses)
                plt.savefig(output_dir+ '/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_losses.png')

            if step%10000 ==0:
                state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_model-{step}.pt'
                torch.save(model.state_dict(),state_dict_path)

        state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_model-final.pt'
        torch.save(model.state_dict(),state_dict_path)

    state_dict_path = output_dir+'/model/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + '_model-final.pt'
    torch.save(model.state_dict(),state_dict_path)


if __name__ == "__main__":
    app.run(main)
