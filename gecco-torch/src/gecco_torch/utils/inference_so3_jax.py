# Main script used to train a particular model on a particular dataset.
from absl import app
from absl import flags

import sys
sys.path.append('../')

import haiku as hk
import pickle
import jax.numpy as jnp

import haiku as hk
from pathlib import Path
import numpy as np
import numpy as onp
import jax
import jax.numpy as jnp
import jaxlie
import lietorch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_plotting import visualize_so3_density, visualize_so3_probabilities
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from tqdm import tqdm

from gecco_torch.additional_metrics.metrics_so3 import c2st, minimum_distance

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "so3models/so3ddpm/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 1024, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate for the optimizer.")
# flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.") # 400_000
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
# flags.DEFINE_integer("test_nsamples", 100_000, "Number of samples to draw at testing time.")
flags.DEFINE_integer("test_nsamples", 100_000, "Number of samples to draw at testing time.")
flags.DEFINE_string("input_rotation_param", "matrix", "Parameterisation of the rotation at the input of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("output_rotation_param", "matrix", "Parameterisation of the rotation at the output of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("diffusion_type", "vexp", "Variance preserving or variance exploding diffusion 'vexp' or 'vpres'") 

flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")

flags.DEFINE_integer("n_steps", 256, "Number of steps in noise schedule")

flags.DEFINE_integer("n_folds", 5, "Number of folds in c2st")
    
FLAGS = flags.FLAGS

@torch.no_grad()
def eval(rng_seq,noise_schedule, model, output_dir, step="eval", params=None):
    print("eval...")
    # Starting sampling from the trained model
    if FLAGS.diffusion_type == "vexp":
        X0 = lietorch.SO3([],from_uniform_sampled=FLAGS.test_nsamples).vec()

    @torch.no_grad()
    def fn_sample(x, delta_mu, s, key):
        # sampled = IsotropicGaussianSO3((lietorch.SO3(delta_mu.cuda()) * lietorch.SO3(x.cuda())).vec(), s, force_small_scale=True).sample()
        sampled = IsotropicGaussianSO3((lietorch.SO3(delta_mu.cuda()) * lietorch.SO3(x.cuda())).vec(), s, force_small_scale=True).sample_jaxseed(key)
        return sampled

    x_t = X0.cuda()

    print("sampling...")
    dir = Path(output_dir + '/vis/' + FLAGS.dataset +'_'+ FLAGS.diffusion_type + f'_sampling_{step}')
    print(dir)
    dir.mkdir(exist_ok=True)
    with torch.no_grad():
        every = 16
        for j,sn in tqdm(enumerate(torch.flip(noise_schedule,[0]))):
            mu, s = model.apply(params, x_t, sn*torch.ones([FLAGS.test_nsamples,1]).cuda())
            x_t_new = torch.zeros_like(x_t)
            batch_size = 10000
            for i in range(0,mu.shape[0],batch_size):
                x_t_new[i:i+batch_size] = fn_sample(x_t[i:i+batch_size], mu[i:i+batch_size], s[i:i+batch_size],next(rng_seq))
            x_t = x_t_new
            # if j % every == 0 or j==255:
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
            true_samp = jax.vmap(lambda m: jaxlie.SO3.from_matrix(m).wxyz)(true_samp) # print(X.shape)

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


def model_fn(x,s):
    # intern nimmt das model als input eine rotation matrix (flattened)
    x = jnp.array(x.cpu().numpy()[:,[3,0,1,2]]) # xyzw zu wxyz
    x = jax.vmap(lambda u: jaxlie.SO3(u).as_matrix().flatten())(x)

    s = jnp.array(s.cpu().numpy())

    net = jnp.concatenate([x,s],axis=-1)
    net = hk.nets.MLP([256, 256, 256, 256, 256], activation=jax.nn.leaky_relu)(net)

    # Note that we only output the residual with respect to the input rotation

    # construct a valid rotation matrix (all rows are orthonormal)
    net = hk.Linear(6)(net)
    R1 = net[:,0:3] / jnp.linalg.norm(net[:,0:3], axis=-1, keepdims=True)
    R3 = jnp.cross(R1, net[:, 3:],axis=-1)
    R3 = R3 / jnp.linalg.norm(R3, axis=-1, keepdims=True)
    R2 = jnp.cross(R3, R1)
    delta_mu = jnp.stack([R1, R2, R3], axis=-1) # rotation matrix
    delta_mu = jax.vmap(lambda u: jaxlie.SO3.from_matrix(u).wxyz)(delta_mu)

    scale = jax.nn.softplus(hk.Linear(1)(net)) + 0.0001

    delta_mu = torch.from_numpy(np.array(delta_mu))[:,[1,2,3,0]] # von wxyz zu xyzw
    scale = torch.from_numpy(np.array(scale))
    return delta_mu, scale



def main(_):
    run = FLAGS.dataset + "_" + "eval_jaxmodel"
    output_dir = FLAGS.output_dir  + run + "/"

    Path(output_dir +'/vis').mkdir(parents=True, exist_ok=True)

    # Instantiate the network
    model = hk.without_apply_rng(hk.transform(model_fn))
    with open('/home/giese/Documents/SO3DiffusionModels/models/so3ddpm/checkerboard_vexp_model_20000_.pckl', 'rb') as file:
        params = pickle.load(file)
    
    rng_seq = hk.PRNGSequence(42)
    
    if FLAGS.diffusion_type == "vexp":
        noise_schedule = torch.linspace(0.05, 1.25, FLAGS.n_steps).cuda()
        noise_schedule = noise_schedule**2 + 0.0001

    dis, c2_score = eval(rng_seq,noise_schedule, model, output_dir, params=params)

if __name__ == "__main__":
    app.run(main)
