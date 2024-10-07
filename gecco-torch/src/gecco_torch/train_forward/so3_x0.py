import torch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_gaussian_no_vmap import IsotropicGaussianSO3 as IsotropicGaussianSO3_no_vmap
from gecco_torch.additional_metrics.metrics_so3 import rotational_distance_between_pairs, rotational_distance_between_pairs_dot_product
from gecco_torch.structs import Mode
import lietorch
from tqdm import tqdm
import math
from scipy.stats import multivariate_normal
import numpy as np


FORCE_SMALL_SCALE = False

def reverse_sigma(sigma, log_sigma_min, log_sigma_max):
    u = (torch.log(sigma) - log_sigma_min) / (log_sigma_max - log_sigma_min)
    return u

def get_sigma(u , log_sigma_min, log_sigma_max):
    return torch.exp(u * (log_sigma_max - log_sigma_min) + log_sigma_min)

def forward(self, net, examples, context, train_step, log_fun):
    """
    Wenn die Daten reinkommen, sind die aufgeteilt als [xyz, rgb, scales, opacity, rotations] rotations xyzw
    """

    data_no_rot = examples[:,:,:10]
    # net.reparam.mean = net.reparam.mean.to(data_no_rot.device)
    # net.reparam.sigma = net.reparam.sigma.to(data_no_rot.device)
    ex_diff = net.reparam.data_to_diffusion(data_no_rot, context) # reparametrisierte Punktwolke

    sigma = self.schedule(ex_diff) # shape [batchsize, 1, 1]
    weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)


    scale = sigma.squeeze(-1).squeeze(-1).repeat_interleave(examples.shape[1])

    q = examples[:,:,10:14]
    batch_size = q.shape[0]
    q = q.reshape(-1,4).contiguous()

    axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                            s_single,
                                                            force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                        randomness="different")(q, scale)
    qn = (lietorch.SO3(q) * lietorch.SO3.exp(axis_angles)).vec()

    qn = qn.reshape(batch_size,-1,4)

    """
    noise non-rotations
    """
    noise = torch.randn_like(ex_diff) * sigma
    noised = ex_diff + noise

    """
    Das Netzwerk erwartet die Daten im Format [xyz, rgb, scales, opacity, rotations]
    """
    noised_data = torch.cat([noised, qn],dim=-1)

    D_yn = net(noised_data, sigma, context)

    mu = D_yn[:,:,10:14]
    scale = D_yn[:,:,14]

    def fn(y, mu, scale):
        scale = scale.reshape(-1,1)
        mu = lietorch.SO3(mu.reshape(-1,4).contiguous())
        y = y.reshape(-1,4).contiguous()
        # mu = mu * lietorch.SO3(y) # apply residual rotation -> nö wir sagen direkt x0 voraus

        dist = IsotropicGaussianSO3_no_vmap(mu, scale, 
                                    force_small_scale=FORCE_SMALL_SCALE)

        prob_dist = dist.log_prob(y)
        return prob_dist # shape 512

    small_scale_loss = scale.mean() / 2 # damit er nicht einfach ne riesige scale vorhersagt
    log_prob_loss = (-fn(q, mu, scale)).mean()
    rotation_loss = log_prob_loss + small_scale_loss

    log_fun("small_scale_loss", small_scale_loss, on_step=True)
    log_fun("log_prob_loss", log_prob_loss, on_step=True)
    log_fun("rotation_loss",rotation_loss,on_step=True)

    gecco_loss = self.loss_scale * weight * ((D_yn[:,:,:10] - ex_diff) ** 2)

    log_fun("gecco_loss", gecco_loss.mean(), on_step=True)

    if Mode.rgb in self.mode:
        log_fun("mean_rgb_loss",gecco_loss[:,:,3:6].mean(),on_step=True)
    else:
        log_fun("mean_sh_loss",gecco_loss[:,:,3:6].mean(),on_step=True)

    log_fun("mean_scale_loss",gecco_loss[:,:,6:9].mean(),on_step=True)
    log_fun("mean_opacity_loss",gecco_loss[:,:,9].mean(),on_step=True)
    log_fun("mean_xyz_loss",gecco_loss[:,:,:3].mean(),on_step=True)

    data = net.reparam.diffusion_to_data(D_yn[:,:,:10],context)
    
    data = torch.cat([data[:,:,:10], D_yn[:,:,10:]],dim=-1)

    mean_loss = gecco_loss.mean() + 2 * rotation_loss # rotation loss ~ im bereich -9 bis 6

    # fürs gaussian splat loss nehme ground truth für die rotations
    # data[:,:,10:14] = q.reshape(n,-1,4)
    """
    data shape xyz, rgb, scales, opacity, rotations, scale
    rotations in xyzw
    """
    return data, mean_loss, sigma



def forward_preconditioning(
        self,
        x,
        sigma,
        raw_context,  # raw_context comes from the dataset, before any preprocessing
        post_context,  # post_context comes from the conditioner
        do_cache = False,  # whether to return a cache of inducer states for upsampling
        cache = None,  # cache of inducer states for upsampling
):
    def ones(n: int):
        return (1,) * n
    
    """
    Das netzwerk kriegt die Daten im Format [xyz, rgb, scales, opacity, rotations]
    """
    sigma = sigma.reshape(-1, *ones(x.ndim - 1))

    # sind das alles nur die preconditioning calculations -> scale inputs according to the 
    # "Elucidating the Design Space of Diffusion-Based Generative Models" by Kerras et al. 
    # p. 3 section "any"
    c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
    # print(f"c_skip: {c_skip}")
    c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
    # print(f"c_out: {c_out}")
    c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
    # print(f"c_in: {c_in}")
    c_noise = sigma.log() / 4
    # print(f"c_noise: {c_noise}")
    c_noise = c_noise

    # alle werte eines batches werden mit dem zugehörigen c_in scalar multipliziert


    """
    x ist im format xyz, rgb, scales, opacity, rotations
    """
    model_in = torch.concat([c_in * x[:,:,:10],x[:,:,10:]],dim=-1)
    
    F_x, cache = self.model(
        model_in, c_noise, raw_context, post_context, do_cache, cache
    )
    """
    F_x output format xyz, rgb, scales, opacity, quat, scale
    """
    # output conditioning: c_skip * x + c_out * F_x
    skip_result = c_skip * x[:,:,:10]
    out_result = c_out * F_x[:,:,:10]
    denoised = skip_result + out_result 
    denoised = torch.cat([denoised, F_x[:,:,10:]],dim=-1)

    """
    denoised output format xyz, rgb, scales, opacity, rotations, scale
    """
    
    if not do_cache:
        return_dict = {
            'denoised' : denoised,
        }
    else:
        return_dict = {
        'denoised' : denoised,
        'cache' : cache
    }
    return return_dict


def fn_sample(x, s):
    """
    rotationen im xyzw format
    """
    batch_size = x.shape[0]
    x = x.reshape(-1,4)
    s = s.reshape(-1,1)
    axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                s_single,
                                                                force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                            randomness="different")(x, s)
    samples = (lietorch.SO3(x) * lietorch.SO3.exp(axis_angles)).vec()
    samples = samples.reshape(batch_size,-1,4)
    return samples

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
def sample(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    """
    1st order ddpm sampling für rotation
    """
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

    # num_steps = kwargs["actual_steps"]
    # sigma_max = kwargs['sigma_actual_max']

    rho = kwargs["rho"]
    S_churn = kwargs["S_churn"]
    S_min = kwargs["S_min"]
    S_max = kwargs["S_max"]
    S_noise = kwargs["S_noise"]
    with_pbar = kwargs["with_pbar"]

    device = self.example_param.device
    dtype = self.example_param.dtype
    if rng is None:
        rng = torch.Generator(device).manual_seed(42)

    B = shape[0] # batch size
    


    # context is features of the reference image -> generate pointcloud for that image
    # only used once - as input to SetTransformer
    post_context = self.conditioner(context)

    # Time step discretization.
    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
    # print(f"t steps in sampling(): {t_steps.shape}")


    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)

    batch_size = shape[0]
    n = shape[1]

    # latents is in shape of desired output (batchsize,points in pointcloud, 3)
    latents = torch.randn((batch_size, n, 10), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # noisy start rotations
    initial_noisy_rotation = lietorch.SO3([],from_uniform_sampled=(batch_size * n)).vec().cuda().reshape(batch_size, n, 4) # in xyzw format

    # Main sampling loop.
    x_next_rest = latents.to(torch.float64) * t_steps[0]

    x_next_rotation = initial_noisy_rotation

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur_rest = x_next_rest
        x_cur_rotation = x_next_rotation

        # noise rotations

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, math.sqrt(2.0) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = t_cur + gamma * t_cur
        noise = torch.randn((batch_size, n, 10), device=device, generator=rng, dtype=dtype) * S_noise
        extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        x_hat = x_cur_rest + extra_noise

        """
        x_in shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_hat, x_cur_rotation], dim=-1)

        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        """
        denoised shape xyz, rgb, scale, opacity, rotations, scale
        """
        denoised_rest = denoised[:,:,:10]

        denoised_rotations = denoised[:,:,10:14]
        denoised_scale = denoised[:,:,14]

        x_next_rotation = fn_sample(denoised_rotations, denoised_scale)

        d_cur = (x_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_next_rest = x_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # Apply 2nd order correction.
        if i < num_steps - 1:

            x_in = torch.cat([x_next_rest, x_next_rotation], dim=-1)

            denoised = self( # D_theta in paper
                x_in.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            """ 
            denoised shape xyz, rgb, scale, opacity, rotations, scale
            """
            denoised_rest = denoised[:,:,:10]

            d_prime = (x_next_rest - denoised_rest) / t_next # d'_i in pseudo code
            x_next_rest = x_next_rest + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

            # füge bisschen noise zu rotations hinzu
            noisy_rotation = get_noisy_rotation(batch_size * n, t_next)
            x_next_rotation = (lietorch.SO3(x_next_rotation.type(noisy_rotation.dtype).reshape(-1,4)) * noisy_rotation.inv()).vec().reshape(batch_size, n, 4)


    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_next_rest, context)
    rotations = x_next_rotation


    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations # rotations im xyzw format
    """
    return data


def likelihood(
        model, 
        batch,
):
    kwargs = {
        'reverse_ode' : True,
        'gt_data' : batch.data,
        }
    
    reverse_sample = sample_rot_step(model, batch.data.shape, batch.ctx, None, **kwargs)
    dimension = reverse_sample.shape[2]
    reverse_sample_np = reverse_sample.cpu().numpy().reshape(reverse_sample.shape[0] * reverse_sample.shape[1], -1)
    mu = np.zeros(dimension)  # Replace with the 13-dimensional mean vector
    sigma = np.diag(165*np.ones(dimension))  # Replace with the 13x13 covariance matrix

    # Initialize the multivariate normal distribution
    mvn = multivariate_normal(mean=mu, cov=sigma)
    prob_densities = mvn.pdf(reverse_sample_np)

    # Calculate the joint likelihood
    sum_likelihood = np.sum(-np.log(prob_densities+1e-10))
    averaged = sum_likelihood / (reverse_sample.shape[0] * reverse_sample.shape[1])
    return averaged

@torch.no_grad()
def sample_rot_step(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    """
    2nd order take steps on rotation
    """

    debug = kwargs.get("debug", False)
    if not debug:
        kwargs = {**self.sampler_kwargs, **kwargs}
        num_steps = kwargs["num_steps"]
        sigma_min = kwargs["sigma_min"]
        sigma_max = kwargs["sigma_max"]

        # num_steps = kwargs["actual_steps"]
        # sigma_max = kwargs['sigma_actual_max']

        rho = kwargs["rho"]
        S_churn = kwargs["S_churn"]
        S_min = kwargs["S_min"]
        S_max = kwargs["S_max"]
        S_noise = kwargs["S_noise"]

        device = self.example_param.device
        dtype = self.example_param.dtype
    else:
        device = kwargs['denoised_rest'].device
        dtype = kwargs['denoised_rest'].dtype

    if rng is None:
        rng = torch.Generator(device).manual_seed(42)
    with_pbar = kwargs.get("with_pbar", False)

    B = shape[0] # batch size
    

    # context is features of the reference image -> generate pointcloud for that image
    # only used once - as input to SetTransformer
    if not debug:
        post_context = self.conditioner(context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
    else:
        t_steps = kwargs['t_steps']
        num_steps = t_steps.shape[0] - 1
    # print(f"t steps in sampling(): {t_steps.shape}")


    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
    batch_size = shape[0]
    n = shape[1]

    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data[:,:,:10], context)
        x_next = latents.to(torch.float64)
        x_next_rest = x_next
        x_next_rotation = gt_data[:,:,10:14]
    else:
        print("Sampling...")

        # latents is in shape of desired output (batchsize, points in pointcloud, 3)
        latents = torch.randn((batch_size, n, 10), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # noisy start rotations
        initial_noisy_rotation = lietorch.SO3([],from_uniform_sampled=(batch_size * n)).vec().cuda().reshape(batch_size, n, 4) # in xyzw format

        # Main sampling loop
        x_next_rest = latents.to(torch.float64) * t_steps[0]

        x_next_rotation = initial_noisy_rotation

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur_rest = x_next_rest
        x_cur_rotation = x_next_rotation

        # noise rotations

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        # noise = torch.randn((batch_size, n, 10), device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        """
        x_in shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_cur_rest, x_cur_rotation], dim=-1)

        if not debug:
            # Euler step.
            denoised = self( # D_theta in pseudo code in paper
                x_in.to(dtype),
                t_cur.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            """
            denoised shape xyz, rgb, scale, opacity, rotations, scale
            """
            denoised_rest = denoised[:,:,:10]

            denoised_rotations = denoised[:,:,10:14]
            denoised_scale = denoised[:,:,14]
            x_denoised_rotation = fn_sample(denoised_rotations, denoised_scale)

        else:
            denoised_rest = kwargs['denoised_rest']
            x_denoised_rotation = kwargs['denoised_rotations']


        step_size = t_next - t_cur

        # make step (fro x_cur_rotation to x_denoised_rotation)
        relative_rotation = lietorch.SO3(x_denoised_rotation.type(x_cur_rotation.dtype)) * lietorch.SO3(x_cur_rotation).inv()
        relative_rotation_log = relative_rotation.log() / t_cur
        small_step_rotvec = -step_size * relative_rotation_log
        small_step_rotation = lietorch.SO3.exp(small_step_rotvec)
        x_next_rotation = (small_step_rotation * lietorch.SO3(x_cur_rotation)).vec()

        # print(f"rotational distance start ende: {rotational_distance_between_pairs(x_cur_rotation.reshape(-1,4), x_denoised_rotation.type(x_cur_rotation.dtype).reshape(-1,4)).mean()}")
        # print(f"rotational distance step ende: {rotational_distance_between_pairs(x_next_rotation.reshape(-1,4), x_denoised_rotation.type(x_cur_rotation.dtype).reshape(-1,4)).mean()}")


        d_cur = (x_cur_rest - denoised_rest) / t_cur # d_i in pseudo code
        x_next_rest = x_cur_rest + step_size * d_cur # x_i+1 in pseudo code

        # Apply 2nd order correction.
        if i < num_steps - 1:

            x_in = torch.cat([x_next_rest, x_next_rotation], dim=-1)
            if not debug:
                denoised = self( # D_theta in paper
                    x_in.to(dtype),
                    t_next.repeat(B).to(dtype),
                    context,
                    post_context,
                ).to(torch.float64)
                """ 
                denoised shape xyz, rgb, scale, opacity, rotations, scale
                """
                denoised_rest = denoised[:,:,:10]
                denoised_rotations_2nd = denoised[:,:,10:14]
                denoised_scale_2nd = denoised[:,:,14]

                x_denoised_rotation_2nd = fn_sample(denoised_rotations_2nd, denoised_scale_2nd)

            else:
                denoised_rest = kwargs['denoised_rest']
                x_denoised_rotation_2nd = kwargs['denoised_rotations']

            d_prime = (x_next_rest - denoised_rest) / t_next # d'_i in pseudo code
            x_next_rest = x_cur_rest + step_size * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code


            # make step (fro x_cur_rotation to x_denoised_rotation)
            relative_rotation = lietorch.SO3(x_denoised_rotation_2nd.type(x_next_rotation.dtype)) * lietorch.SO3(x_next_rotation).inv()
            relative_rotation_log_2nd = relative_rotation.log() / t_next
            direction = relative_rotation_log_2nd / 2 + relative_rotation_log / 2
            small_step_rotation = lietorch.SO3.exp(-step_size * direction)
            x_next_rotation_2nd = (small_step_rotation * lietorch.SO3(x_cur_rotation)).vec()

            # print(f"rotational distance start ende: {rotational_distance_between_pairs(x_next_rotation.reshape(-1,4), x_denoised_rotation_2nd.reshape(-1,4).type(x_next_rotation.dtype)).mean()}")   
            # print(f"rotational distance step ende: {rotational_distance_between_pairs(x_next_rotation_2nd.reshape(-1,4),x_denoised_rotation_2nd.reshape(-1,4).type(x_next_rotation.dtype)).mean()}")
            x_next_rotation = x_next_rotation_2nd

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    if not debug:
        data = self.reparam.diffusion_to_data(x_next_rest, context)
    else:
        data = x_next_rest
    rotations = x_next_rotation

    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations # rotations im xyzw format
    """
    return data

if __name__ == "__main__":
    from gecco_torch.scene.gaussian_model import GaussianModel
    from gecco_torch.diffusionsplat import LogUniformSchedule
    from gecco_torch.utils.render_gaussian_circle import render_gaussian

    gaussian = GaussianModel(3)
    gaussian.load_ply(("/globalwork/giese/gaussians/02958343/8c6c271a149d8b68949b12cf3977a48b/point_cloud/iteration_" + str(10000) +"/point_cloud.ply"))
    schedule = LogUniformSchedule(165)
    t_steps = torch.flip(schedule.return_schedule(128),[0])
    xyz = gaussian._xyz
    color = gaussian._features_dc.squeeze(1)
    scale = gaussian._scaling
    opacity = gaussian._opacity
    rotation = gaussian._rotation
    data = torch.cat([xyz, color, scale, opacity],dim=-1)
    kwargs = {
        'debug' : True,
        'denoised_rest' : data.unsqueeze(0),
        'denoised_rotations' : rotation.unsqueeze(0),
        't_steps' : t_steps,
    }
    gaussian_data = sample_rot_step(None, (1, xyz.shape[0], data.shape[1] + rotation.shape[1]), None, None, **kwargs)
    def insert_gaussian(gaussian, xyz, color, scale, opacity, rotation):
        gaussian._xyz = xyz
        gaussian._features_dc = color
        gaussian._scaling = scale
        gaussian._opacity = opacity
        gaussian._rotation = rotation
        return gaussian
    gaussian_data = gaussian_data.type(torch.float32)
    gaussian = insert_gaussian(gaussian,
                               gaussian_data[0,:,:3].squeeze(0),
                               gaussian_data[0,:,3:6].reshape(gaussian_data.shape[1],1,3),
                               gaussian_data[0,:,6:9].squeeze(0),
                               gaussian_data[0,:,9].squeeze(0).unsqueeze(-1),
                               gaussian_data[0,:,10:].squeeze(0))
    render_gaussian(gaussian)
    pass