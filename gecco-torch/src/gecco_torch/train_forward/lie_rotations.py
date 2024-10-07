import torch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.additional_metrics.metrics_so3 import rotational_distance_between_pairs, rotational_distance_between_pairs_dot_product
from gecco_torch.structs import Mode
import lietorch
from tqdm import tqdm
import math


FORCE_SMALL_SCALE = False

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

    denoised_rotation = D_yn[:,:,10:14]

    rotational_loss = rotational_distance_between_pairs(q.reshape(-1,4), denoised_rotation.reshape(-1,4)).mean()
    log_fun("rotational_loss", rotational_loss, on_step=True)

    gecco_loss = self.loss_scale * weight * ((D_yn[:,:,:10] - ex_diff) ** 2)

    log_fun("gecco_loss",gecco_loss.mean(),on_step=True)

    if Mode.rgb in self.mode:
        log_fun("mean_rgb_loss",gecco_loss[:,:,3:6].mean(),on_step=True)
    else:
        log_fun("mean_sh_loss",gecco_loss[:,:,3:6].mean(),on_step=True)

    log_fun("mean_scale_loss",gecco_loss[:,:,6:9].mean(),on_step=True)
    log_fun("mean_opacity_loss",gecco_loss[:,:,9].mean(),on_step=True)
    log_fun("mean_xyz_loss",gecco_loss[:,:,:3].mean(),on_step=True)

    data = net.reparam.diffusion_to_data(D_yn[:,:,:10],context)
    
    data = torch.cat([data[:,:,:10], D_yn[:,:,10:]],dim=-1)

    mean_loss = gecco_loss.mean() + 50 * rotational_loss # rotation loss ~ im bereich 0 bis pi

    # fürs gaussian splat loss nehme ground truth für die rotations
    # data[:,:,10:14] = q.reshape(n,-1,4)    
    """
    data shape xyz, rgb, scales, opacity, rotations
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
    F_x output format xyz, rgb, scales, opacity, rotation
    """
    # output conditioning: c_skip * x + c_out * F_x
    skip_result = c_skip * x[:,:,:10]
    out_result = c_out * F_x[:,:,:10]
    denoised = skip_result + out_result 
    denoised = torch.cat([denoised,F_x[:,:,10:]],dim=-1)

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
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur # + gamma * t_cur
        # noise = torch.randn((batch_size, n, 10), device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        """
        x_in shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_cur_rest, x_cur_rotation], dim=-1)

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
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur # + gamma * t_cur
        # noise = torch.randn((batch_size, n, 10), device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        x_hat = x_cur_rest # + extra_noise

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

        x_denoised_rotation = fn_sample(denoised_rotations, denoised_scale)

        # make step (fro x_cur_rotation to x_denoised_rotation)
        dif_rotation = lietorch.SO3(x_cur_rotation).inv() * lietorch.SO3(x_denoised_rotation.type(x_cur_rotation.dtype))
        log_dif = dif_rotation.log() / t_cur
        step_rotation = lietorch.SO3.exp(-(t_next - t_hat) * log_dif)
        x_next_rotation = (lietorch.SO3(x_cur_rotation) * step_rotation).vec()

        print(f"rotational distance start ende: {rotational_distance_between_pairs(x_cur_rotation.reshape(-1,4), x_denoised_rotation.type(x_cur_rotation.dtype).reshape(-1,4)).mean()}")
        print(f"rotational distance step ende: {rotational_distance_between_pairs(x_next_rotation.reshape(-1,4), x_denoised_rotation.type(x_cur_rotation.dtype).reshape(-1,4)).mean()}")


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

            denoised_rotations_2nd = denoised[:,:,10:14]
            denoised_scale_2nd = denoised[:,:,14]

            x_denoised_rotation_2nd = fn_sample(denoised_rotations_2nd, denoised_scale_2nd)

            # make step (fro x_cur_rotation to x_denoised_rotation)
            dif_rotation_2nd = lietorch.SO3(x_next_rotation).inv() * lietorch.SO3(x_denoised_rotation_2nd.type(x_next_rotation.dtype))
            log_dif_2nd = dif_rotation_2nd.log() / t_next
            step_rotation_2nd = lietorch.SO3.exp(-(t_next - t_hat) * (log_dif / 2 + log_dif_2nd / 2))

            x_next_rotation_2nd = (lietorch.SO3(x_cur_rotation) * step_rotation_2nd).vec()

            print(f"rotational distance start ende: {rotational_distance_between_pairs(x_next_rotation.reshape(-1,4), x_denoised_rotation_2nd.reshape(-1,4).type(x_next_rotation.dtype)).mean()}")   
            print(f"rotational distance step ende: {rotational_distance_between_pairs(x_next_rotation_2nd.reshape(-1,4),x_denoised_rotation_2nd.reshape(-1,4).type(x_next_rotation.dtype)).mean()}")
            x_next_rotation = x_next_rotation_2nd

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