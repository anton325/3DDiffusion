import torch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.isotropic_gaussian_no_vmap import IsotropicGaussianSO3 as IsotropicGaussianSO3_no_vmap
from gecco_torch.structs import Mode
import lietorch
from tqdm import tqdm
import math


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

    x_sigma = reverse_sigma(sigma, torch.log(torch.tensor(0.002, device=sigma.device)), torch.log(torch.tensor(165, device=sigma.device)))
    x_sigma_plus1 = x_sigma + (1/128)
    sigma_plus1 = get_sigma(x_sigma_plus1, torch.log(torch.tensor(0.002, device=sigma.device)), torch.log(torch.tensor(165, device=sigma.device)))

    # sigma_discrete_schedule = self.schedule.return_schedule(128) 
    # sigma_discrete_index = (sigma_discrete_schedule < sigma.squeeze(1)).sum(dim=1) - 1

    # scale = sigma_discrete_schedule[sigma_discrete_index].repeat_interleave(examples.shape[1])
    # scalenplus1 = sigma_discrete_schedule[sigma_discrete_index + 1].repeat_interleave(examples.shape[1])
    scale = sigma.repeat_interleave(examples.shape[1])
    scalenplus1 = sigma_plus1.repeat_interleave(examples.shape[1])

    delta = scalenplus1**2 - scale**2

    q = examples[:,:,10:14]
    n = q.shape[0]
    q = q.reshape(-1,4).contiguous()

    axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                            s_single,
                                                            force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                        randomness="different")(q, scale)
    qn = (lietorch.SO3(q) * lietorch.SO3.exp(axis_angles)).vec()

    delta_scale = torch.sqrt(delta * torch.ones(q.shape[0],device=qn.device))

    axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                            s_single,
                                                            force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                        randomness="different")(qn, delta_scale)
    qnplus1 = (lietorch.SO3(qn) * lietorch.SO3.exp(axis_angles)).vec()

    qn = qn.reshape(n,-1,4)
    qnplus1 = qnplus1.reshape(n,-1,4)

    """
    noise non-rotations
    """
    noise = torch.randn_like(ex_diff) * sigma
    noised = ex_diff + noise

    """
    Das Netzwerk erwartet die Daten im Format [xyz, rgb, scales, opacity, rotations]
    """
    noised_data = torch.cat([noised[:,:,:10], qnplus1],dim=-1)

    D_yn = net(noised_data, sigma, context)

    mu = D_yn[:,:,10:14]
    scale = D_yn[:,:,14]

    def fn(x, y, mu, scale):
        x = x.reshape(-1,4)
        scale = scale.reshape(-1,1)
        mu = lietorch.SO3(mu.reshape(-1,4).contiguous())
        y = y.reshape(-1,4).contiguous()
        mu = mu * lietorch.SO3(y) # apply residual rotation
        # dist = IsotropicGaussianSO3(mu, scale, 
        #                             force_small_scale=FORCE_SMALL_SCALE)

        # prob_dist = dist.log_prob(x) 
        dist = IsotropicGaussianSO3_no_vmap(mu, scale, 
                                    force_small_scale=FORCE_SMALL_SCALE)

        prob_dist = dist.log_prob(x)
        return prob_dist # shape 512

    rotation_loss = (-fn(qn, qnplus1, mu, scale)).mean()

    log_fun("rotation_loss",rotation_loss,on_step=True)

    gecco_loss = self.loss_scale * weight * ((D_yn[:,:,:10] - ex_diff) ** 2)

    log_fun("gecco_loss",gecco_loss.mean(),on_step=True)

    if Mode.rgb in self.mode:
        log_fun("mean_rgb_loss",gecco_loss[:,:,3:6].mean(),on_step=True)

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


def fn_sample(x, delta_mu, s):
    """
    rotationen im xyzw format
    """
    batch_size = x.shape[0]
    x = x.reshape(-1,4)
    delta_mu = delta_mu.reshape(-1,4)
    s = s.reshape(-1,1)
    rotated_mu = (lietorch.SO3(delta_mu) * lietorch.SO3(x.type(delta_mu.dtype))).vec()
    axis_angles = torch.vmap(lambda r,s_single: IsotropicGaussianSO3(r,
                                                                s_single,
                                                                force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                            randomness="different")(rotated_mu, s)
    samples = (lietorch.SO3(rotated_mu) * lietorch.SO3.exp(axis_angles)).vec()
    samples = samples.reshape(batch_size,-1,4)
    return samples

def sample(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    
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

    # latents is in shape of desired output (batchsize,points in pointcloud, 3)
    latents = torch.randn((shape[0],shape[1],10), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # noisy start rotations
    initial_noisy_rotation = lietorch.SO3([],from_uniform_sampled=(shape[0]*shape[1])).vec().cuda().reshape(shape[0],shape[1], 4) # in xyzw format

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    # combine normal noise with rotation noise
    x_next_rotations = initial_noisy_rotation

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur = x_next
        x_cur_rotations = x_next_rotations

        # noise rotations

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur #+ gamma * t_cur
        # noise = torch.randn((x_cur.shape[0],x_cur.shape[1],10), device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        # x_hat = x_cur[:,:,:10] #+ extra_noise

        """
        x_hat shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_cur, x_cur_rotations],dim=-1)

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
        denoised_rotations_scale = denoised[:,:,10:]
        denoised_rotations = denoised_rotations_scale[:,:,:4]
        denoised_scale = denoised_rotations_scale[:,:,4]
        x_next_rotations = fn_sample(x_cur_rotations, denoised_rotations, denoised_scale)

        denoised_no_rotations = denoised[:,:,:10]

        """
        x_hat_no_rotations shape xyz, rgb, scale, opacity
        """

        d_cur = (x_cur - denoised_no_rotations) / t_hat # d_i in pseudo code
        x_next = x_cur + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     x_in = torch.cat([x_next, x_next_rotations],dim=-1) 
        #     denoised = self( # D_theta in paper
        #         x_in.to(dtype),
        #         t_next.repeat(B).to(dtype),
        #         context,
        #         post_context,
        #     ).to(torch.float64)
        #     """ 
        #     denoised shape xyz, rgb, scale, opacity, rotations, scale
        #     """
        #     x_next_no_rotations = x_next[:,:,:10]
        #     denoised_no_rotations = denoised[:,:,:10]

        #     d_prime = (x_next_no_rotations - denoised_no_rotations) / t_next # d'_i in pseudo code
        #     x_next = x_hat_no_rotations + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code
        #     x_next = torch.cat([x_next, new_sampled_rotations],dim=-1)
        #     """
        #     x_next shape xyz, rgb, scale, opacity, rotations
        #     """

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_next, context)
    rotations = x_next_rotations


    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations # rotations im xyzw format
    """
    return data

@torch.no_grad()
def sample_2step(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

    rho = kwargs["rho"]
    with_pbar = kwargs["with_pbar"]

    device = self.example_param.device
    dtype = self.example_param.dtype
    if rng is None:
        rng = torch.Generator(device).manual_seed(42)

    B = shape[0] # batch size
    
    post_context = self.conditioner(context)

    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    latents = torch.randn((shape[0], shape[1], 10), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # noisy start rotations
    initial_noisy_rotation = lietorch.SO3([],from_uniform_sampled=(shape[0]*shape[1])).vec().cuda().reshape(shape[0],shape[1], 4) # in xyzw format

    x_next = latents.to(torch.float64) * t_steps[0]

    x_next_rotations = initial_noisy_rotation

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur = x_next
        x_cur_rotations = x_next_rotations

        """
        x_hat shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_cur, x_cur_rotations],dim=-1)

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
        denoised_rotations_scale = denoised[:,:,10:]
        denoised_rotations = denoised_rotations_scale[:,:,:4]
        denoised_scale = denoised_rotations_scale[:,:,4]
        x_next_rotations = fn_sample(x_cur_rotations, denoised_rotations, denoised_scale)

        denoised_no_rotations = denoised[:,:,:10]

        """
        x_hat_no_rotations shape xyz, rgb, scale, opacity
        """

        d_cur = (x_cur - denoised_no_rotations) / t_cur # d_i in pseudo code
        x_next = x_cur + (t_next - t_cur) * d_cur # x_i+1 in pseudo code
        # # Apply 2nd order correction.
        if i < num_steps - 1:
            x_in = torch.cat([x_next, x_next_rotations],dim=-1) 
            denoised = self( # D_theta in paper
                x_in.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            """ 
            denoised shape xyz, rgb, scale, opacity, rotations, scale
            """
            x_next_no_rotations = x_next[:,:,:10]
            denoised_no_rotations = denoised[:,:,:10]

            d_prime = (x_next_no_rotations - denoised_no_rotations) / t_next # d'_i in pseudo code
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code
            """
            x_next shape xyz, rgb, scale, opacity, rotations
            """

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_next, context)
    rotations = x_next_rotations


    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations # rotations im xyzw format
    """
    return data

@torch.no_grad()
def sample_2step_both(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

    rho = kwargs["rho"]
    with_pbar = kwargs["with_pbar"]

    device = self.example_param.device
    dtype = self.example_param.dtype
    if rng is None:
        rng = torch.Generator(device).manual_seed(42)

    B = shape[0] # batch size
    
    post_context = self.conditioner(context)

    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    latents = torch.randn((shape[0], shape[1], 10), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # noisy start rotations
    initial_noisy_rotation = lietorch.SO3([],from_uniform_sampled=(shape[0]*shape[1])).vec().cuda().reshape(shape[0],shape[1], 4) # in xyzw format

    x_next = latents.to(torch.float64) * t_steps[0]

    x_next_rotations = initial_noisy_rotation

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur = x_next
        x_cur_rotations = x_next_rotations

        """
        x_hat shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_cur, x_cur_rotations],dim=-1)

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
        denoised_rotations_scale = denoised[:,:,10:]
        denoised_rotations = denoised_rotations_scale[:,:,:4]
        denoised_scale = denoised_rotations_scale[:,:,4]
        x_next_rotations = fn_sample(x_cur_rotations, denoised_rotations, denoised_scale)

        denoised_no_rotations = denoised[:,:,:10]

        """
        x_hat_no_rotations shape xyz, rgb, scale, opacity
        """

        d_cur = (x_cur - denoised_no_rotations) / t_cur # d_i in pseudo code
        x_next = x_cur + (t_next - t_cur) * d_cur # x_i+1 in pseudo code
        # # Apply 2nd order correction.
        if i < num_steps - 1:
            x_in = torch.cat([x_next, x_next_rotations],dim=-1) 
            denoised = self( # D_theta in paper
                x_in.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            """ 
            denoised shape xyz, rgb, scale, opacity, rotations, scale
            """
            denoised_rotations_scale = denoised[:,:,10:]
            denoised_rotations = denoised_rotations_scale[:,:,:4]
            denoised_scale = denoised_rotations_scale[:,:,4]
            x_next_rotations = fn_sample(x_next_rotations, denoised_rotations, denoised_scale)

            x_next_no_rotations = x_next[:,:,:10]
            denoised_no_rotations = denoised[:,:,:10]

            d_prime = (x_next_no_rotations - denoised_no_rotations) / t_next # d'_i in pseudo code
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code
            """
            x_next shape xyz, rgb, scale, opacity, rotations
            """

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_next, context)
    rotations = x_next_rotations


    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations # rotations im xyzw format
    """
    return data


@torch.no_grad()
def sample_1step(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

    rho = kwargs["rho"]
    with_pbar = kwargs["with_pbar"]

    device = self.example_param.device
    dtype = self.example_param.dtype
    if rng is None:
        rng = torch.Generator(device).manual_seed(42)

    B = shape[0] # batch size
    
    post_context = self.conditioner(context)

    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    latents = torch.randn((shape[0], shape[1], 10), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # noisy start rotations
    initial_noisy_rotation = lietorch.SO3([],from_uniform_sampled=(shape[0]*shape[1])).vec().cuda().reshape(shape[0],shape[1], 4) # in xyzw format

    x_next = latents.to(torch.float64) * t_steps[0]

    x_next_rotations = initial_noisy_rotation

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur = x_next
        x_cur_rotations = x_next_rotations

        """
        x_hat shape xyz, rgb, scale, opacity, rotations
        """
        x_in = torch.cat([x_cur, x_cur_rotations],dim=-1)

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
        denoised_rotations_scale = denoised[:,:,10:]
        denoised_rotations = denoised_rotations_scale[:,:,:4]
        denoised_scale = denoised_rotations_scale[:,:,4]
        x_next_rotations = fn_sample(x_cur_rotations, denoised_rotations, denoised_scale)

        denoised_no_rotations = denoised[:,:,:10]

        """
        x_hat_no_rotations shape xyz, rgb, scale, opacity
        """

        d_cur = (x_cur - denoised_no_rotations) / t_cur # d_i in pseudo code
        x_next = x_cur + (t_next - t_cur) * d_cur # x_i+1 in pseudo code

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_next, context)
    rotations = x_next_rotations


    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations # rotations im xyzw format
    """
    return data