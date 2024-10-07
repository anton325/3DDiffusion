import torch
from gecco_torch.structs import Mode
import math
from tqdm import tqdm
import numpy as np
from scipy.stats import multivariate_normal
from gecco_torch.utils.render_tensors import render_no_modifications
from torchvision.utils import save_image
import lietorch

def forward(self, net, examples, context, log_fun, train_step):
    if Mode.procrustes in self.mode:
        ex_diff = torch.cat([net.reparam.data_to_diffusion(examples[:,:,:10], context),examples[:,:,10:]],dim = -1)
    else:
        ex_diff = net.reparam.data_to_diffusion(examples, context) # reparametrisierte Punktwolke
    # print(f"forward shape ex diff: {ex_diff.shape}") # (batchsize,num points, 3)
    sigma = self.schedule(ex_diff)
    # print(f"Sigma: {sigma}")
    # print(f"forward shape sigma: {sigma.shape}") # (batchsize,1,1)
    weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    sigma = sigma #/ 2*torch.linspace(1,sigma.shape[0],sigma.shape[0],device=ex_diff.device).reshape(-1,1,1)
    n = torch.randn_like(ex_diff) * sigma
    noised_data = ex_diff + n

    # data2 = examples.data.clone()
    # sigma[0,0,0] = 0.01
    # n = torch.randn_like(ex_diff) * sigma
    # data2 += n
    # random_samples = [x>0.2 for x in torch.rand(data2.shape[1],device=ex_diff.device)]
    # data2 = data2[:,random_samples,:]
    # data2[:,:,6:9] = torch.ones_like(data2[:,:,6:9]) * -3000
    # data2[:,:,6:9] += torch.randn_like(data2[:,:,6:9]) * 2.5
    # context.camera.world_view_transform[0] = context.camera.world_view_transform[2]
    # data2[:,:,3:6] = torch.ones_like(data2[:,:,3:6]) * 0
    # renderings = render_no_modifications(data2, context, self.mode)
    # save_image(renderings['render'][0], f"image.png")
    D_yn = net(noised_data, sigma, context) # input ist pointcloud, die mit noise verändert wurde, und das sigma

    # if Mode.procrustes in self.mode:
    #     r1 = lietorch.SO3(D_yn[:,:,10:].reshape(-1,3,3), from_rotation_matrix=True)
    #     r2 = lietorch.SO3(ex_diff[:,:,10:].reshape(-1,3,3), from_rotation_matrix=True)
    #     rel = r1* r2.inv()
    #     traces = torch.einsum('bii->b', rel.matrix()[:,:3,:3])
    #     inner = (traces-1)/2
    #     angle = torch.acos(torch.clamp((inner),-1,1))
    #     loss = self.loss_scale * weight * ((D_yn[:,:,:10] - ex_diff[:,:,:10]) ** 2) + 10 * angle.mean() # wegen preconditioning mehr stability?
    #     log_fun("angle",angle.mean(),on_step=True)
    # else:
    loss = self.loss_scale * weight * ((D_yn - ex_diff) ** 2) # wegen preconditioning mehr stability?

    # print(f"loss na: {loss.isnan().any()}")
    loss_xyz = loss[:,:,:3].mean()
    loss_rest = loss[:,:,3:].mean()
    if (Mode.warmup_xyz in self.mode and train_step < self.splatting_loss['warmup_steps']) or Mode.only_xyz in self.mode or Mode.fill_xyz in self.mode:
        mean_loss = loss_xyz
    else:
        mean_loss = loss_xyz + loss_rest
    # mean_loss = 5*loss_xyz + loss_rest

    if Mode.procrustes in self.mode:
        data = torch.cat([net.reparam.diffusion_to_data(D_yn[:,:,:10],context),D_yn[:,:,10:]],dim=-1)
    else:
        data = net.reparam.diffusion_to_data(D_yn, context)


    # regularization loss scalings zu eckig
    if Mode.normal in self.mode:
        scalings = data[:,:,6:9]
        scalings_activated = torch.exp(scalings)
        scalings_activated_norm_std_sum = torch.linalg.norm(scalings_activated,ord=2,dim=-1).std(dim=-1).sum()
        difference = scalings.max(dim=-1).values - scalings.min(dim=-1).values
        ratio = scalings.max(dim=-1).values / scalings.min(dim=-1).values

        # Sum up these differences
        scalings_range = difference.sum()

        scalings_norm = torch.linalg.norm(scalings, ord=2, dim=-1)
        scalings_norm_sum = scalings_norm.sum() * 50

        log_fun("scalings_dif_max_min",scalings_range,on_step=True)
        log_fun("scalings norm",scalings_norm_sum,on_step=True)
        log_fun("scalings loss",scalings_range.mean(),on_step=True)
        log_fun("rest_sum",loss[:,:,3:].sum(), on_step=True)
        log_fun("scalings_activated_norm_std_sum",scalings_activated_norm_std_sum, on_step=True)
    
    log_fun("diff_loss", mean_loss, on_step=True)
    log_fun("loss_xyz", loss_xyz, on_step=True)
    log_fun("loss_rest",loss_rest,on_step=True)

    # opactiy ist in der regel der letzte channel
    log_fun("opacity_loss", loss[:,:,-1].mean(), on_step=True)

    if Mode.rgb in self.mode:
        log_fun("mean_rgb_loss",loss[:,:,3:6].mean(),on_step=True)

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

    # alle werte eines batches werden mit dem zugehörigen c_in scalar multipliziert
    model_in = c_in * x
    F_x, cache = self.model(
        model_in, c_noise, raw_context, post_context, do_cache, cache
    )
    denoised = c_skip * x + c_out * F_x
    # print(f"denoised: {denoised}")
    # print(f"denoised na: {denoised.isnan().any()}")
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

def no_forward_preconditioning(
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
    sigma = sigma.reshape(-1, *ones(x.ndim - 1))
    
    F_x, cache = self.model(
        x, sigma, raw_context, post_context, do_cache, cache
    )
    denoised = F_x
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


def forward_preconditioning_not_rotation(
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
    x shape: xyz, color, scales, rotation, opacity
    """
    x_no_rotation = torch.cat([x[:,:,:9],x[:,:,13].unsqueeze(-1)],dim=-1)
    model_in_no_rotation = c_in * x_no_rotation
    model_in = torch.cat([model_in_no_rotation[:,:,:9],x[:,:,9:13], model_in_no_rotation[:,:,9].unsqueeze(-1)],dim=-1)
    F_x, cache = self.model(
        model_in, c_noise, raw_context, post_context, do_cache, cache
    )

    F_x_no_rotation = torch.cat([F_x[:,:,:9],F_x[:,:,13].unsqueeze(-1)],dim=-1)
    denoised_no_rotation = c_skip * x_no_rotation + c_out * F_x_no_rotation
    denoised = torch.cat([denoised_no_rotation[:,:,:9],F_x[:,:,9:13], denoised_no_rotation[:,:,9].unsqueeze(-1)],dim=-1)
    # print(f"denoised: {denoised}")
    # print(f"denoised na: {denoised.isnan().any()}")
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


def likelihood(
        model, 
        batch,
):
    kwargs = {
        'reverse_ode' : True,
        'gt_data' : batch.data,
        }
    reverse_sample = sample(model, batch.data.shape, batch.ctx, None, **kwargs)
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
def sample(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    # with torch.no_grad():
    #     example_image = torch.ones_like(context.image)
    #     example_tensor = torch.ones(shape).to(example_image.device)
    #     for c in range(example_image.shape[0]):
    #         context.image[c] = example_image[c]
    #     sigma_example = torch.ones((shape[0],1,1)).to(example_tensor.device)
    #     example_output = self(example_tensor, sigma_example, context, do_cache=False, cache=None)
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


    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        if Mode.procrustes in self.mode:
            latents = torch.cat([self.reparam.data_to_diffusion(gt_data[:,:,:10], context),gt_data[:,:,10:]],dim=-1)
        else:
            latents = self.reparam.data_to_diffusion(gt_data, context)
        x_next = latents.to(torch.float64)
    else:
        print("Sampling...")
        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]

    # if Mode.lie_rotations in self.mode:
    #     x_next[:,:,9:18] = batched_lietorch_tangential_to_rotation_matrix(x_next[:,:,9:12]).reshape(B,shape[1],9)
    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur # + gamma * t_cur
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        # if Mode.lie_rotations in self.mode:
        #     x_hat[:9,:,:] = x_cur[:9,:,:] + extra_noise[:9,:,:]
        #     noisy_rotation = batched_lietorch_tangential_to_rotation_matrix(extra_noise[:,:,9:12])
        #     resulting_rotations = torch.matmul(x_hat[9:18,:,:], noisy_rotation).view(shape[0],shape[1],9) 
        #     x_hat [9:18,:,:] = resulting_rotations
        # else:

        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_cur.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        d_cur = (x_cur - denoised) / t_hat # d_i in pseudo code
        x_next = x_cur + (t_next - t_hat) * d_cur # x_i+1 in pseudo code
        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = self( # D_theta in paper
                x_next.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next # d'_i in pseudo code
            x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    if reverse_ode:
        data = x_next
    else:
        if Mode.procrustes in self.mode:
            data = torch.cat([self.reparam.diffusion_to_data(x_next[:,:,:10], context),x_next[:,:,10:]],dim=-1)
        else:
            data = self.reparam.diffusion_to_data(x_next, context)
    return data

@torch.no_grad()
def sample_procrustes_so3(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    # with torch.no_grad():
    #     example_image = torch.ones_like(context.image)
    #     example_tensor = torch.ones(shape).to(example_image.device)
    #     for c in range(example_image.shape[0]):
    #         context.image[c] = example_image[c]
    #     sigma_example = torch.ones((shape[0],1,1)).to(example_tensor.device)
    #     example_output = self(example_tensor, sigma_example, context, do_cache=False, cache=None)
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


    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        if Mode.procrustes in self.mode:
            latents = torch.cat([self.reparam.data_to_diffusion(gt_data[:,:,:10], context),gt_data[:,:,10:]],dim=-1)
        else:
            latents = self.reparam.data_to_diffusion(gt_data, context)
        x_next = latents.to(torch.float64)
    else:
        print("Sampling...")
        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]

        r_next = x_next[:,:,10:]
        x_next = x_next[:,:,:10]

    # if Mode.lie_rotations in self.mode:
    #     x_next[:,:,9:18] = batched_lietorch_tangential_to_rotation_matrix(x_next[:,:,9:12]).reshape(B,shape[1],9)
    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_next
        r_cur = r_next
        t_hat = t_cur # + gamma * t_cur
        # Euler step.

        x_in = torch.cat([x_cur,r_cur],dim=-1)
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        x_denoised = denoised[:,:,:10]
        r_denoised = denoised[:,:,10:].reshape(-1,3,3)
        r_denoised_lietorch = lietorch.SO3(r_denoised, from_rotation_matrix=True)
        r_cur_lietorch = lietorch.SO3(r_cur.reshape(-1,3,3), from_rotation_matrix=True)
        relative_rotation = r_denoised_lietorch * r_cur_lietorch.inv()
        relative_rotation_log = relative_rotation.log() / t_cur
        small_step_rotvec = -(t_next - t_hat) * relative_rotation_log
        small_step_rotation = lietorch.SO3.exp(small_step_rotvec)
        r_next_lietorch = (small_step_rotation * r_cur_lietorch)
        r_next = r_next_lietorch.matrix()[:,:3,:3].reshape(B,-1,9)

        d_cur = (x_cur - x_denoised) / t_hat # d_i in pseudo code
        x_next = x_cur + (t_next - t_hat) * d_cur # x_i+1 in pseudo code


        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_in = torch.cat([x_next,r_next],dim=-1)
            denoised = self( # D_theta in paper
                x_in.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            x_denoised = denoised[:,:,:10]
            r_denoised = denoised[:,:,10:].reshape(-1,3,3)
            r_denoised_lietorch = lietorch.SO3(r_denoised, from_rotation_matrix=True)
            r_next_lietorch = lietorch.SO3(r_next.reshape(-1,3,3), from_rotation_matrix=True)
            relative_rotation = r_denoised_lietorch * r_next_lietorch.inv()
            relative_rotation_log_2 = relative_rotation.log() / t_next
            direction = relative_rotation_log_2 / 2 + relative_rotation_log / 2
            small_step_rotvec = -(t_next - t_hat) * direction
            small_step_rotation = lietorch.SO3.exp(small_step_rotvec)

            r_cur_lietorch = lietorch.SO3(r_cur.reshape(-1,3,3), from_rotation_matrix=True)
            r_next_lietorch = (small_step_rotation * r_cur_lietorch)
            # from gecco_torch.additional_metrics.metrics_so3 import rotational_distance_between_pairs_dot_product
            # print(rotational_distance_between_pairs_dot_product(r_denoised_lietorch.vec(),r_next_lietorch.vec()).mean())

            d_prime = (x_next - x_denoised) / t_next # d'_i in pseudo code
            x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    if reverse_ode:
        data = x_next
    else:
        if Mode.procrustes in self.mode:
            data = torch.cat([self.reparam.diffusion_to_data(x_next[:,:,:10], context),r_next],dim=-1)
        else:
            data = self.reparam.diffusion_to_data(x_next, context)
    return data



@torch.no_grad()
def sample_churn(
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


    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        if Mode.procrustes in self.mode:
            latents = torch.cat([self.reparam.data_to_diffusion(gt_data[:,:,:10], context),gt_data[:,:,10:]],dim=-1)
        else:
            latents = self.reparam.data_to_diffusion(gt_data, context)
        x_next = latents.to(torch.float64)
    else:
        print("Sampling...")
        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]

    # if Mode.lie_rotations in self.mode:
    #     x_next[:,:,9:18] = batched_lietorch_tangential_to_rotation_matrix(x_next[:,:,9:12]).reshape(B,shape[1],9)
    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, math.sqrt(2.0) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = t_cur + gamma * t_cur
        noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        x_hat = x_cur + extra_noise


        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_hat.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat # d_i in pseudo code
        x_next = x_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code
        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = self( # D_theta in paper
                x_next.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next # d'_i in pseudo code
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    if reverse_ode:
        data = x_next
    else:
        if Mode.procrustes in self.mode:
            data = torch.cat([self.reparam.diffusion_to_data(x_next[:,:,:10], context),x_next[:,:,10:]],dim=-1)
        else:
            data = self.reparam.diffusion_to_data(x_next, context)
    return data


@torch.no_grad()
def sample_archaic(
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


    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data, context)
        x_next = latents.to(torch.float64)
    else:
        print("Sampling...")
        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn(shape, device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]

    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_cur.to(dtype),
            t_cur.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        d_cur = (x_cur - denoised) / t_cur # d_i in pseudo code
        x_next = x_cur + (t_next - t_cur) * d_cur # x_i+1 in pseudo code
        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_next = x_next +  torch.randn_like(x_next) * t_next
    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    if reverse_ode:
        data = x_next
    else:
        data = self.reparam.diffusion_to_data(x_next, context)
    return data
