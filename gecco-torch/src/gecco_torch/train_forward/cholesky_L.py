import numpy as np
import torch
from gecco_torch.utils.build_cov_matrix_torch import build_covariance_from_activated_scaling_rotation, build_covariance_from_scaling_rotation_xyzw
from gecco_torch.utils.riemannian_helper_functions import L_to_cov_x6, find_cholesky_L, L_to_scale_rotation, group_operation_add, group_operation_inverse_L, exp_map_at_L, log_map_at_L, is_in_L_plus, geodesic_step
from gecco_torch.additional_metrics.metrics_so3 import geodesic_distance
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
import lietorch
from gecco_torch.structs import Mode
from tqdm import tqdm
import math
from gecco_torch.additional_metrics.metrics_so3 import best_fit_geodesic_distance, best_fit_euclidean_distance
from scipy.stats import multivariate_normal



FORCE_SMALL_SCALE = False

NOISE_SCALE_DIVISOR = 1

def find_noisy_L(scaling, rotation, sigma, batch_size, n):
    rotation_xyzw = rotation[:,[1,2,3,0]] # von wxyz zu xyzw
    noise_for_rotation = sigma.repeat_interleave(n)

    # Sampling from current temperature
    axis_angles = torch.vmap(lambda mu, s_single: IsotropicGaussianSO3(mu,
                                                                s_single,
                                                                force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                            randomness="different")(rotation_xyzw, noise_for_rotation)
    noisy_rotation_xyzw = (lietorch.SO3(rotation_xyzw) * lietorch.SO3.exp(axis_angles)).vec()

    noisy_rotation = noisy_rotation_xyzw[:,[3,0,1,2]] # von xyzw zu wxyz

    noise_for_scaling = torch.randn((batch_size,n,3)).cuda() * sigma / NOISE_SCALE_DIVISOR # durch 48 weil dann wird das noch aktiviert mit exp und wird zu groß
    # noise_for_scaling = torch.clip(noise_for_scaling, max = 7) # -> trotzdem noch bei 7 clippen
    noisy_scaling = scaling.reshape(batch_size,n , 3) + noise_for_scaling
    noisy_scaling = noisy_scaling.reshape(-1,3)

    with torch.autocast(device_type="cuda", enabled=False): # enabled False sagt kein autocast
        noisy_cov = build_covariance_from_activated_scaling_rotation(noisy_scaling, noisy_rotation)
        noisy_L = find_cholesky_L(noisy_cov)
    noisy_L = noisy_L.reshape(batch_size, n, 6)
    # print(torch.isinf(noisy_L).any())
    return noisy_L

def forward(self, net, examples, context, train_step, log_fun):
    ex_diff = net.reparam.data_to_diffusion(examples[:,:,:7], context) # alles bis auf scaling und rotation

    sigma = self.schedule(ex_diff)

    weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    """
    noise L
    """
    batch_size = examples.shape[0]
    n = examples.shape[1]

    scaling = examples[:,:,7:10].reshape(-1,3)
    rotation = examples[:,:,10:14].reshape(-1,4)

    # s2 = torch.tensor([-5,-5,-5],device=scaling.device,dtype=scaling.dtype).reshape(-1,3)   
    # rot2 = torch.tensor([0.9,0.5,0.04,0.1],device=scaling.device,dtype=scaling.dtype).reshape(-1,4)
    # # rot2 = rot2/torch.norm(rot2)
    # rot2 = torch.tensor([0.8694,0.4830,0.0386,0.0966],device=scaling.device,dtype=scaling.dtype).reshape(-1,4)
    # c =  build_covariance_from_scaling_rotation(s2,rot2)
    # l = find_cholesky_L(c)

    with torch.autocast(device_type="cuda", enabled=False):
        gt_cov = build_covariance_from_activated_scaling_rotation(scaling, rotation)
    gt_L = find_cholesky_L(gt_cov)

    noisy_L = find_noisy_L(scaling, rotation, sigma, batch_size, n)
    while torch.isinf(noisy_L).any():
        print("inf in noisy_L")
        noisy_L = find_noisy_L(scaling, rotation, sigma, batch_size, n)

    noise_for_rest = torch.randn_like(ex_diff) * sigma
    noised_data = ex_diff + noise_for_rest
    noisy_tensor = torch.concat([noised_data, noisy_L],dim=-1)

    D_yn = net(noisy_tensor, sigma, context) # input ist pointcloud, die mit noise verändert wurde, und das sigma

    # convert L back to covariance matrix using LLT
    denoised_L = D_yn[:,:,7:13]
    denoised_rest = D_yn[:,:,:7]

    denoised_for_distance = denoised_L.reshape(-1,6)
    mask_has_inf = ~torch.any(torch.isinf(denoised_for_distance), dim=1)
    denoised_for_distance = denoised_for_distance[mask_has_inf]
    gt_L_for_distance = gt_L[mask_has_inf]

    # check na
    if torch.isnan(D_yn).any():
        print("NA in D_yn")
        if not torch.isnan(denoised_rest).any():
            print("no NA in denoised_rest")
        if torch.isnan(denoised_L).any():
            print("NA in denoised_L")
            
    # check inf
    if torch.isinf(D_yn).any():
        print("inf in D_yn")
        if not torch.isinf(denoised_rest).any():
            print("no inf in denoised_rest")
        if torch.isinf(denoised_L).any():
            print("inf in denoised_L")
            if torch.isinf(denoised_for_distance).any():
                print("inf in denoised_for_distance")
            else:
                print(f"removed {(~mask_has_inf).sum()} inf values from denoised_for_distance")

    loss_L = geodesic_distance(denoised_for_distance, gt_L_for_distance).mean()
    # loss_Ls = geodesic_distance(denoised_for_distance, gt_L_for_distance)
    # mask_loss_ls = ~torch.isinf(loss_Ls)
    # print("Removed ", (~mask_loss_ls).sum(), " INF values from loss_Ls")
    # loss_L = loss_Ls[mask_loss_ls].mean()

    log_fun("geodesic_distance_loss", loss_L, on_step=True)

    # compare the orignal vs predicted
    gecco_loss = self.loss_scale * weight * ((denoised_rest - ex_diff) ** 2) # wegen preconditioning mehr stability?

    if Mode.rgb in self.mode:
        log_fun("mean_rgb_loss",gecco_loss[:,:,3:6].mean(),on_step=True)

    log_fun("mean_opacity_loss",gecco_loss[:,:,6].mean(),on_step=True)
    log_fun("mean_xyz_loss",gecco_loss[:,:,:3].mean(),on_step=True)


    data = net.reparam.diffusion_to_data(denoised_rest,context)
    data = torch.concat([data, denoised_L],dim=-1)

    if not torch.isnan(loss_L) and not torch.isinf(loss_L):
        mean_loss = gecco_loss.mean() + loss_L
    else:
        print(f"loss_L is NA: {torch.isnan(loss_L)}, inf: {torch.isinf(loss_L)}, skip, gecco loss NA: {torch.isnan(gecco_loss).any()}, inf: {torch.isinf(gecco_loss).any()}")
        mean_loss = gecco_loss.mean()

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
    model_in = torch.concat([c_in * x[:,:,:7],x[:,:,7:]],dim=-1)
    
    F_x, cache = self.model(
        model_in, c_noise, raw_context, post_context, do_cache, cache
    )
    """
    F_x output format xyz, rgb, scales, opacity, quat, scale
    """
    # output conditioning: c_skip * x + c_out * F_x
    skip_result = c_skip * x[:,:,:7]
    out_result = c_out * F_x[:,:,:7]
    denoised = skip_result + out_result
    denoised = torch.cat([denoised,F_x[:,:,7:]],dim=-1)

    """
    denoised output format xyz, rgb, opacity, L
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
    
    """
    Das netzwerk kriegt die Daten im Format [xyz, rgb, scales, opacity, rotations]
    """
    sigma = sigma.reshape(-1, *ones(x.ndim - 1))

    F_x, cache = self.model(
        x, sigma, raw_context, post_context, do_cache, cache
    )
    """
    F_x output format xyz, rgb, scales, opacity, quat, scale
    """
    denoised = F_x

    """
    denoised output format xyz, rgb, opacity, L
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


def get_noisy_L(gt_scaling, gt_rotation_wxyz, sigma, batch_size, n):
    rotation_xyzw = gt_rotation_wxyz[:,:,[1,2,3,0]].reshape(-1,4) # von wxyz zu xyzw
    noise_for_rotation = sigma.repeat(batch_size).repeat_interleave(n).type(rotation_xyzw.dtype)

    # Sampling from current temperature
    axis_angles = torch.vmap(lambda mu, s_single: IsotropicGaussianSO3(mu,
                                                                s_single,
                                                                force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                            randomness="different")(rotation_xyzw, noise_for_rotation)
    noisy_rotation_xyzw = (lietorch.SO3(rotation_xyzw) * lietorch.SO3.exp(axis_angles)).vec()

    noisy_rotation = noisy_rotation_xyzw[:,[3,0,1,2]] # von xyzw zu wxyz

    noise_for_scaling = torch.randn((batch_size,n,3)).cuda() * sigma / NOISE_SCALE_DIVISOR # durch 48 weil dann wird das noch aktiviert mit exp und wird zu groß
    # noise_for_scaling = torch.clip(noise_for_scaling, max = 7) # -> trotzdem noch bei 7 clippen
    noisy_scaling = gt_scaling.reshape(batch_size,n,3) + noise_for_scaling
    noisy_scaling = noisy_scaling.reshape(-1,3)


    with torch.autocast(device_type="cuda", enabled=False): # enabled False sagt kein autocast
        noisy_cov = build_covariance_from_activated_scaling_rotation(noisy_scaling, noisy_rotation)
        noisy_L = find_cholesky_L(noisy_cov)
    noisy_L = noisy_L.reshape(batch_size, n, 6)
    return noisy_L

@torch.no_grad()
def sample_standard_ode(
        self,
        shape,
        context,
        rng,
        **kwargs,
):
    print("Sampling...")
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

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


    # gt_rotation_wxyz = kwargs['gt_rotation_wxyz']
    # gt_scaling = kwargs['gt_scaling']
    # gt_cov = build_covariance_from_scaling_rotation(gt_scaling.reshape(-1,3), gt_rotation_wxyz.reshape(-1,4))
    # gt_L = find_cholesky_L(gt_cov)

    B = shape[0] # batch size
    N = shape[1] # number of points in pointcloud
    post_context = self.conditioner(context)

    # Time step discretization.
    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    # latents is in shape of desired output (batchsize,points in pointcloud, 3)
    latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # Main sampling loop.
    x_rest = latents.to(torch.float64) * t_steps[0]

    # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
    unit_rotation = torch.ones(B * N, 4).cuda()
    axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                t_steps[0],
                                                                force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                            randomness="different")(unit_rotation).to(unit_rotation.dtype)
    noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
    noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
    # noisy_scale = torch.clip(noisy_scale, max = 7)
    with torch.autocast(device_type="cuda", enabled=False):
        noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
        noisy_start_L = find_cholesky_L(noisy_cov)
    x_L = noisy_start_L.reshape(B, N, 6)
    # x_L = torch.randn_like(x_L) # * t_steps[0]

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")

    x_next = torch.cat([x_rest, x_L], dim=-1)
    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_next

        t_hat = t_cur 
        x_hat = x_cur


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
    data = self.reparam.diffusion_to_data(x_next[:,:,:7], context)
    data = torch.concat([data, x_next[:,:,7:]], dim=-1)
    return data

@torch.no_grad()
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

    # num_steps = 128
    # sigma_min = 0.002
    # sigma_max = 165
    # rho = 7
    # S_churn = 0.5
    # S_min = 0
    # S_max = float('inf')
    # S_noise = 1

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
    N = shape[1] # number of points in pointcloud
    post_context = self.conditioner(context)

    # Time step discretization.
    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    # latents is in shape of desired output (batchsize,points in pointcloud, 3)
    latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # Main sampling loop.
    x_rest = latents.to(torch.float64) * t_steps[0]

    # noisy_rotation = lietorch.SO3([],from_uniform_sampled = B * N).vec().to(latents.device)
    # noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
    # # noisy_scale = torch.clip(noisy_scale, max = 7)
    # noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    # noisy_start_L = find_cholesky_L(noisy_cov)
    # x_L = noisy_start_L.reshape(B, N, 6)
    gt_rotation_wxyz = kwargs['gt_rotation_wxyz']
    gt_scaling = kwargs['gt_scaling']
    gt_cov = build_covariance_from_activated_scaling_rotation(gt_scaling.reshape(-1,3), gt_rotation_wxyz.reshape(-1,4))
    gt_L = find_cholesky_L(gt_cov)

    noisy_start_L = get_noisy_L(gt_scaling, gt_rotation_wxyz, t_steps[0], B, N)
    x_L = noisy_start_L.clone()
    # best_fit_dist = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: Best fit distance zwischen noisy und gt: {best_fit_dist}") # 508915
    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        x_in = torch.cat([x_cur, x_L], dim=-1)

        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_cur.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        denoised_rest = denoised[:,:,:7]
        denoised_L = denoised[:,:,7:13]

        inverse_L_i = group_operation_inverse_L(x_L.reshape(-1,6))

        group_op_res = group_operation_add(inverse_L_i, denoised_L.reshape(-1,6))

        d_i = log_map_at_L(group_op_res, x_L.reshape(-1,6)) / t_cur

        x_L_next = geodesic_step(x_L.reshape(-1,6), d_i, -(t_next - t_cur)).reshape(B, N, 6)

        d_cur = (x_cur - denoised_rest) / t_cur # d_i in pseudo code
        x_rest_next = x_cur + (t_next - t_cur) * d_cur # x_i+1 in pseudo code

        print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L.reshape(-1,6))}")
        best_fit_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L[:4000])
        print(f"Best fit distance zwischen noisy und gt: {best_fit_dist}") # 508915

        # x_L_next = get_noisy_L(gt_scaling, gt_rotation_wxyz, t_next, B, N)

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     x_L_next = get_noisy_L(gt_scaling, gt_rotation_wxyz, t_next, B, N)
        #     x_in_2nd = torch.cat([x_rest_next, x_L_next], dim=-1)
        #     denoised = self( # D_theta in paper
        #         x_in_2nd.to(dtype),
        #         t_next.repeat(B).to(dtype),
        #         context,
        #         post_context,
        #     ).to(torch.float64)

        #     denoised_rest_2nd = denoised[:,:,:7]
        #     denoised_L_2nd = denoised[:,:,7:13]

        #     inverse_L_i_next = group_operation_inverse_L(x_L_next.reshape(-1,6))

        #     group_op_res_2nd = group_operation_add(inverse_L_i_next, denoised_L_2nd.reshape(-1,6))

        #     d_i_strich = log_map_at_L(group_op_res_2nd, x_L_next.reshape(-1,6)) / t_next

        #     d_i_strichs = d_i + d_i_strich

        #     x_L_next = geodesic_step(x_L.reshape(-1,6), d_i_strichs, -(t_next - t_cur)).reshape(B, N, 6)
        #     x_L_next = get_noisy_L(gt_scaling, gt_rotation_wxyz, t_next, B, N)
        #     # print(f"Distanz original zu Fortschritt mit 2nd Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        #     d_prime = (x_rest_next - denoised_rest_2nd) / t_next # d'_i in pseudo code
        #     x_rest_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_rest[:,:,:7], context)
    data = torch.concat([data, x_L], dim=-1)
    return data

@torch.no_grad()
def churn_sample(
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

    # num_steps = 128
    # sigma_min = 0.002
    # sigma_max = 165
    # rho = 7
    # S_churn = 0.5
    # S_min = 0
    # S_max = float('inf')
    # S_noise = 1

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
    N = shape[1] # number of points in pointcloud
    post_context = self.conditioner(context)

    # Time step discretization.
    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    # latents is in shape of desired output (batchsize,points in pointcloud, 3)
    latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

    # Main sampling loop.
    x_rest = latents.to(torch.float64) * t_steps[0]

    noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
    noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
    # noisy_scale = torch.clip(noisy_scale, max = 7)
    noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
    noisy_start_L = find_cholesky_L(noisy_cov)
    x_L = noisy_start_L.reshape(B, N, 6)

    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, math.sqrt(2.0) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = t_cur + gamma * t_cur
        noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        x_rest_hat = x_cur + extra_noise

        x_in = torch.cat([x_rest_hat, x_L], dim=-1)

        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        denoised_rest = denoised[:,:,:7]
        denoised_L = denoised[:,:,7:13]

        inverse_L_i = group_operation_inverse_L(x_L.reshape(-1,6))

        group_op_res = group_operation_add(inverse_L_i, denoised_L.reshape(-1,6))

        d_i = log_map_at_L(group_op_res, x_L.reshape(-1,6)) / t_cur

        x_L_next = geodesic_step(x_L.reshape(-1,6), d_i, -(t_next - t_cur)).reshape(B, N, 6)

        d_cur = (x_rest_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_rest_next = x_rest_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_in_2nd = torch.cat([x_rest_next, x_L_next], dim=-1)
            denoised = self( # D_theta in paper
                x_in_2nd.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            denoised_rest_2nd = denoised[:,:,:7]
            denoised_L_2nd = denoised[:,:,7:13]

            inverse_L_i_next = group_operation_inverse_L(x_L_next.reshape(-1,6))

            group_op_res_2nd = group_operation_add(inverse_L_i_next, denoised_L_2nd.reshape(-1,6))

            d_i_strich = log_map_at_L(group_op_res_2nd, x_L_next.reshape(-1,6)) / t_next

            d_i_strichs = d_i + d_i_strich

            x_L_next = geodesic_step(x_L.reshape(-1,6), d_i_strichs, -(t_next - t_cur)).reshape(B, N, 6)
            # print(f"Distanz original zu Fortschritt mit 2nd Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
            d_prime = (x_rest_next - denoised_rest_2nd) / t_next # d'_i in pseudo code
            x_rest_next = x_rest_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_rest[:,:,:7], context)
    data = torch.concat([data, x_L], dim=-1)
    return data


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

def likelihood(
        model, 
        batch,
):
    kwargs = {
        'reverse_ode' : True,
        'gt_data' : batch.data,
        }
    
    reverse_sample = sample_logdirection_tangent(model, batch.data.shape, batch.ctx, None, **kwargs)
    reverse_sample_rest = reverse_sample[:,:,:7]

    # retrieve scale from cov = LLT 
    scaling_decomposition, rotations_decomposition = L_to_scale_rotation(reverse_sample[:,:,7:].reshape(-1,6))


    mu = np.zeros(10)  # Replace with the 13-dimensional mean vector
    sigma = np.diag(165*np.ones(10))  # Replace with the 13x13 covariance matrix

    # Initialize the multivariate normal distribution
    mvn = multivariate_normal(mean=mu, cov=sigma)
    prob_densities = mvn.pdf(torch.cat([reverse_sample_rest.reshape(-1,7),scaling_decomposition.reshape(-1,3)],dim=-1).detach().cpu().numpy())

    # Calculate the joint likelihood
    sum_likelihood = np.sum(-np.log(prob_densities+1e-10))
    averaged = sum_likelihood / (reverse_sample_rest.shape[0] * reverse_sample_rest.shape[1])
    return averaged


@torch.no_grad()
def sample_logdirection_tangent_doesntwork(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    print("Sampling logdirection ...")
    
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

    N = shape[1] # number of points in pointcloud

    if not debug:
        post_context = self.conditioner(context)
        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
    else:
        t_steps = kwargs['t_steps']
        num_steps = t_steps.shape[0] - 1


    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
        x_next = latents.to(torch.float64)
        cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
        x_L_next = find_cholesky_L(cov).reshape(B, N, 6)

        x_next_rest = x_next
    else:
        print("Sampling...")


        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1
        # Main sampling loop.
        x_next_rest = latents.to(torch.float64) * t_steps[0]

        # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
        unit_rotation = torch.ones(B * N, 4).cuda()
        axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                    t_steps[0],
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation).to(unit_rotation.dtype)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
        # import numpy as np
        # np_file = np.load("/home/giese/Documents/gecco/latents.npz")
        # latents = torch.from_numpy(np_file['latents']).cuda()
        # noisy_rotation = torch.from_numpy(np_file['noisy_rotation']).cuda()
        # noisy_scale = torch.from_numpy(np_file['noisy_scale']).cuda()
        # noisy_scale = torch.clip(noisy_scale, max = 7)
        with torch.autocast(device_type="cuda", enabled=False):
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
            noisy_start_L = find_cholesky_L(noisy_cov)
        x_L_next = noisy_start_L.reshape(B, N, 6)

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")

    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        x_cur_rest = x_next_rest # + extra_noise

        # replace possible infs
        x_L = x_L_next.to(dtype)
        x_L[torch.isinf(x_L)] = 1e6
        # zwischenstand.append(torch.cat([x_next_rest, x_L], dim=-1))
        x_in = torch.cat([x_cur_rest, x_L], dim=-1)

        if not debug:
            # Euler step.
            denoised = self( # D_theta in pseudo code in paper
                x_in.to(dtype),
                t_cur.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            denoised_rest = denoised[:,:,:7]
            denoised_L = denoised[:,:,7:13] 

            # replace inf with really high values
            denoised_L[torch.isinf(denoised_L)] = 1e5
        else:
            denoised_rest = kwargs['denoised_rest']
            denoised_L = kwargs['denoised_L']

        # predicted_dist = best_fit_geodesic_distance(denoised_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} denoised best fit distance: {predicted_dist}")
        step_size = t_next - t_cur

        direction = log_map_at_L(K = denoised_L.reshape(-1,6),L = x_L.reshape(-1,6)) / t_cur
        x_L_next = exp_map_at_L(X = -step_size * direction, L = x_L.reshape(-1,6)).reshape(B, N, 6)

        # step_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} step best fit distance: {step_dist}")

        d_cur = (x_cur_rest - denoised_rest) / t_cur # d_i in pseudo code
        x_next_rest = x_cur_rest + step_size * d_cur # x_i+1 in pseudo code

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        if i < num_steps - 1:
            # replace infs again
            x_L_next = x_L_next.to(dtype)
            x_L_next[torch.isinf(x_L_next)] = 1e5
            x_in_2nd = torch.cat([x_next_rest, x_L_next], dim=-1)
            if not debug:
                denoised = self( # D_theta in paper
                    x_in_2nd.to(dtype),
                    t_next.repeat(B).to(dtype),
                    context,
                    post_context,
                ).to(torch.float64)

                denoised_rest_2nd = denoised[:,:,:7]
                denoised_L_2nd = denoised[:,:,7:13]
                # replace inf with really high values
                denoised_L_2nd[torch.isinf(denoised_L_2nd)] = 1e5
            else:
                denoised_rest_2nd = kwargs['denoised_rest']
                denoised_L_2nd = kwargs['denoised_L']

            direction_2 = log_map_at_L(K = denoised_L_2nd.reshape(-1,6), L = x_L_next.reshape(-1,6)) / t_next
            direction_combined = -step_size * (direction / 2 + direction_2 / 2)
            x_L_next = exp_map_at_L(X = direction_combined, L = x_L.reshape(-1,6)).reshape(B, N, 6)
            x_L_next[torch.isinf(x_L_next)] = 1e5

            # step_2_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
            # print(f"Iteration {i} step 2 best fit distance: {step_2_dist}")

            d_prime = (x_next_rest - denoised_rest_2nd) / t_next # d'_i in pseudo code
            x_next_rest = x_cur_rest + step_size * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

    if with_pbar:
        ts.close()

    if reverse_ode:
        data = torch.cat([x_next_rest, x_L], dim=-1)
    else:
        if not debug:
            # we were in diffusion space previously, so naturally we have to go back after sampling
            data = self.reparam.diffusion_to_data(x_next_rest, context)
            data = torch.cat([data, x_L], dim=-1)
            # print(best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000]))
        else: 
            data = torch.cat([x_next_rest, x_L], dim=-1)
    return data


@torch.no_grad()
def sample_logdirection_test(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    print("Sampling logdirection ...")
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

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


    # gt_rotation_wxyz = kwargs['gt_rotation_wxyz']
    # gt_scaling = kwargs['gt_scaling']
    # gt_cov = build_covariance_from_scaling_rotation(gt_scaling.reshape(-1,3), gt_rotation_wxyz.reshape(-1,4))
    # gt_L = find_cholesky_L(gt_cov)
    B = shape[0] # batch size
    N = shape[1] # number of points in pointcloud
    post_context = self.conditioner(context)

    # Time step discretization.
    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
        x_next = latents.to(torch.float64)
        cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
        x_L = find_cholesky_L(cov).reshape(B, N, 6)

        x_rest = x_next
    else:
        print("Sampling...")


        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_rest = latents.to(torch.float64) * t_steps[0]

        # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
        unit_rotation = torch.ones(B * N, 4).cuda()
        axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                    t_steps[0],
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation).to(unit_rotation.dtype)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
        # noisy_scale = torch.clip(noisy_scale, max = 7)
        with torch.autocast(device_type="cuda", enabled=False):
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
            noisy_start_L = find_cholesky_L(noisy_cov)
        x_L = noisy_start_L.reshape(B, N, 6)
    # x_L = torch.randn_like(x_L) # * t_steps[0]

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")


    gt_data = kwargs["gt_data"]
    gt_data_diffusion_space = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
    gt_cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
    gt_x_L = find_cholesky_L(gt_cov).reshape(B, N, 6)

    # start_shortest_distance_rest = best_fit_euclidean_distance(x_rest[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
    # start_shortest_distance_L = best_fit_geodesic_distance(x_L[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur #+ gamma * t_cur
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        x_rest_hat = x_cur # + extra_noise

        # replace possible infs
        x_L = x_L.to(dtype)
        x_L[torch.isinf(x_L)] = 1e6
        x_in = torch.cat([x_rest_hat, x_L], dim=-1)


        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        denoised_rest = denoised[:,:,:7]
        denoised_L = denoised[:,:,7:13] 

        # shortest_distance_rest = best_fit_euclidean_distance(denoised_rest[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
        # shortest_distance_L = best_fit_geodesic_distance(denoised_L[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
        # print(f"denoised {i} rest distance {shortest_distance_rest}, entspricht sigma {shortest_distance_rest / 10100}")
        # print(f"denoised {i} L distance {shortest_distance_L}, entspricht sigma {shortest_distance_L / 3110} ")

        # replace inf with really high values
        denoised_L[torch.isinf(denoised_L)] = 1e5

        direction = log_map_at_L(denoised_L.reshape(-1,6), x_L.reshape(-1,6)) / t_cur
        x_L_next = exp_map_at_L(-(t_next - t_cur) * direction, x_L.reshape(-1,6)).reshape(B, N, 6)

        # step_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} step best fit distance: {step_dist}")

        d_cur = (x_rest_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_rest_next = x_rest_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # shortest_distance_rest_step = best_fit_euclidean_distance(x_rest_next[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
        # shortest_distance_L_step = best_fit_geodesic_distance(x_L_next[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
        # print(f"step {i} rest distance {shortest_distance_rest_step}, entspricht sigma {shortest_distance_rest_step / 10100}")
        # print(f"step {i} L distance {shortest_distance_L_step}, entspricht sigma {shortest_distance_L_step / 3110}")

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        if i < num_steps - 1:
            # replace infs again
            x_L_next = x_L_next.to(dtype)
            x_L_next[torch.isinf(x_L_next)] = 1e5
            x_in_2nd = torch.cat([x_rest_next, x_L_next], dim=-1)
            denoised = self( # D_theta in paper
                x_in_2nd.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            denoised_rest_2nd = denoised[:,:,:7]
            denoised_L_2nd = denoised[:,:,7:13]
            # replace inf with really high values
            denoised_L_2nd[torch.isinf(denoised_L_2nd)] = 1e5

            direction_2 = log_map_at_L(denoised_L_2nd.reshape(-1,6), x_L_next.reshape(-1,6)) / t_next
            x_L_next = exp_map_at_L(-(t_next - t_cur) * (direction + direction_2) / 2, x_L.reshape(-1,6)).reshape(B, N, 6)
            x_L_next[torch.isinf(x_L_next)] = 1e5

            d_prime = (x_rest_next - denoised_rest_2nd) / t_next # d'_i in pseudo code
            x_rest_next = x_rest_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

            # shortest_distance_rest_step_2 = best_fit_euclidean_distance(x_rest_next[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
            # shortest_distance_L_step_2 = best_fit_geodesic_distance(x_L_next[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))

            # print(f"step 2nd {i} rest distance {shortest_distance_rest_step_2}, entspricht sigma {shortest_distance_rest_step_2 / 10100}")
            # print(f"step 2nd {i} L distance {shortest_distance_L_step_2}, entspricht sigma {shortest_distance_L_step_2 / 3110} ")

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    if reverse_ode:
        data = torch.cat([x_rest, x_L], dim=-1)
    else:
        # we were in diffusion space previously, so naturally we have to go back after sampling
        data = self.reparam.diffusion_to_data(x_rest[:,:,:7], context)
        data = torch.cat([data, x_L], dim=-1)
        # print(best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000]))
    return data

@torch.no_grad()
def sample_ddpm(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    print("Sampling ddpm ...")
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

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

    t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
    B = shape[0] # batch size
    N = shape[1] # number of points in pointcloud
    post_context = self.conditioner(context)

    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
        x_next = latents.to(torch.float64)
        x_L = gt_data[:,:,7:13]
        x_rest = x_next
    else:
        print("Sampling...")


        # Time step discretization.

        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_rest = latents.to(torch.float64) * t_steps[0]

        # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
        unit_rotation = torch.ones(B * N, 4).cuda()
        axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                    t_steps[0],
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation).to(unit_rotation.dtype)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
        # noisy_scale = torch.clip(noisy_scale, max = 7)
        with torch.autocast(device_type="cuda", enabled=False):
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
            noisy_start_L = find_cholesky_L(noisy_cov)
        x_L = noisy_start_L.reshape(B, N, 6)
    # x_L = torch.randn_like(x_L) # * t_steps[0]

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")

    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur #+ gamma * t_cur
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        x_rest_hat = x_cur # + extra_noise

        # replace possible infs
        x_L = x_L.to(dtype)
        x_L[torch.isinf(x_L)] = 1e6
        x_in = torch.cat([x_rest_hat, x_L], dim=-1)


        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        denoised_rest = denoised[:,:,:7]
        denoised_L = denoised[:,:,7:13] 

        # replace inf with really high values
        denoised_L[torch.isinf(denoised_L)] = 1e10

        # predicted_dist = best_fit_geodesic_distance(denoised_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} denoised best fit distance: {predicted_dist}")

        direction = log_map_at_L(denoised_L.reshape(-1,6), x_L.reshape(-1,6)) / t_cur
        x_L_next = exp_map_at_L(-(t_next - t_cur) * direction, x_L.reshape(-1,6)).reshape(B, N, 6)

        # step_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} step best fit distance: {step_dist}")

        d_cur = (x_rest_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_rest_next = x_rest_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     # replace infs again
        #     x_L_next = x_L_next.to(dtype)
        #     x_L_next[torch.isinf(x_L_next)] = 1e6   
        #     scaling_decomp, rotations_decomp = L_to_scale_rotation(x_L_next.reshape(-1,6))
        #     noisy_rotation = get_noisy_rotation(B * N, t_next)
        #     noised_rotation = (lietorch.SO3(rotations_decomp) * noisy_rotation.inv()).vec()

        #     noisy_scale = scaling_decomp + torch.randn(B * N, 3).cuda() * t_next
        #     x_rest_next = x_rest_next + torch.randn(B, N, 7).cuda() * t_next
        #     noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noised_rotation)
        #     x_L_next = find_cholesky_L(noisy_cov).reshape(B, N, 6)

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    if reverse_ode:
        data = torch.concat(x_rest, x_L, dim=-1)
    else:
        # we were in diffusion space previously, so naturally we have to go back after sampling
        data = self.reparam.diffusion_to_data(x_rest[:,:,:7], context)
        data = torch.concat([data, x_L], dim=-1)
        # print(best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000]))
    return data


@torch.no_grad()
def sample_ddpm_rot_step(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    print("Sampling ddpm ...")
    
    kwargs = {**self.sampler_kwargs, **kwargs}
    num_steps = kwargs["num_steps"]
    sigma_min = kwargs["sigma_min"]
    sigma_max = kwargs["sigma_max"]

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

    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data, context)
        x_next = latents.to(torch.float64)
        x_L = gt_data[:,:,7:13]
        x_rest = x_next[:,:,:7]
    else:
        print("Sampling...")

        B = shape[0] # batch size
        N = shape[1] # number of points in pointcloud
        post_context = self.conditioner(context)

        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)

        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_rest = latents.to(torch.float64) * t_steps[0]

        # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
        unit_rotation = torch.ones(B * N, 4).cuda()
        axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                    t_steps[0],
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation).to(unit_rotation.dtype)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
        # noisy_scale = torch.clip(noisy_scale, max = 7)
        with torch.autocast(device_type="cuda", enabled=False):
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
            noisy_start_L = find_cholesky_L(noisy_cov)
        x_L = noisy_start_L.reshape(B, N, 6)
    # x_L = torch.randn_like(x_L) # * t_steps[0]

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")

    
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur #+ gamma * t_cur
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        x_rest_hat = x_cur # + extra_noise

        # replace possible infs
        x_L = x_L.to(dtype)
        x_L[torch.isinf(x_L)] = 1e6
        x_in = torch.cat([x_rest_hat, x_L], dim=-1)


        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_in.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        denoised_rest = denoised[:,:,:7]
        denoised_L = denoised[:,:,7:13] 

        # replace inf with really high values
        denoised_L[torch.isinf(denoised_L)] = 1e10
        scaling_decomp, rotations_decomp = L_to_scale_rotation(x_L_next.reshape(-1,6))
        scaling_decomp = scaling_decomp.reshape(B, N, 3)
        rotations_decomp = rotations_decomp.reshape(B, N, 4)

        # predicted_dist = best_fit_geodesic_distance(denoised_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} denoised best fit distance: {predicted_dist}")

        direction = log_map_at_L(denoised_L.reshape(-1,6), x_L.reshape(-1,6)) / t_cur
        x_L_next = exp_map_at_L(-(t_next - t_cur) * direction, x_L.reshape(-1,6)).reshape(B, N, 6)

        # step_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} step best fit distance: {step_dist}")

        d_cur = (x_rest_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_rest_next = x_rest_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        if i < num_steps - 1:
            # replace infs again
            x_L_next = x_L_next.to(dtype)
            x_L_next[torch.isinf(x_L_next)] = 1e6   
            scaling_decomp, rotations_decomp = L_to_scale_rotation(x_L_next.reshape(-1,6))
            noisy_rotation = get_noisy_rotation(B * N, t_next)
            noised_rotation = (lietorch.SO3(rotations_decomp) * noisy_rotation.inv()).vec()

            noisy_scale = scaling_decomp + torch.randn(B * N, 3).cuda() * t_next
            x_rest_next = x_rest_next + torch.randn(B, N, 7).cuda() * t_next
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noised_rotation)
            x_L_next = find_cholesky_L(noisy_cov).reshape(B, N, 6)

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    if reverse_ode:
        data = torch.concat(x_rest, x_L, dim=-1)
    else:
        # we were in diffusion space previously, so naturally we have to go back after sampling
        data = self.reparam.diffusion_to_data(x_rest[:,:,:7], context)
        data = torch.concat([data, x_L], dim=-1)
        # print(best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000]))
    return data



@torch.no_grad()
def sample_logdirection_tangent(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    print("Sampling logdirection ...")
    
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


    # gt_rotation_wxyz = kwargs['gt_rotation_wxyz']
    # gt_scaling = kwargs['gt_scaling']
    # gt_cov = build_covariance_from_scaling_rotation(gt_scaling.reshape(-1,3), gt_rotation_wxyz.reshape(-1,4))
    # gt_L = find_cholesky_L(gt_cov)
    B = shape[0] # batch size
    N = shape[1] # number of points in pointcloud
    if not debug:
        post_context = self.conditioner(context)
        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
    else:
        t_steps = kwargs['t_steps']
        num_steps = t_steps.shape[0] - 1

    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
        x_next = latents.to(torch.float64)
        cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
        x_L = find_cholesky_L(cov).reshape(B, N, 6)

        x_rest = x_next
    else:
        print("Sampling...")


        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_rest = latents.to(torch.float64) * t_steps[0]

        # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
        unit_rotation = torch.ones(B * N, 4).cuda()
        axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                    t_steps[0],
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation).to(unit_rotation.dtype)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
        # import numpy as np
        # np.savez("latents.npz", latents=latents.cpu().numpy(), noisy_rotation=noisy_rotation.cpu().numpy(), noisy_scale=noisy_scale.cpu().numpy())
        # noisy_scale = torch.clip(noisy_scale, max = 7)
        with torch.autocast(device_type="cuda", enabled=False):
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
            noisy_start_L = find_cholesky_L(noisy_cov)
        x_L = noisy_start_L.reshape(B, N, 6)
    # x_L = torch.randn_like(x_L) # * t_steps[0]

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")


    # gt_data = kwargs["gt_data"]
    # gt_data_diffusion_space = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
    # gt_cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
    # gt_x_L = find_cholesky_L(gt_cov).reshape(B, N, 6)

    # start_shortest_distance_rest = best_fit_euclidean_distance(x_rest[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
    # start_shortest_distance_L = best_fit_geodesic_distance(x_L[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur #+ gamma * t_cur
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        x_rest_hat = x_cur # + extra_noise

        # replace possible infs
        x_L = x_L.to(dtype)
        x_L[torch.isinf(x_L)] = 1e6
        x_in = torch.cat([x_rest_hat, x_L], dim=-1)


        if not debug:
            # Euler step.
            denoised = self( # D_theta in pseudo code in paper
                x_in.to(dtype),
                t_cur.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            denoised_rest = denoised[:,:,:7]
            denoised_L = denoised[:,:,7:13] 

            # replace inf with really high values
            denoised_L[torch.isinf(denoised_L)] = 1e5
        else:
            denoised_rest = kwargs['denoised_rest']
            denoised_L = kwargs['denoised_L']

        # shortest_distance_rest = best_fit_euclidean_distance(denoised_rest[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
        # shortest_distance_L = best_fit_geodesic_distance(denoised_L[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
        # print(f"denoised {i} rest distance {shortest_distance_rest}, entspricht sigma {shortest_distance_rest / 10100}")
        # print(f"denoised {i} L distance {shortest_distance_L}, entspricht sigma {shortest_distance_L / 3110} ")E
        # log_map_at_L(K,L)
        # exp_map_at_L(X,L)
        direction = log_map_at_L(denoised_L.reshape(-1,6), x_L.reshape(-1,6)) / t_cur
        x_L_next = exp_map_at_L(-(t_next - t_cur) * direction, x_L.reshape(-1,6)).reshape(B, N, 6)

        # step_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} step best fit distance: {step_dist}")

        d_cur = (x_rest_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_rest_next = x_rest_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # shortest_distance_rest_step = best_fit_euclidean_distance(x_rest_next[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
        # shortest_distance_L_step = best_fit_geodesic_distance(x_L_next[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
        # print(f"step {i} rest distance {shortest_distance_rest_step}, entspricht sigma {shortest_distance_rest_step / 10100}")
        # print(f"step {i} L distance {shortest_distance_L_step}, entspricht sigma {shortest_distance_L_step / 3110}")

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        if i < num_steps - 1:
            # replace infs again
            x_L_next = x_L_next.to(dtype)
            x_L_next[torch.isinf(x_L_next)] = 1e5
            x_in_2nd = torch.cat([x_rest_next, x_L_next], dim=-1)
            if not debug:
                denoised = self( # D_theta in paper
                    x_in_2nd.to(dtype),
                    t_next.repeat(B).to(dtype),
                    context,
                    post_context,
                ).to(torch.float64)

                denoised_rest_2nd = denoised[:,:,:7]
                denoised_L_2nd = denoised[:,:,7:13]
                # replace inf with really high values
                denoised_L_2nd[torch.isinf(denoised_L_2nd)] = 1e5
            else:
                denoised_rest_2nd = kwargs['denoised_rest']
                denoised_L_2nd = kwargs['denoised_L']

            direction_2 = log_map_at_L(denoised_L_2nd.reshape(-1,6), x_L_next.reshape(-1,6)) / t_next
            x_L_next = exp_map_at_L(-(t_next - t_cur) * (direction + direction_2) / 2, x_L.reshape(-1,6)).reshape(B, N, 6)
            x_L_next[torch.isinf(x_L_next)] = 1e5

            d_prime = (x_rest_next - denoised_rest_2nd) / t_next # d'_i in pseudo code
            x_rest_next = x_rest_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

            # shortest_distance_rest_step_2 = best_fit_euclidean_distance(x_rest_next[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
            # shortest_distance_L_step_2 = best_fit_geodesic_distance(x_L_next[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))

            # print(f"step 2nd {i} rest distance {shortest_distance_rest_step_2}, entspricht sigma {shortest_distance_rest_step_2 / 10100}")
            # print(f"step 2nd {i} L distance {shortest_distance_L_step_2}, entspricht sigma {shortest_distance_L_step_2 / 3110} ")

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    if reverse_ode:
        data = torch.cat([x_rest, x_L], dim=-1)
    else:
        # we were in diffusion space previously, so naturally we have to go back after sampling
        if not debug:
            # we were in diffusion space previously, so naturally we have to go back after sampling
            data = self.reparam.diffusion_to_data(x_rest, context)
            data = torch.cat([data, x_L], dim=-1)
            # print(best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000]))
        else: 
            data = torch.cat([x_rest, x_L], dim=-1)
    return data

if __name__ == "__main__":
    from gecco_torch.scene.gaussian_model import GaussianModel
    from gecco_torch.diffusionsplat import LogUniformSchedule
    from gecco_torch.utils.render_gaussian_circle import render_gaussian
    from gecco_torch.utils.riemannian_helper_functions import L_to_cov_x6

    gaussian = GaussianModel(3)
    gaussian.load_ply(("/globalwork/giese/gaussians/02958343/8c6c271a149d8b68949b12cf3977a48b/point_cloud/iteration_" + str(10000) +"/point_cloud.ply"))
    schedule = LogUniformSchedule(165)
    t_steps = torch.flip(schedule.return_schedule(128),[0])
    xyz = gaussian._xyz
    color = gaussian._features_dc.squeeze(1)
    opacity = gaussian._opacity
    scale = gaussian._scaling
    rotation = gaussian._rotation
    with torch.autocast(device_type="cuda", enabled=False):
        gt_cov = build_covariance_from_activated_scaling_rotation(torch.exp(scale), rotation)
    gt_L = find_cholesky_L(gt_cov)
    data = torch.cat([xyz, color, opacity],dim=-1)
    kwargs = {
        'debug' : True,
        'denoised_rest' : data.unsqueeze(0),
        'denoised_L' : gt_L.unsqueeze(0),
        't_steps' : t_steps,
    }
    gaussian_data = sample_logdirection_tangent(None, (1, xyz.shape[0], data.shape[1] + gt_L.shape[1]), None, None, **kwargs)
    # gaussian_data = gaussian_datas[50]
    def insert_gaussian(gaussian, xyz, color, opacity):
        gaussian._xyz = xyz
        gaussian._features_dc = color
        gaussian._opacity = opacity
        gaussian._rotation = None
        gaussian._scaling = None
        return gaussian
    gaussian_data = gaussian_data.type(torch.float32)
    gaussian = insert_gaussian(gaussian,
                               gaussian_data[0,:,:3].squeeze(0),
                               gaussian_data[0,:,3:6].reshape(gaussian_data.shape[1],1,3),
                               gaussian_data[0,:,6].squeeze(0).unsqueeze(-1)
                              )
    cov = L_to_cov_x6(gaussian_data[0,:,7:].reshape(gaussian_data.shape[1],6))
    kwargs = {
        'use_cov' : True,
        'cov3D_precomp' : cov,
    }
    render_gaussian(gaussian, **kwargs)
    pass

@torch.no_grad()
def sample_logdirection_tangent_worked(
        self,
        shape,
        context,
        rng,
        **kwargs,
    ):
    print("Sampling logdirection ...")
    
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


    # gt_rotation_wxyz = kwargs['gt_rotation_wxyz']
    # gt_scaling = kwargs['gt_scaling']
    # gt_cov = build_covariance_from_scaling_rotation(gt_scaling.reshape(-1,3), gt_rotation_wxyz.reshape(-1,4))
    # gt_L = find_cholesky_L(gt_cov)
    B = shape[0] # batch size
    N = shape[1] # number of points in pointcloud
    if not debug:
        post_context = self.conditioner(context)
        # Time step discretization.
        t_steps = self.t_steps(num_steps, sigma_max, sigma_min, rho)
    else:
        t_steps = kwargs['t_steps']
        num_steps = t_steps.shape[0] - 1

    reverse_ode = kwargs.get("reverse_ode", False)
    if reverse_ode:
        print("Reverse sampling...")
        t_steps = t_steps.flip(0)[1:] # keine 0
    # if Mode.lie_rotations in self.mode:
    #     shape = (B, shape[1], shape[2] - 6)
        gt_data = kwargs["gt_data"]
        latents = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
        x_next = latents.to(torch.float64)
        cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
        x_L = find_cholesky_L(cov).reshape(B, N, 6)

        x_rest = x_next
    else:
        print("Sampling...")


        # latents is in shape of desired output (batchsize,points in pointcloud, 3)
        latents = torch.randn((shape[0], shape[1], 7), device=device, generator=rng, dtype=dtype) # gaussian noise, sample from gaussian mean 0 std 1

        # Main sampling loop.
        x_rest = latents.to(torch.float64) * t_steps[0]

        # noisy_rotation = lietorch.SO3([],from_uniform_sampled= B * N).vec().to(latents.device)
        unit_rotation = torch.ones(B * N, 4).cuda()
        axis_angles = torch.vmap(lambda mu: IsotropicGaussianSO3(mu,
                                                                    t_steps[0],
                                                                    force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                                randomness="different")(unit_rotation).to(unit_rotation.dtype)
        noisy_rotation = (lietorch.SO3(unit_rotation) * lietorch.SO3.exp(axis_angles)).vec()
        noisy_scale = torch.randn(B * N, 3).cuda() * t_steps[0] / NOISE_SCALE_DIVISOR
        # import numpy as np
        # np.savez("latents.npz", latents=latents.cpu().numpy(), noisy_rotation=noisy_rotation.cpu().numpy(), noisy_scale=noisy_scale.cpu().numpy())
        # noisy_scale = torch.clip(noisy_scale, max = 7)
        with torch.autocast(device_type="cuda", enabled=False):
            noisy_cov = build_covariance_from_scaling_rotation_xyzw(noisy_scale, noisy_rotation)
            noisy_start_L = find_cholesky_L(noisy_cov)
        x_L = noisy_start_L.reshape(B, N, 6)
    # x_L = torch.randn_like(x_L) # * t_steps[0]

    # distance_start_noise = best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L[:4000])
    # print(f"Am Anfang: best fit distance {distance_start_noise}")


    # gt_data = kwargs["gt_data"]
    # gt_data_diffusion_space = self.reparam.data_to_diffusion(gt_data[:,:,:7], context)
    # gt_cov = build_covariance_from_activated_scaling_rotation(gt_data[:,:,7:10].reshape(-1,3), gt_data[:,:,10:14].reshape(-1,4))
    # gt_x_L = find_cholesky_L(gt_cov).reshape(B, N, 6)

    # start_shortest_distance_rest = best_fit_euclidean_distance(x_rest[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
    # start_shortest_distance_L = best_fit_geodesic_distance(x_L[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        x_cur = x_rest

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, math.sqrt(2.0) - 1)
        #     if S_min <= t_cur <= S_max
        #     else 0
        # )
        t_hat = t_cur #+ gamma * t_cur
        # noise = torch.randn(x_cur.shape, device=device, generator=rng, dtype=dtype) * S_noise
        # extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise

        x_rest_hat = x_cur # + extra_noise

        # replace possible infs
        x_L = x_L.to(dtype)
        x_L[torch.isinf(x_L)] = 1e6
        x_in = torch.cat([x_rest_hat, x_L], dim=-1)


        if not debug:
            # Euler step.
            denoised = self( # D_theta in pseudo code in paper
                x_in.to(dtype),
                t_cur.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)

            denoised_rest = denoised[:,:,:7]
            denoised_L = denoised[:,:,7:13] 

            # replace inf with really high values
            denoised_L[torch.isinf(denoised_L)] = 1e5
        else:
            denoised_rest = kwargs['denoised_rest']
            denoised_L = kwargs['denoised_L']

        # shortest_distance_rest = best_fit_euclidean_distance(denoised_rest[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
        # shortest_distance_L = best_fit_geodesic_distance(denoised_L[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
        # print(f"denoised {i} rest distance {shortest_distance_rest}, entspricht sigma {shortest_distance_rest / 10100}")
        # print(f"denoised {i} L distance {shortest_distance_L}, entspricht sigma {shortest_distance_L / 3110} ")

        direction = log_map_at_L(denoised_L.reshape(-1,6), x_L.reshape(-1,6)) / t_cur
        x_L_next = exp_map_at_L(-(t_next - t_cur) * direction, x_L.reshape(-1,6)).reshape(B, N, 6)

        # step_dist = best_fit_geodesic_distance(x_L_next.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000])
        # print(f"Iteration {i} step best fit distance: {step_dist}")

        d_cur = (x_rest_hat - denoised_rest) / t_hat # d_i in pseudo code
        x_rest_next = x_rest_hat + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        # shortest_distance_rest_step = best_fit_euclidean_distance(x_rest_next[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
        # shortest_distance_L_step = best_fit_geodesic_distance(x_L_next[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))
        # print(f"step {i} rest distance {shortest_distance_rest_step}, entspricht sigma {shortest_distance_rest_step / 10100}")
        # print(f"step {i} L distance {shortest_distance_L_step}, entspricht sigma {shortest_distance_L_step / 3110}")

        # print(f"Distanz original zu Fortschritt mit Schritt: \n{geodesic_distance(x_L_next.reshape(-1,6),noisy_start_L)}")
        # Apply 2nd order correction.
        if i < num_steps - 1:
            # replace infs again
            x_L_next = x_L_next.to(dtype)
            x_L_next[torch.isinf(x_L_next)] = 1e5
            x_in_2nd = torch.cat([x_rest_next, x_L_next], dim=-1)
            if not debug:
                denoised = self( # D_theta in paper
                    x_in_2nd.to(dtype),
                    t_next.repeat(B).to(dtype),
                    context,
                    post_context,
                ).to(torch.float64)

                denoised_rest_2nd = denoised[:,:,:7]
                denoised_L_2nd = denoised[:,:,7:13]
                # replace inf with really high values
                denoised_L_2nd[torch.isinf(denoised_L_2nd)] = 1e5
            else:
                denoised_rest_2nd = kwargs['denoised_rest']
                denoised_L_2nd = kwargs['denoised_L']

            direction_2 = log_map_at_L(denoised_L_2nd.reshape(-1,6), x_L_next.reshape(-1,6)) / t_next
            x_L_next = exp_map_at_L(-(t_next - t_cur) * (direction + direction_2) / 2, x_L.reshape(-1,6)).reshape(B, N, 6)
            x_L_next[torch.isinf(x_L_next)] = 1e5

            d_prime = (x_rest_next - denoised_rest_2nd) / t_next # d'_i in pseudo code
            x_rest_next = x_rest_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code

            # shortest_distance_rest_step_2 = best_fit_euclidean_distance(x_rest_next[0].reshape(-1,7),gt_data_diffusion_space[0].reshape(-1,7))
            # shortest_distance_L_step_2 = best_fit_geodesic_distance(x_L_next[0].reshape(-1,6),gt_x_L[0].reshape(-1,6))

            # print(f"step 2nd {i} rest distance {shortest_distance_rest_step_2}, entspricht sigma {shortest_distance_rest_step_2 / 10100}")
            # print(f"step 2nd {i} L distance {shortest_distance_L_step_2}, entspricht sigma {shortest_distance_L_step_2 / 3110} ")

        x_rest = x_rest_next
        x_L = x_L_next

    if with_pbar:
        ts.close()

    if reverse_ode:
        data = torch.cat([x_rest, x_L], dim=-1)
    else:
        # we were in diffusion space previously, so naturally we have to go back after sampling
        if not debug:
            # we were in diffusion space previously, so naturally we have to go back after sampling
            data = self.reparam.diffusion_to_data(x_rest, context)
            data = torch.cat([data, x_L], dim=-1)
            # print(best_fit_geodesic_distance(x_L.reshape(-1,6)[:4000],gt_L.reshape(-1,6)[:4000]))
        else: 
            data = torch.cat([x_rest, x_L], dim=-1)
    return data