import torch
import lietorch
from gecco_torch.utils.isotropic_gaussian import IsotropicGaussianSO3
from gecco_torch.utils.build_cov_matrix_torch import build_covariance_from_scaling_rotation_batched

FORCE_SMALL_SCALE = False


def geodesic_distance(L,K):
    lower_triangle_dif = L[:,:,3:6] - K[:,:,3:6]
    frobenius_lower_triangle_dif = torch.square(lower_triangle_dif).sum(dim = -1) # frobenius norm einer jeden einzelnen L matrix

    log_diag_dif = L[:,:,0:3] - K[:,:,0:3]
    frobenius_lower_triangle_diff = torch.square(log_diag_dif).sum(dim = -1)

    distance = torch.sqrt(frobenius_lower_triangle_diff + frobenius_lower_triangle_dif)
    return distance
    


def find_log_L(cov_matrices):
    """
    rein als rotation matrix.
    Dann führen wir die Cholesky Zerlegung durch. Allerdings logaritmieren wir die Einträge auf der Diagonalen.
    Dadurch können wir in der letzten layer vom Netzwerk diese Einträge exponentiieren, damit die Diagonalelemente >= 0
    Output: |L[0]    0      0 |
            |L[1]   L[2]    0 |
            |L[3]   L[4]  L[5]|
    """
    Ls = torch.zeros((cov_matrices.shape[0], cov_matrices.shape[1],6))
    eigenvalues, _ = torch.linalg.eigh(cov_matrices)
    non_pos_def = eigenvalues.min(dim=-1).values <= 0  # Check the smallest eigenvalue

    # Correct non-positive-definite matrices
    """
    Manche haben nicht alle Eigenvalues >=0, das ist nicht erlaubt. Wir nehmen an, dass es numerische Ungenauigkeiten sind, und ersetzen diese einfach durch 0
    und berechnen die Covariance-Matrix neu
    """
    corrected_matrices = cov_matrices.clone()
    for i in torch.where(non_pos_def)[0]:
        for j in torch.where(non_pos_def[i])[0]:
            eigvals, eigvecs = torch.linalg.eigh(cov_matrices[i, j])
            eigvals[eigvals < 0] = 0
            corrected_matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.transpose(-2, -1)
            corrected_matrices[i, j] = corrected_matrix

    L = torch.linalg.cholesky(corrected_matrices)

    # Prepare output tensor
    Ls[..., 0] = torch.log(L[..., 0, 0])
    Ls[..., 1] = torch.log(L[..., 1, 1])
    Ls[..., 2] = torch.log(L[..., 2, 2])
    Ls[..., 3] = L[..., 1, 0]
    Ls[..., 4] = L[..., 2, 0]
    Ls[..., 5] = L[..., 2, 1]

    return Ls

def L_to_cov(Ls_vals):
    # convert L back to covariance matrix using LLT
    Ls = torch.zeros((Ls_vals.shape[0], Ls_vals.shape[1], 3, 3), device=Ls_vals.device)
    Ls[:, :, 0, 0] = Ls_vals[:, :, 0]
    Ls[:, :, 1, 1] = Ls_vals[:, :, 1]
    Ls[:, :, 2, 2] = Ls_vals[:, :, 2]
    Ls[:, :, 1, 0] = Ls_vals[:, :, 3]
    Ls[:, :, 2, 0] = Ls_vals[:, :, 4]
    Ls[:, :, 2, 1] = Ls_vals[:, :, 5]

    # Exponentiate the diagonal elements in-place, weil die wurden ja vorher gelogged
    diagonal = Ls.diagonal(dim1=-2, dim2=-1)
    diagonal_exp = torch.exp(diagonal)
    diagonal.copy_(diagonal_exp)

    # Compute the 3x3 covariance matrices using batch matrix multiplication cov = LLT
    cov3D_precomp_3x3 = torch.matmul(Ls, Ls.transpose(-1, -2))
    return cov3D_precomp_3x3

def forward(self, net, examples, context, data, Mode, train_step, log_fun):
    """
    examples shape: (xyz, rgb, opacity, scale, rotation), rotation in wxyz
    """
    examples_no_rot_no_scale = examples[:,:,:7]
    ex_diff = net.reparam.data_to_diffusion(examples_no_rot_no_scale, context) # reparametrisierte Punktwolke

    ex_diff = torch.cat([ex_diff[:,:,:7],examples[:,:,7:10]],dim=-1)

    sigma = self.schedule(ex_diff)

    weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)


    n = torch.randn_like(ex_diff) * sigma

    noised_data = ex_diff + n

    batch_size = ex_diff.shape[0]

    rotations = examples[:,:,10:14]

    noise_scales = sigma.repeat_interleave(examples.shape[1])

    axis_angles = torch.vmap(lambda mu,s_single: IsotropicGaussianSO3(mu,
                                                            s_single,
                                                            force_small_scale=FORCE_SMALL_SCALE).sample_one_vmap(),
                        randomness="different")(rotations, noise_scales)
    
    noisy_rotation = (lietorch.SO3(rotations) * lietorch.SO3.exp(axis_angles)) #.vec().reshape(batch_size, -1, 4)

    noised_rotations = lietorch.SO3(rotations) * noisy_rotation
    noised_rotations = noised_rotations.vec().reshape(batch_size, -1, 4)

    cov_gt = build_covariance_from_scaling_rotation_batched(examples[:,:,7:10],examples[:,:,10:14])
    cov_noised = build_covariance_from_scaling_rotation_batched(examples[:,:,7:10],noised_rotations)

    log_L_gt = find_log_L(cov_gt)
    log_L_noised = find_log_L(cov_noised)

    """
    noised data: xyz, rgb, opacity, Ls
    """
    noised_data = torch.cat([noised_data[:,:,:7],log_L_noised],dim=-1)

    D_yn = net(noised_data, sigma, context) # input ist pointcloud, die mit noise verändert wurde, und das sigma
    """
    D_yn: xyz, rgb, scale, opacity, Ls
    """

    D_yn_no_L = D_yn[:,:,:7]
    gecco_loss = self.loss_scale * weight * ((D_yn_no_L - ex_diff[:,:,:7]) ** 2) # wegen preconditioning mehr stability?

    Ls = D_yn[:,:,7:13]

    geodesic_distance = geodesic_distance(Ls,log_L_gt)

    denoised_cov3D_precomp_3x3 = L_to_cov(Ls) # natürlich nur shape 6, nicht 3x3 matrizen

    log_fun("gecco_loss",gecco_loss.mean(),on_step=True)

    if Mode.rgb in self.mode:
        log_fun("mean_rgb_loss",gecco_loss[:,:,3:6].mean(),on_step=True)

    log_fun("geodesic_distance", geodesic_distance.mean(), on_step=True)

    log_fun("mean_xyz_loss",gecco_loss[:,:,:3].mean(),on_step=True)
    log_fun("mean_opacity_loss",gecco_loss[:,:,6].mean(),on_step=True)



    D_yn = torch.concat([D_yn[:,:,:7],denoised_cov3D_precomp_3x3] ,dim=-1)

    data = net.reparam.diffusion_to_data(D_yn,context)

    mean_loss = gecco_loss + geodesic_distance

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
    Das netzwerk kriegt die Daten im Format [xyz, rgb, opacity, scalings, rotations]
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
    x ist im format xyz, rgb, opacity, rotations, scales
    """
    model_in = torch.concat([c_in * x[:,:,:7],x[:,:,7:]],dim=-1)
    
    F_x, cache = self.model(
        model_in, c_noise, raw_context, post_context, do_cache, cache
    )
    """
    F_x output format xyz, rgb, opacity, covariance_matrix -> als lower triangle matrix
    """
    # output conditioning: c_skip * x + c_out * F_x
    skip_result = c_skip * x[:,:,:7]
    out_result = c_out * F_x[:,:,:7]
    denoised = skip_result + out_result 
    denoised = torch.cat([denoised,F_x[:,:,7:]],dim=-1)

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
    rotated_mu = (lietorch.SO3(delta_mu) * lietorch.SO3(x)).vec()
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
    X0 = lietorch.SO3([],from_uniform_sampled=(shape[0]*shape[1])).vec().cuda().reshape(shape[0],shape[1], 4) # in xyzw format

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    # combine normal noise with rotation noise
    x_next = torch.cat([x_next[:,:,:10], X0],dim=-1)

    ts = list(enumerate(zip(t_steps[:-1], t_steps[1:])))
    # ts ist liste von tuplen -> t,t+1

    if with_pbar:
        ts = tqdm(ts, unit="step")

    for i, (t_cur, t_next) in ts:  # 0, ..., N-1
        """
        x_cur shape xyz, rgb, scale, opacity, rotations
        """
        x_cur = x_next

        # noise rotations

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, math.sqrt(2.0) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = t_cur + gamma * t_cur
        noise = torch.randn((x_cur.shape[0],x_cur.shape[1],10), device=device, generator=rng, dtype=dtype) * S_noise
        extra_noise = (t_hat**2 - t_cur**2).sqrt() * noise
        x_hat = torch.cat([x_cur[:,:,:10]]) + extra_noise

        """
        x_hat shape xyz, rgb, scale, rotations, opacity
        """
        x_hat = torch.cat([x_hat[:,:,:9],x_cur[:,:,10:],x_hat[:,:,9].unsqueeze(-1)],dim=-1)

        # Euler step.
        denoised = self( # D_theta in pseudo code in paper
            x_hat.to(dtype),
            t_hat.repeat(B).to(dtype),
            context,
            post_context,
        ).to(torch.float64)

        """
        denoised shape xyz, rgb, scale, opacity, rotations, scale
        """

        denoised_rotations_scale = denoised[:,:,10:]
        new_sampled_rotations = fn_sample(x_hat[:,:,10:14], denoised_rotations_scale[:,:,:4], denoised_rotations_scale[:,:,4])

        denoised_no_rotations = denoised[:,:,:10]

        """
        x_hat_no_rotations shape xyz, rgb, scale, opacity
        """
        x_hat_no_rotations = torch.cat([x_hat[:,:,:9], x_hat[:,:,13].unsqueeze(-1)],dim = -1)

        d_cur = (x_hat_no_rotations - denoised_no_rotations) / t_hat # d_i in pseudo code
        x_next = x_hat_no_rotations + (t_next - t_hat) * d_cur # x_i+1 in pseudo code

        """
        x_next shape xyz, rgb, scale, rotations, opacity
        """
        x_next = torch.cat([x_next[:,:,:9], new_sampled_rotations, x_next[:,:,9].unsqueeze(-1)],dim=-1)
        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = self( # D_theta in paper
                x_next.to(dtype),
                t_next.repeat(B).to(dtype),
                context,
                post_context,
            ).to(torch.float64)
            """ 
            denoised shape xyz, rgb, scale, opacity, rotations, scale
            """
            x_next_no_rotations = torch.cat([x_next[:,:,:9], x_next[:,:,13].unsqueeze(-1)],dim=-1)
            denoised_no_rotations = denoised[:,:,:10]

            d_prime = (x_next_no_rotations - denoised_no_rotations) / t_next # d'_i in pseudo code
            x_next = x_hat_no_rotations + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # x_i+1 in pseudo code
            x_next = torch.cat([x_next, new_sampled_rotations],dim=-1)
            """
            x_next shape xyz, rgb, scale, opacity, rotations
            """

    if with_pbar:
        ts.close()

    # we were in diffusion space previously, so naturally we have to go back after sampling
    data = self.reparam.diffusion_to_data(x_next[:,:,:10], context)
    rotations = x_next[:,:,10:]


    data = torch.cat([data, rotations],dim=-1)
    """
    data shape xyz, rgb, scale, opacity, rotations, scale # rotations im xyzw format
    """
    return data