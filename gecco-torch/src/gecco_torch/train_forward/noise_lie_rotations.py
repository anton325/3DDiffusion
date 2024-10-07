import torch
from gecco_torch.utils.lie_utils import batched_lietorch_tangential_to_rotation_matrix


def forward(self,net,examples,context, log_fun):
    ex_diff = net.reparam.data_to_diffusion(examples, context) # reparametrisierte Punktwolke
    # print(f"forward shape ex diff: {ex_diff.shape}") # (batchsize,num points, 3)
    sigma = self.schedule(ex_diff)
    # print(f"Sigma: {sigma}")
    # print(f"forward shape sigma: {sigma.shape}") # (batchsize,1,1)
    weight = (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    # calculate the noise
    """
    lie rotations: noise auf die rotationen, indem 1x3 noise im euclidian raum gesampled wird und mit lietorch.exp die rotation. 
    Dann werden die gt rotationen darum gedreht
    """
    n = torch.randn((ex_diff.shape[0],ex_diff.shape[1],ex_diff.shape[2] - 6),device = ex_diff.device) * sigma# / 2*torch.linspace(1,sigma.shape[0],sigma.shape[0],device=ex_diff.device).reshape(-1,1,1))
    noised_data = ex_diff.clone()
    rotations = ex_diff[:,:,9:18].view(ex_diff.shape[0], ex_diff.shape[1], 3, 3)
    rotation_noise = batched_lietorch_tangential_to_rotation_matrix(n[:,:,9:12]) # ergebnis 3x3 matrix
    
    # # apply rotation
    resulting_rotations = torch.matmul(rotations, rotation_noise).view(ex_diff.shape[0],ex_diff.shape[1],9)
    if ex_diff.shape[2] > 18:
        noised_data[:,:,18] = ex_diff[:,:,18] + n[:,:,12]
    noised_data[:,:,:9] = ex_diff[:,:,:9] + n[:,:,:9]
    noised_data[:,:,9:18] = resulting_rotations


    D_yn = net(noised_data, sigma, context) # input ist pointcloud, die mit noise verändert wurde, und das sigma


    # compare the orignal vs predicted
    loss = self.loss_scale * weight * ((D_yn - ex_diff) ** 2) # wegen preconditioning mehr stability?
    # print(f"loss na: {loss.isnan().any()}")
    loss_xyz = loss[:,:,:3].mean()
    loss_rest = loss[:,:,3:].mean()
    # if (Mode.warmup_xyz in self.mode and train_step < self.splatting_loss['warmup_steps']) or Mode.only_xyz in self.mode or Mode.fill_xyz in self.mode:
    #     mean_loss = loss_xyz
    # else:
    #     mean_loss = loss_xyz + loss_rest
    # mean_loss = 5*loss_xyz + loss_rest

    log_fun("mean_scale_loss",loss[:,:,6:9].mean(),on_step=True)
    log_fun("mean_rot_loss",loss[:,:,9:18].mean(),on_step=True)

    data = net.reparam.diffusion_to_data(D_yn,context)

    return data, loss, sigma