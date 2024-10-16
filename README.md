# Image-conditioned Gaussian Splatting Diffusion

 [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) is a powerful method for learning 3D structures and enabling high-fidelity novel view synthesis. 

To circumvent long optimization times and the dense accurately posed dataset requirements, in this project, Tyszkiewicz et al.’s point cloud diffusion model [GECCO](https://arxiv.org/abs/2303.05916) is extended to Gaussian Splatting point clouds. This allows generation of Gaussian Splatting scenes either conditionally on an image or unconditionally for a certain class.

Diffusion is the process of adding noise to samples from an unknown distribution with a fixed noise schedule that guarantees transformation of the original sample to a data point from $\mathcal{N}(0,\sigma_{\max} I)$. A neural network $p_\theta$ learns how to undo the noising process, which allows transforming a sample from $\mathcal{N}(0,\sigma_{\max} I)$ to a sample from the target distribution.

![Diffusion](assets/diffusion.png "Markov Chain for Diffusion")


## Method

During training, the Gaussian scene is noised based on the noise level t and projected onto a [ConvNeXT-tiny](https://arxiv.org/abs/2201.03545)-derived feature map. This enhanced point cloud is denoised with the Set Transformer. The loss is calculated by comparing the denoised scene against the ground truth scene and photometrically against a ground truth image.
![Method Overview](assets/Methode.png "Method overview of the project")

The denoising backbone is based on Lee et al.'s [Set Transformer](https://arxiv.org/abs/1810.00825) which reduces attention's quadratic complexity to one that is linear in the number of data points w.r.t. the number of Learned Inducers.
<img src="assets/set_transformer.png" alt="Set Transformer" title="Set Transformer to reduce complexity" style="width: 70%; display: block; margin: 0 auto;"/>




## Conditional generation
From the different investigated methods, the Procrustes and SO(3) methods emerged as the most effective. Both methods perform diffusion on the Gaussian parameters in the Euclidean space, but adopt distinct strategies for handling the rotational parts of the Gaussian points. [Procrustes](https://arxiv.org/abs/2103.16317) learns a differentiable mapping from $3\times3$ matrices to rotation matrices and [SO(3)](https://arxiv.org/abs/2312.11707) models the rotations as samples drawn from a rotational distribution, which is the rotational equivalent of the Gaussian normal distribution.

<table>
  <tr>
    <td align="center"><img src="assets/cond_gt.png" /><br>Conditioning image for the diffusion process</td>
    <td align="center"><img src="assets/car_rotate_gt.gif" /><br>Ground truth Gaussian scene</td>
    <td align="center"><img src="assets/proc.gif" /><br>Diffused scene using Procrustes mapping</td>
    <td align="center"><img src="assets/so3.gif"/><br>Generated scene using SO(3) diffusion</td>
  </tr>
</table>


  
## Unconditional generation

<table>
  <tr>
    <td align="center"><img src="assets/uncond_car_1.gif"/></td>
    <td align="center"><img src="assets/uncond_car_2.gif"/></td>
    <td align="center"><img src="assets/uncond_car_3.gif"/></td>
    <td align="center"><img src="assets/uncond_car_4.gif"/></td>
  </tr>
</table>