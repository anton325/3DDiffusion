"""
Definition of reparameterization schemes. Each reparameterization scheme
implements a `data_to_diffusion` and `diffusion_to_data` method, which
convert the data between "viewable" representation (simple xyz) and the
"diffusion" representation (possibly normalized, logarithmically spaced, etc).
"""
import torch
from torch import Tensor
from kornia.geometry.camera.perspective import project_points, unproject_points

from gecco_torch.structs import GaussianContext3d, Mode

def get_reparam_parameters(mode):
    if Mode.normal in mode and Mode.in_world_space in mode and not Mode.rgb in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02  ,4.9520561e-01,\
                            3.7242898e-01 , 2.2604160e-01 ,-7.5496311e+00 ,-7.5219121e+00,\
                            -8.9276352e+00 , 1.2672280e+00 , 3.5620501e-04 , 8.8820038e-03,\
                            4.1175491e-04,  8.3008547e+00]) 
        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067, 0.99246985, 1.0316856  ,1.0882676,\
                            3.830141   ,3.8691757  ,4.2265983  ,0.5354247  ,0.22764333 ,0.20837784,\
                            0.18193687 ,4.5544424 ])
        
    elif Mode.xyz_sh in mode and Mode.in_world_space in mode and not Mode.rgb in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02  ,4.9520561e-01,\
                            3.7242898e-01 , 2.2604160e-01 ]) 
        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067, 0.99246985, 1.0316856  ,1.0882676])

    elif Mode.xyz_scaling_rotation in mode and Mode.in_world_space in mode and not Mode.rgb in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02,-7.5496311e+00 ,-7.5219121e+00,\
                            -8.9276352e+00 , 1.2672280e+00 , 3.5620501e-04 , 8.8820038e-03, 4.1175491e-04]) 
        
        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067, 3.830141   ,3.8691757  ,4.2265983, 0.5354247  ,0.22764333 ,0.20837784, 0.18193687])
        
    elif Mode.xyz_scaling in mode and Mode.in_world_space in mode and not Mode.rgb in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02  ,-7.5496311e+00 ,-7.5219121e+00,\
                            -8.9276352e+00]) 
        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067,\
                            3.830141   ,3.8691757  ,4.2265983 ])
        
    elif Mode.xyz_rotation in mode and Mode.in_world_space in mode and not Mode.rgb in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02 , 1.2672280e+00 , 3.5620501e-04 , 8.8820038e-03,\
                            4.1175491e-04]) 
        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067, ])
        
    elif Mode.xyz_opacity in mode and Mode.in_world_space in mode and not Mode.rgb in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02,  8.3008547e+00]) 
        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067,4.5544424 ])
        
    elif Mode.normal in mode and Mode.in_world_space in mode and Mode.rgb in mode:
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00,  1.2619663e+00, -7.2187868e-05,  7.5299055e-03,\
                            1.6448057e-04,  8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972,   0.54818636, 0.22542284, 0.20672962, \
                            0.18788442, 4.6488266 ])
        
    elif Mode.normal_opac in mode and Mode.in_world_space in mode and Mode.rgb in mode:
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00,  1.2619663e+00, -7.2187868e-05,  7.5299055e-03,\
                            1.6448057e-04])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972,   0.54818636, 0.22542284, 0.20672962, \
                            0.18788442])
        
    elif Mode.normal_gt_rotations in mode and Mode.in_world_space in mode and Mode.rgb in mode:
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00,  8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972, 4.6488266 ])
        
    elif Mode.gt_rotations_canonical in mode and Mode.in_world_space in mode and Mode.rgb in mode:
        mean=torch.tensor([ 0.01326304 , -0.01196139,   0.01989827 ,  0.64496136 ,  0.6067938,\
                            0.57008225 , -4.2432656 ,  -7.900725  , -11.799877  ,   8.301066,  ])
        sigma=torch.tensor([0.23507258, 0.22746627, 0.17036082, 0.2790444 , 0.29125118 ,0.30589345,\
                            0.9229725 , 2.994794  , 3.1104007 , 4.5833945 ])
        
    elif Mode.normal in mode and Mode.in_camera_space in mode:
        mean=torch.tensor([ 4.8089529e-05, -1.1701969e-02 , 1.6838436e+00 ,5.4624385e-01,\
                            4.2480728e-01 , 2.9202390e-01, -7.5787973e+00 ,-7.4542193e+00,\
                            -8.8554258e+00 , 2.0670894e-01,  5.0425225e-01 , 4.8853916e-01,\
                            -1.8264063e-01 , 8.3796606e+00]) 
        sigma=torch.tensor([0.22739889, 0.20584697, 0.20211232 ,0.961495 ,  1.0030788 , 1.0607022,\
                            3.8616502 , 3.8394434,  4.2717657 , 0.427957  , 0.7460296 , 0.74092704,\
                            0.4333618 , 4.6431065 ])
        
    elif Mode.only_xyz in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 1.4861753e-02, -1.3030048e-02,  1.5905004e-02])

        sigma=torch.tensor([0.23680757, 0.22555158 ,0.16818067])

    elif Mode.only_xyz in mode and Mode.in_camera_space in mode:
        mean=torch.tensor([  8.8730734e-04, -1.3547215e-02 ,  1.6827493e+00])

        sigma=torch.tensor([0.22620456, 0.20213033, 0.20713408])
        
    elif Mode.isotropic_rgb in mode and Mode.in_camera_space in mode:
        # mit i=70 berechnet
        mean=torch.tensor([-9.0636930e-04, -1.2507324e-02,  1.8224740e+00,  6.5684569e-01,
                            6.1902285e-01,  5.7762039e-01, -8.0658159e+00 , 8.3834028e+00]) 
        sigma=torch.tensor([0.22614507, 0.2017216 , 0.24125136, 0.276944 ,  0.28800848, 0.3023833,
                            1.8794155 , 4.5370426])
        
    elif Mode.isotropic_rgb in mode and Mode.in_world_space in mode:
        # mit i=500 berechnet
        mean=torch.tensor([ 1.4362294e-02 ,-1.3202298e-02 , 2.2774480e-02  ,6.4743823e-01,\
                            6.1384833e-01 , 5.6792581e-01 ,-7.8748960e+00  ,8.2941589e+00]) 
        sigma=torch.tensor([0.22830224, 0.22298233, 0.17920762, 0.27750957 ,0.28848377 ,0.30566606, \
                            3.031977  , 4.5717874, ])
        
    elif Mode.lie_rotations_wrong in mode and Mode.in_world_space in mode:
        # mit i=70 berechnet
        mean=torch.tensor([ 1.3963073e-02 ,-1.4282994e-02 , 1.8466769e-02,  5.2403539e-01,\
        4.0525046e-01 , 2.6789317e-01, -7.5206585e+00 ,-7.5742846e+00,\
        -8.8459759e+00 , 1.1459263e-04,  1.5577792e-02, -1.0328351e-03,\
        8.4338923e+00])

        sigma=torch.tensor([0.235859  , 0.2214013 , 0.17719182, 0.9556701 , 1.0030986 , 1.059124,\
        3.8550124 , 3.8684397 , 4.2957907 , 0.40797395, 0.406119 ,  0.39119327,\
        4.625224  ])

    elif Mode.rgb in mode and Mode.rotation_matrix_mode in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 1.3922053e-02 ,-1.1898792e-02 , 2.0099899e-02 , 6.4417851e-01, \
                            6.0855269e-01 , 5.6955040e-01, -7.6414661e+00 ,-7.4200335e+00,\
                            -8.8608370e+00 , 8.5014206e-01 ,-1.5407366e-03 , 1.8359296e-02,\
                            7.3466270e-04 , 8.3521098e-01 ,-5.2811945e-04 ,-1.8683912e-02,\
                            6.7414218e-03 , 8.1332916e-01])#,  8.3119402e+00])

        sigma=torch.tensor([0.22892031, 0.22720556 ,0.17366375 ,0.27895942 ,0.29069296, 0.30584064,\
                            3.8416526 , 3.8323114,  4.257036 ,  0.29360428 ,0.42984894 ,0.57643104,\
                            0.43081707, 0.34523875, 0.71162236, 0.57572967 ,0.7122642,  0.36856097])#,\
                            # 4.5624323 ])
        
    elif Mode.log_L in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 1.3443031e-02, -1.1173394e-02 , 1.6560141e-02,  5.2075750e-01,\
                            3.8556287e-01,  2.3774694e-01 , 8.3209829e+00 ,-5.7257214e+00,\
                            -7.1717815e+00, -9.4089594e+00 , 3.6587730e-06 , 3.1100502e-05, \
                            -2.8015775e-04])

        sigma=torch.tensor([0.23454128 ,0.22688565, 0.17105229, 0.978251,  1.032045,   1.0880752, \
                            4.5698853 , 1.7659042 , 2.4440556 , 2.019935 ,  0.01091467, 0.00850422,\
                            0.00697142])

    elif Mode.rgb in mode and Mode.cov_matrix_3x3 in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 1.3494034e-02, -1.4271554e-02 , 1.7335776e-02, 6.4477903e-01, \
                            6.0748762e-01,  5.7559872e-01,  8.3629519e-04, -1.0098650e-07,\
                            -2.0604456e-07, -1.0098650e-07 , 7.7705149e-04, -4.4734961e-06, \
                            -2.0604456e-07, -4.4734961e-06, 6.8550603e-04])
        sigma=torch.tensor([2.3130791e-01, 2.2562689e-01, 1.7164323e-01, 2.7548730e-01, 2.8930408e-01,
                            3.0290458e-01, 2.3553984e-02, 1.2040257e-04, 8.0512560e-05,1.2040257e-04,
                            2.3548391e-02, 9.7148812e-05, 8.0512560e-05, 9.7148812e-05,2.3549171e-02])


    elif Mode.rgb in mode and Mode.rotation_matrix_mode in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 1.2945357e-02 ,-1.2970554e-02 , 2.2214262e-02 , 6.4211392e-01,\
                            6.0509813e-01 , 5.6611013e-01 ,-7.6385126e+00 -7.4772782e+00,\
                            -8.9027920e+00 , 8.4999079e-01,  2.2525671e-03 , 1.8808544e-02,\
                            -2.4929470e-03 , 8.3348459e-01, -3.2131465e-03, -1.9199083e-02,\
                            3.6250320e-03 , 8.1053758e-01])# , 8.4201365e+00])

        sigma=torch.tensor([0.22785449, 0.22982158, 0.177012  , 0.27846038, 0.2917578 , 0.3062492,\
                            3.8907576 , 3.9113941 , 4.2936463 , 0.3060506 , 0.43439245, 0.5971719,\
                            0.43512776 ,0.35850567, 0.7455628 , 0.59661776, 0.7460839 , 0.3918429,])#\
                            #4.6626925 ])

    elif Mode.only_xyz_cov in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 1.4362294e-02 ,-1.3202298e-02 , 2.2774480e-02 , 1.9467224e+00 , 1.8979480e+00, \
                            2.0142193e+00 , 4.7313985e-03 , 1.5664754e-02 , 1.3795926e-01])

        sigma=torch.tensor([0.23316363, 0.22798394, 0.18033974, \
                            0.47234952, 0.45922846, 0.46038923, 2.285684,  2.2745519,  2.3797605])
        
    elif Mode.fill_xyz in mode and Mode.in_world_space in mode:
        mean=torch.tensor([ 0.01381367, -0.01018987, 0.01920099,  0.,          0.,          0.,\
                            0.     ,     0.      ,    0.   ,       0.    ,      0.  ,        0., \
                            0.      ,    0.      ,  ])
        
        sigma=torch.tensor([0.22960462 ,0.22404285, 0.18151371, 1     ,    1    ,   1, \
                1     ,    1      ,   1    ,     1     ,    1    ,     1, \
                1      ,   1   ,     ])
    elif Mode.so3_diffusion in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        wir reparameterisieren nur xyz, rgb, scale, opacity
        """
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00, 8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972, 4.6488266 ])
        
    elif Mode.so3_x0 in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        wir reparameterisieren nur xyz, rgb, scale, opacity
        """
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00, 8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972, 4.6488266 ])
    elif Mode.so3_diffusion in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        wir reparameterisieren nur xyz, rgb, scale, opacity
        """
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00, 8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972, 4.6488266 ])
        
    elif Mode.no_rotation in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        wir reparameterisieren nur xyz, rgb, scale, opacity
        """
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00, 8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            3.8697808,  3.8589494,  4.231972, 4.6488266 ])
        
    elif Mode.cholesky in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        wir reparameterisieren nur xyz, rgb, opacity
        """
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01,  8.2221346e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, 4.6488266 ])
        
    elif Mode.activated_scales in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        """
        mean=torch.tensor([ 1.3108904e-02, -1.4706432e-02,  2.1905240e-02,  6.3924962e-01,\
                            6.0627502e-01,  5.6895167e-01,  8.7499488e-03,  8.5165016e-03,\
                            5.6827241e-03,  1.2570626e+00, -6.5995250e-06,  8.2122758e-03,\
                            -6.8666448e-04,  8.2726393e+00])
        sigma=torch.tensor([0.22674118, 0.22558771 ,0.18274911, 0.27996165, 0.29173335, 0.30702743,\
                            0.01349693, 0.0122104 , 0.01048797, 0.53082716, 0.221474  , 0.20775917, \
                            0.18827651, 4.621929  ])
        
    elif Mode.procrustes in mode and Mode.rgb in Mode and Mode.in_world_space in mode:
        """
        """
        mean=torch.tensor([ 1.4290623e-02, -9.6069984e-03,  2.1215543e-02,  6.4854586e-01,\
                            6.1227012e-01,  5.7130802e-01, 8.2221346e+00, -7.6133890e+00, -7.4436655e+00,\
                            -8.7931414e+00])
        sigma=torch.tensor([0.22648855, 0.22809915, 0.1794058,  0.27678514, 0.2900197 , 0.30645472, \
                            4.6488266, 3.8697808,  3.8589494,  4.231972])
    

    else:
        raise Exception(f"no gaussian reparam parameters for {mode} implemented")

    return mean,sigma

class Reparam(torch.nn.Module):
    """
    Base class for reparameterization schemes.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

    def data_to_diffusion(self, data: Tensor, ctx: GaussianContext3d) -> Tensor:
        raise NotImplementedError()

    def diffusion_to_data(self, diff: Tensor, ctx: GaussianContext3d) -> Tensor:
        raise NotImplementedError()


class NoReparam(Reparam):
    """
    A placeholder for non-conditional models.
    """

    def data_to_diffusion(self, data: Tensor, ctx: GaussianContext3d) -> Tensor:
        return data

    def diffusion_to_data(self, diff: Tensor, ctx: GaussianContext3d) -> Tensor:
        return diff


class GaussianReparam(Reparam):
    """
    Maps data to diffusion space by normalizing it with the mean and standard deviation.
    """

    def __init__(self, mean: Tensor, sigma: Tensor):
        assert mean.ndim == 1 # wieso hat mean dimension 1?
        assert mean.shape == sigma.shape

        super().__init__(mean.shape[0])
        # register buffer: keine module parameters, kein gradient, aber sie werden im state dict included. Außerdem kann man sie mit self.name ansprechen
        self.register_buffer("mean", mean) 
        self.register_buffer("sigma", sigma)

    def data_to_diffusion(self, data: Tensor, ctx: GaussianContext3d) -> Tensor: # normalizing it
        # only deletes in scope of this function aka we dont use context for gaussian reparametrization
        del ctx
        return (data - self.mean) / self.sigma

    def diffusion_to_data(self, diff: Tensor, ctx: GaussianContext3d) -> Tensor:
        del ctx
        return diff * self.sigma + self.mean

    def extra_repr(self) -> str:
        return f"mean={self.mean.flatten().tolist()}, sigma={self.sigma.flatten().tolist()}"


class UVLReparam(Reparam):
    """
    Maps data to diffusion space by projecting it to the image plane and
        1. taking the logarithm of the depth to map from [0, inf] to [-inf, inf]
        2. using the inverse hyperbolic tangent on projection coordinates (h, w)
           to map them from [0, 1] to [-inf, inf]
    and finally normalizing the result with the mean and standard deviation (like GaussianReparam).

    This allows for modelling point clouds constrained to the viewing frustum. Since a point exactly
    at an image border would have infinite inverse tanh, we relax the constraint a bit by mapping the
    projection coordinates to as if they were in [0, logit_scale] instead of [0, 1].
    """

    def __init__(self, mean: Tensor, sigma: Tensor, logit_scale: float = 1.1):
        assert mean.shape == (3,)
        assert sigma.shape == (3,)

        super().__init__(dim=3)

        self.register_buffer("uvl_mean", mean)
        self.register_buffer("uvl_std", sigma)
        self.logit_scale = logit_scale

    depth_to_real = staticmethod(torch.log)
    real_to_depth = staticmethod(torch.exp)

    def extra_repr(self) -> str:
        return (
            f"uvl_mean={self.uvl_mean.flatten().tolist()}, "
            f"uvl_std={self.uvl_std.flatten().tolist()}, "
            f"logit_scale={self.logit_scale}"
        )

    def _real_to_01(self, r: Tensor) -> Tensor:
        """
        Maps a real number (reparametrized projection height, width) to [0, 1]
        """
        s = torch.tanh(r)
        s = s * self.logit_scale
        s = s + 1.0
        s = s / 2
        return s

    def _01_to_real(self, s: Tensor) -> Tensor:
        """
        Maps a number in [0, 1] (projection height, width) to a real number
        """
        s = 2 * s
        s = s - 1.0
        s = s / self.logit_scale
        r = torch.arctanh(s)
        return r

    def xyz_to_hwd(self, xyz: Tensor, ctx: GaussianContext3d) -> Tensor:
        """
        Maps a point cloud to (image_plane, depth).
        """
        hw = project_points(xyz, ctx.K.unsqueeze(1))
        d = torch.linalg.norm(xyz, dim=-1, keepdim=True)

        return torch.cat([hw, d], dim=-1)

    def hwd_to_xyz(self, hwd: Tensor, ctx: GaussianContext3d) -> Tensor:
        """
        Maps (image_plane, depth) to a point cloud.
        """
        hw, d = hwd[..., :2], hwd[..., 2:]
        xyz = unproject_points(hw, d, ctx.K.unsqueeze(1), normalize=True)
        return xyz

    def hwd_to_uvl(self, hwd: Tensor) -> Tensor:
        """
        Maps (image_plane, depth) to R^3 (including normalization).
        """
        assert hwd.shape[-1] == 3

        h, w, d = hwd.unbind(-1)

        uvl_denorm = torch.stack(
            [
                self._01_to_real(h),
                self._01_to_real(w),
                self.depth_to_real(d),
            ],
            dim=-1,
        )

        uvl_norm = (uvl_denorm - self.uvl_mean) / self.uvl_std
        return uvl_norm

    def uvl_to_hwd(self, uvl: Tensor) -> Tensor:
        """
        Maps R^3 (including normalization) to (image_plane, depth).
        """
        assert uvl.shape[-1] == 3

        uvl_denorm = uvl * self.uvl_std + self.uvl_mean
        u, v, l = uvl_denorm.unbind(-1)

        hwd = torch.stack(
            [
                self._real_to_01(u),
                self._real_to_01(v),
                self.real_to_depth(l),
            ],
            dim=-1,
        )

        return hwd

    def data_to_diffusion(self, data: Tensor, ctx: GaussianContext3d) -> Tensor:
        """
        Maps a point cloud to R^3 (including normalization).
        """
        assert isinstance(ctx, GaussianContext3d)

        xyz = data
        hwd = self.xyz_to_hwd(xyz, ctx)
        uvl = self.hwd_to_uvl(hwd)

        return uvl

    def diffusion_to_data(self, diff: Tensor, ctx: GaussianContext3d) -> Tensor:
        """
        Maps R^3 (including normalization) to a point cloud.
        """
        assert isinstance(ctx, GaussianContext3d)

        uvl = diff
        hwd = self.uvl_to_hwd(uvl)
        xyz = self.hwd_to_xyz(hwd, ctx)

        return xyz
