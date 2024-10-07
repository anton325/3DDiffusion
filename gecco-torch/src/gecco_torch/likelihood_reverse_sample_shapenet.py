from typing import Any
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import numpy as np
import pathlib
import datetime
import wandb
from gecco_torch.diffusion import Diffusion
from gecco_torch.structs import GaussianExample,GaussianContext3d, Camera, Mode
from gecco_torch.scene.gaussian_model import GaussianModel
from gecco_torch.utils.loss_utils import l1_loss
from gecco_torch.train_forward import vanilla, cholesky_L

from gecco_torch.utils.render_tensors import render_fn_options
from scipy.stats import multivariate_normal



class LikelihoodCallback(pl.Callback):
    """
    A callback which does inverse sampling to calculate the likelihood of the data given the distribution we usually start from when doing forward sampling
    """

    def __init__(self, 
                 n,
                 n_steps: int = 64, 
                 ):
        super().__init__()
        
        self.n = n
        self.n_steps = n_steps
        self.batch: GaussianExample | None = None
        self.likelihood = 0
        self.number_considered = 0
        self.log_fun = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        print("On validation epoch_start likelihood")
        if self.log_fun is None:
            self.log_fun = pl_module.log

        if self.number_considered < self.n:
            kwargs = {
                "reverse" : True,
                'gt_data' : batch.data,
            }
            reverse_samples = pl_module.sample_stochastic(
                shape=batch.data.shape,
                context=batch.ctx,
                sigma_max=pl_module.sigma_max,
                num_steps=self.n_steps,
                **kwargs,
            )
            mu = np.zeros(3)  # Replace with the 13-dimensional mean vector
            sigma = np.diag(165*np.ones(3))  # Replace with the 13x13 covariance matrix
            # Initialize the multivariate normal distribution
            mvn = multivariate_normal(mean=mu, cov=sigma)
            prob_densities = mvn.pdf(reverse_samples.detach().cpu().numpy())

            
            # Calculate the joint likelihood
            sum_likelihood = np.sum(-np.log(prob_densities+1e-10))
            likelihood = sum_likelihood / (reverse_samples.shape[0] * reverse_samples.shape[1])
        
            self.number_considered += batch.data.shape[0]
        
            self.log_fun("likelihood", likelihood , on_step=True, batch_size=batch.data.shape[0])
            print(f"Likelihood: {likelihood}")
            # sample_distribution = np.random.multivariate_normal(mu, sigma, reverse_samples.shape[0] * reverse_samples.shape[1])
            # prob_densities_sample_dist = mvn.pdf(sample_distribution)
            # sum_likelihood_sample_dist = np.sum(-np.log(prob_densities_sample_dist+1e-10))
            # likelihood_sample_dist = sum_likelihood_sample_dist / (reverse_samples.shape[0] * reverse_samples.shape[1])

    def on_validation_epoch_end(
        self,
        outputs: Any,
        batch: GaussianExample,
    ) -> None:
        self.number_considered = 0
