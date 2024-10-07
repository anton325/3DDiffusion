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
from gecco_torch.train_forward import vanilla, cholesky_L, so3_x0

from gecco_torch.utils.render_tensors import render_fn_options



class LikelihoodCallback(pl.Callback):
    """
    A callback which does inverse sampling to calculate the likelihood of the data given the distribution we usually start from when doing forward sampling
    """

    def __init__(self, 
                 mode,
                 batch_size,
                 n,
                 n_steps: int = 64, 
                 ):
        super().__init__()
        
        self.n = n
        self.n_steps = n_steps
        self.mode = mode
        self.batch: GaussianExample | None = None
        self.likelihood = 0
        self.number_considered = 0
        self.log_fun = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        print("On validation epoch_start likelihood")
        if self.log_fun is None:
            self.log_fun = pl_module.log
        skip = False
        with torch.no_grad():
            
            if self.number_considered < self.n:
                if Mode.cholesky in self.mode:
                    likelihood = cholesky_L.likelihood(pl_module, batch)
                elif Mode.so3_x0 in self.mode:
                    likelihood = so3_x0.likelihood(pl_module, batch)
                elif Mode.so3_diffusion in self.mode:
                    skip = True
                else:
                    likelihood = vanilla.likelihood(pl_module, batch)
                self.number_considered += batch.data.shape[0]
                if not skip:
                    self.log_fun("likelihood", likelihood, on_step=True, batch_size=batch.data.shape[0])

    def on_validation_epoch_end(
        self,
        outputs: Any,
        batch: GaussianExample,
    ) -> None:
        self.number_considered = 0
