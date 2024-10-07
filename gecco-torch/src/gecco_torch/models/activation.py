import torch
import torch.nn as nn


class GaussianActivation(nn.Module):
    """
    Using the activation function proposed in
    "Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs" by Ramasinghe et al.
    allows us to skip Fourier embedding low dimensional inputs such as noise level and 3D coordinates.
    """

    def __init__(self, normalized: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.normalized = normalized
        self.eps = 1e-4
        # print(f"Init gaussian act mit normalized: {normalized}")

    def forward(self, x):
        # print(f"Forward GaussianActivation x na: {torch.isnan(x).any()}")
        # print(f"alpha: {self.alpha}")
        if abs(self.alpha) < self.eps:
            print("ERROR IN ACTIVATION")
            print(f"alpha smaller: {self.alpha}")
            y = (-(x**2) / ((2 * self.alpha**2)+self.eps)).exp()
        else:
            y = (-(x**2) / (2 * self.alpha**2)).exp()
        if self.normalized:
            # normalize by activation mean and std assuming
            # `x ~ N(0, 1)`
            y = (y - 0.7) / 0.28
        yna = torch.isnan(y).any()
        # print(f"y na: {yna}")
        # print(f"y shape: {y.shape}") # erstes mal in broadcastlayers: y shape: torch.Size([batch size, 64, 768]), zweites mal: y shape: torch.Size([batch_size, 2048, 768])
        if yna:
            print("NA IN ACTIVATION")
            print(f" alpha: {self.alpha}")
        return y
