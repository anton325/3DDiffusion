import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import math

class LogUniformSchedule(nn.Module):
    """
    LogUniform noise schedule which seems to work better in our (GECCO) context.

    alle schedules returnen einfach nur für jedes n ein sigma, 
    sie werden gecalled mit schedule(samples) und samples hat shape (batchsize, num_points, 3)
    und dann gibt er für jedes element im batch ein sigma
    """

    def __init__(self, max: float, min: float = 0.002, low_discrepancy: bool = True):
        """
        mit low_discrepancy = True kann man dafür sorgen, dass es nicely ausgebreitet ist über den zur Verfügung stehenden Zahlenraum
        """
        super().__init__()

        self.sigma_min = min
        self.sigma_max = max
        self.log_sigma_min = math.log(min)
        self.log_sigma_max = math.log(max)
        self.low_discrepancy = low_discrepancy

    def forward(self, data):
        def ones(n: int):
            return (1,) * n
        u = torch.rand(data.shape[0], device=data.device) # uniform distribution between 0,1

        if self.low_discrepancy:
            div = 1 / data.shape[0]
            u = div * u
            u = u + div * torch.arange(data.shape[0], device=data.device)

        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma.reshape(-1, *ones(data.ndim - 1))
    
    def return_schedule(self,n):
        u = torch.linspace(0,1,n).cuda()
        sigma = (
            u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min
        ).exp()
        return sigma
    

def t_steps(
    num_steps: int, sigma_max: float, sigma_min: float, rho: float
):
    """
    Returns an array of sampling time steps for the given parameters.
    """
    step_indices = torch.arange(
        num_steps, dtype=torch.float64,
    )
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    return t_steps

if __name__ == "__main__":
    schedule = LogUniformSchedule(165.0)
    n = 128
    s = schedule.return_schedule(n)
    fig,ax = plt.subplots()
    x = np.arange(0,n)
    ax.plot(x,s.cpu().numpy())
    plt.savefig('loguniformschedule.png')

    fig,ax = plt.subplots()
    x = np.arange(0,n)
    ax.scatter(x,s.cpu().numpy(),s=0.1)
    plt.savefig('loguniformschedule_scatter.png')

    print(s)
    print(sum(s>1))


    noise_schedule = torch.linspace(0.05, 1.25, 256).cuda()
    noise_schedule = noise_schedule**2 + 0.0001
    print(noise_schedule)
    print(sum(noise_schedule>1))

    discrete_steps = t_steps(64, 165, 0.002, 7)
    fig,ax = plt.subplots()
    x = np.arange(0,65)
    ax.plot(x,discrete_steps.cpu().numpy())
    plt.savefig('discretesteps.png')
