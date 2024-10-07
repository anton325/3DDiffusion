import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from unified_colors import PSNR_COLOR, SSIM_COLOR, LPIPS_COLOR


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Data setup
lambdas = [50, 500, 10000, 100000, 1000000]
psrns = [19.31, 20.3, 20.43, 19.801, 18.27]
lpips = [0.1608, 0.1378, 0.1556, 0.2445, 0.35]
ssim = [0.8455, 0.853, 0.8441, 0.8024, 0.74]

# Create a figure with three subplots side by side
fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjust figure size as needed

# Plot for PSNR
axs[0].plot(lambdas, psrns, marker='o', linestyle='-', color=PSNR_COLOR, label='PSNR')
axs[0].set_xscale('log')
axs[0].set_xlabel('Lambda')
axs[0].set_ylabel('PSNR Values')
axs[0].set_title('PSNR Metrics')
axs[0].grid(False)

# Plot for LPIPS
axs[2].plot(lambdas, lpips, marker='o', linestyle='-', color=LPIPS_COLOR, label='LPIPS')
axs[2].set_xscale('log')
axs[2].set_xlabel('Lambda')
axs[2].set_ylabel('LPIPS Values')
axs[2].set_title('LPIPS Metrics')
axs[2].grid(False)

# Plot for SSIM
axs[1].plot(lambdas, ssim, marker='o', linestyle='-', color=SSIM_COLOR, label='SSIM')
axs[1].set_xscale('log')
axs[1].set_xlabel('Lambda')
axs[1].set_ylabel('SSIM Values')
axs[1].set_title('SSIM Metrics')
axs[1].grid(False)

# Set a large overarching title for the entire figure
fig.suptitle('Comparison of Image Quality Metrics Across Different Lambda Values', fontsize=16)

# Save the plot
path = Path("plots/plots")
path.mkdir(parents=True, exist_ok=True)
fig.savefig(path / "line_plot_splat_lambda.png", bbox_inches='tight')
fig.savefig(path / "line_plot_splat_lambda.pdf", bbox_inches='tight')

plt.show()
