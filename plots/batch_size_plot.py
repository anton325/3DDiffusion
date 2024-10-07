import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from unified_colors import PSNR_COLOR, SSIM_COLOR, LPIPS_COLOR

# Enable LaTeX and set the font to be handled by LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Access the Pastel1 colormap
pastel1 = mpl.cm.get_cmap('Pastel1')

# Data setup
model_names = ['32', '48', '64']
psrns = [20.237, 20.4, 19.72]
lpips = [0.1672, 0.1577, 0.1959]
ssim = [0.841, 0.844, 0.841]

# Numeric labels for x-axis
x = np.arange(len(model_names))

# Create a figure with 3 subplots side by side
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# First subplot for PSNR
ax[0].plot(x, psrns, marker='o', linestyle='-', color=PSNR_COLOR, label='PSNR')
# ax[0].plot(x, psrns, marker='o', linestyle='-', color=pastel1(0), label='PSNR')
ax[0].set_xlabel('Batch size')
ax[0].set_ylabel('PSNR')
ax[0].set_title('PSNR')
ax[0].set_xticks(x)
ax[0].set_xticklabels(model_names)
for i, v in enumerate(psrns):
    if i == 0 or i == 2:
        continue
    ax[0].text(x[i], v+0.005, f"{v:.2f}", fontsize=9, ha='center', va='bottom')
ax[0].text(x[0], psrns[0]+0.015, f"{psrns[0]:.2f}", fontsize=9, ha='center', va='bottom')
ax[0].text(x[2]-0.12, psrns[2]-0.01, f"{psrns[2]:.2f}", fontsize=9, ha='center', va='bottom')

# Second subplot for SSIM
ax[1].plot(x, ssim, marker='s', linestyle='-', color=SSIM_COLOR, label='SSIM')
ax[1].set_xlabel('Batch size')
ax[1].set_ylabel('SSIM')
ax[1].set_title('SSIM')
ax[1].set_xticks(x)
ax[1].set_xticklabels(model_names)
ax[1].set_ylim(0.83, 0.85)  # Set y-axis limits for better visibility
for i, v in enumerate(ssim):
    ax[1].text(x[i], v+0.0005, f"{v:.3f}", fontsize=9, ha='center', va='bottom')

# Third subplot for LPIPS
ax[2].plot(x, lpips, marker='^', linestyle='-', color=LPIPS_COLOR, label='LPIPS')
ax[2].set_xlabel('Batch size')
ax[2].set_ylabel('LPIPS')
ax[2].set_title('LPIPS')
ax[2].set_xticks(x)
ax[2].set_xticklabels(model_names)
for i, v in enumerate(lpips):
    if i == 0 or i == 1 or i == 2:
        continue
    ax[2].text(x[i], v+0.003, f"{v:.4f}", fontsize=9, ha='center', va='bottom')
ax[2].text(x[0]+0.01, lpips[0]+0.0009, f"{lpips[0]:.4f}", fontsize=9, ha='center', va='bottom')
ax[2].text(x[1]+0.15, lpips[1]+0.0, f"{lpips[1]:.4f}", fontsize=9, ha='center', va='bottom')
ax[2].text(x[2]-0.15, lpips[2]-0.0001, f"{lpips[2]:.4f}", fontsize=9, ha='center', va='bottom')

# Set the global title and legend for the figure
fig.suptitle('Metrics with Respect to Batch Size')

# Save the plot
path = Path("plots/plots")  # Ensure the correct path is used
path.mkdir(parents=True, exist_ok=True)
fig.savefig(path / "batch_size_plot.png", bbox_inches='tight')
fig.savefig(path / "batch_size_plot.pdf", bbox_inches='tight')

plt.show()
