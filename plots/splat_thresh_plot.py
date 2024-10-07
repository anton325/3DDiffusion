import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from unified_colors import PSNR_COLOR, SSIM_COLOR, LPIPS_COLOR


# Enable LaTeX rendering and set the font to be handled by LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Access the Pastel1 colormap
pastel1 = mpl.cm.get_cmap('Pastel1')

# Data setup
model_names = [0.05, 0.1, 0.3, 0.5]
psrns = [19.667, 19.502, 19.58, 19.89]
lpips = [0.147, 0.1456, 0.1511, 0.14]
ssim = [0.849, 0.859, 0.8598, 0.8513]

# Create three side-by-side plots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjust subplot layout to side by side

# Plot for PSNR
axs[0].plot(model_names, psrns, marker='o', color=PSNR_COLOR, label='PSNR')
axs[0].axhline(y=17.87, color='gray', linestyle='--')
axs[0].text(0.17, 17.87 + 0.05, "PSNR without threshold", color='gray', ha='left')
axs[0].set_xlabel('Noise Level Threshold')
axs[0].set_ylabel('PSNR')
axs[0].set_title('PSNR')
for i, v in enumerate(psrns):
    if i == 1 or i == 3:
        continue
    axs[0].text(model_names[i], v + 0.05, f"{v:.2f}", ha='center', va='bottom')
axs[0].text(model_names[1]+0.005, psrns[1] + 0.05, f"{psrns[1]:.2f}", ha='center', va='bottom')
axs[0].text(model_names[3]-0.0, psrns[3] - 0.13, f"{psrns[3]:.2f}", ha='center', va='bottom')

# Plot for SSIM
axs[1].plot(model_names, ssim, marker='s', color=SSIM_COLOR, label='SSIM')
axs[1].axhline(y=0.804, color='gray', linestyle='--')
axs[1].text(0.2, 0.806, "SSIM without threshold", color='gray', ha='left')
axs[1].set_xlabel('Noise Level Threshold')
axs[1].set_ylabel('SSIM')
axs[1].set_title('SSIM')
axs[1].set_ylim(0.8, 0.87)  # Set y-axis limits for better visibility
for i, v in enumerate(ssim):
    if i == 1:
        continue
    axs[1].text(model_names[i], v - 0.005, f"{v:.3f}", ha='center', va='bottom')
axs[1].text(model_names[1]+0.01, ssim[1] - 0.005, f"{ssim[1]:.3f}", ha='center', va='bottom')

# Plot for LPIPS
axs[2].plot(model_names, lpips, marker='^', color = LPIPS_COLOR, label='LPIPS')
axs[2].axhline(y=0.283, color='gray', linestyle='--')
axs[2].text(0.2, 0.283 - 0.01, "LPIPS without threshold", color='gray', ha='left')
axs[2].set_xlabel('Noise Level Threshold')
axs[2].set_ylabel('LPIPS')
axs[2].set_title('LPIPS')
for i, v in enumerate(lpips):
    axs[2].text(model_names[i], v + 0.005, f"{v:.4f}", ha='center', va='bottom')

# Set the global title for the figure
fig.suptitle('Metrics with Respect to Rendering Noise Level Threshold')

# Save the plot
path = Path("plots/plots")  # Ensure the correct path is used
path.mkdir(parents=True, exist_ok=True)
fig.savefig(path / "splat_tresh_plot.png", bbox_inches='tight')
fig.savefig(path / "splat_tresh_plot.pdf", bbox_inches='tight')

plt.show()
