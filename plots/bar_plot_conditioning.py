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
model_names = ['Geometric', 'Depth-aware', 'Triplane']
psrns = [20.411, 20.429, 18.09]
lpips = [0.1576, 0.1556, 0.236]
ssim = [0.8439, 0.844, 0.828]

# Create three side-by-side plots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjusted to make plots side by side

# Bar width
width = 0.35  # Width of the bars

# First plot for PSNR
bars_psnr = axs[0].bar(model_names, psrns, color=PSNR_COLOR, width=width)
axs[0].set_title('PSNR')
axs[0].set_ylabel('PSNR')
axs[0].set_ylim(10, 21)  # Set y-axis to start at 10
for bar in bars_psnr:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

# Second plot for LPIPS
bars_lpips = axs[1].bar(model_names, lpips, color=SSIM_COLOR, width=width)
axs[1].set_title('LPIPS')
axs[1].set_ylabel('LPIPS')
axs[1].set_ylim(0.1, 0.25)  # Set y-a
for bar in bars_lpips:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Third plot for SSIM
bars_ssim = axs[2].bar(model_names, ssim, color=LPIPS_COLOR, width=width)
axs[2].set_title('SSIM')
axs[2].set_ylabel('SSIM')
axs[2].set_ylim(0.8,0.85)
for bar in bars_ssim:
    yval = bar.get_height()
    axs[2].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

# Set the global title for the figure
fig.suptitle('Metrics with Respect to Conditioning Type')

# Save the plot as PNG and PDF
path_png = Path("plots/plots")  # Directory for saving PNGs
path_pdf = Path("plots/plots")  # Directory for saving PDFs
path_png.mkdir(parents=True, exist_ok=True)
path_pdf.mkdir(parents=True, exist_ok=True)
fig.savefig(path_png / "cond_plot_bar.png", bbox_inches='tight')
fig.savefig(path_pdf / "cond_plot_bar.pdf", bbox_inches='tight')

plt.show()
