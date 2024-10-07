import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Define the metrics and titles for the plots
metrics = ['Metric_polar_PSNR', 'Metric_polar_SSIM', 'Metric_polar_LPIPS']
titles = ['PSNR', 'SSIM', 'LPIPS']

# Create the figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 18), sharex='col')

for i, metric in enumerate(metrics):
    # Load both datasets for each metric to find global min and max
    data_ctx = np.load(f'plots/plots/{metric}_ctx.npz')['arr_0']
    data_ref = np.load(f'plots/plots/{metric}.npz')['arr_0']

    # Determine global intensity range for the row
    vmin = min(np.min(data_ctx), np.min(data_ref))
    vmax = max(np.max(data_ctx), np.max(data_ref))

    for j in range(2):
        # Select data based on column
        data = data_ctx if j == 0 else data_ref

        # Angle (theta) and radius (r) setup
        theta = np.linspace(0, 2 * np.pi, data.shape[0])
        r = np.linspace(0, 1, data.shape[1])
        theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')

        # Select color map based on metric, invert for LPIPS
        cmap = 'viridis_r' if metric == 'Metric_polar_LPIPS' else 'viridis'

        # Plotting the data
        heatmap = axs[i, j].contourf(theta_grid, r_grid, data, 50, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        axs[i, j].set_title(f"{titles[i]} ({'Context Image Loss' if j == 0 else 'Ambient Image Loss'})")

        # Customize the ticks
        axs[i, j].set_theta_zero_location('N')
        axs[i, j].set_theta_direction(-1)
        radial_ticks = np.linspace(0, 1, 10) + 0.0001  # To avoid zero overlap
        selected_ticks = radial_ticks[1::2]

        # Set the selected radial ticks
        axs[i, j].set_rticks(selected_ticks)

        # Generate y-tick labels corresponding to the selected ticks
        yticks = [f'{int(tick * 90)}°' for tick in selected_ticks]

        # Set the y-tick labels
        axs[i, j].set_yticklabels(yticks)
        axs[i, j].set_rlabel_position(0)

    # Adding a shared color bar for the row outside the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.68 - 0.33 * i, 0.02, 0.25])  # Adjust the positioning based on your layout
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label(titles[i])

# Set the overarching title for the entire figure
fig.suptitle('Comparison of Metric Evaluations', fontsize=16)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the right margin to make space for the color bar

# Save and show the plot
plt.savefig('plots/plots/metric_evaluation_corrected.png')  # Save the plot as PNG
plt.savefig('plots/plots/metric_evaluation_corrected.pdf')  # Save the plot as PDF
plt.show()
