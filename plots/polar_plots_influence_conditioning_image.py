import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for m in ['Metric_polar_PSNR', 'Metric_polar_LPIPS', 'Metric_polar_SSIM', 'Metric_polar_Loss']:
    # Load data
    data1 = np.load(f'plots/plots/{m}_ctx.npz')['arr_0']
    data2 = np.load(f'plots/plots/{m}.npz')['arr_0']

    # Determine common intensity range
    vmin = min(np.min(data1), np.min(data2))
    vmax = max(np.max(data1), np.max(data2))

    # Angle (theta) setup
    theta = np.linspace(0, 2 * np.pi, data1.shape[0])

    # Radius (r) setup
    r = np.linspace(0, 1, data1.shape[1])

    # Create meshgrid for theta and r
    theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')

    # Create the figure and axes for two subplots
    # Adjust figsize to better suit the aspect ratio for polar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 4))

    if "PSNR" in m or "SSIM" in m:
        cmap = 'viridis'
    else:
        cmap = 'viridis_r'

    # First polar plot
    heatmap1 = ax1.contourf(theta_grid, r_grid, data1, 50, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax1.set_title(f'{m.split("_")[-1]} (Context Image Loss)')

    # Customize the ticks
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    radial_ticks = np.linspace(0, 1, 10) + 0.0001  # To avoid zero overlap
    selected_ticks = radial_ticks[1::2]  # Skip every second starting from the second element

    # Set the selected radial ticks
    ax1.set_rticks(selected_ticks)

    # Generate y-tick labels corresponding to the selected ticks
    yticks = [f'{int(tick * 90)}°' for tick in selected_ticks]

    # Set the y-tick labels
    ax1.set_yticklabels(yticks)

    # Optional: set the radial label position
    ax1.set_rlabel_position(0)

    # Second polar plot
    heatmap2 = ax2.contourf(theta_grid, r_grid, data2, 50, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax2.set_title(f'{m.split("_")[-1]} (Ambient Image Loss)')

    # Customize the ticks
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_rticks(selected_ticks)
    ax2.set_yticklabels(yticks)
    ax2.set_rlabel_position(0)

    # Adjust the spacing between the two plots
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8, bottom=0.2)  # Adjust whitespace around the plots

    # Adding a color bar shared by both plots, adjusting size with 'shrink'
    cbar = fig.colorbar(heatmap1, ax=[ax1, ax2], pad=0.1)#, shrink=0.75)
    cbar.set_label(m.split('_')[2])

    plt.savefig(f'plots/plots/{m}_combined.png',dpi=1200)  # Save the plot
    plt.savefig(f'plots/plots/{m}_combined.pdf')  # Save the plot
    plt.show()
    plt.close(fig)  # Ensure each figure is closed properly
