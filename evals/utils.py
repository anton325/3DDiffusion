import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatlines(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z):
    def plot(data, plot_title, filename, which_elevation):
        heatmap_np = np.array(data) # shape (z rotations, x rotations)
        fig,ax = plt.subplots()
        ax.plot(np.arange(0,total_rotation,step_z),heatmap_np[:,which_elevation[0]],label=f"{(which_elevation[0]) * step_x}")
        ax.plot(np.arange(0,total_rotation,step_z),heatmap_np[:,which_elevation[1]],label=f"{(which_elevation[1]) * step_x}")
        ax.plot(np.arange(0,total_rotation,step_z),heatmap_np[:,which_elevation[2]],label=f"{(which_elevation[2]) * step_x}")
        ax.set_title(f"{plot_title} w.r.t. angle")
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel(f"{plot_title}")
        ax.legend()
        fig.savefig(f"{filename}.png")

    plot(heatmap_ssims, "SSIM", "Metric_line_SSIM", [0, int(np.array(heatmap_ssims).shape[1]/2), 8])
    plot(heatmap_lpips, "LPIPS", "Metric_line_LPIPS", [0, int(np.array(heatmap_lpips).shape[1]/2), 8])
    plot(heatmap_psnrs, "PSNR", "Metric_line_PSNR", [0, int(np.array(heatmap_psnrs).shape[1]/2), 8])
    plot(heatmap_losses, "Loss", "Metric_line_Loss", [0, int(np.array(heatmap_losses).shape[1]/2), 8])

    plot(heatmap_ssims, "SSIM", "Metric_line_SSIM1", [1, 4, 7]) # bei 10er schritt: 10, 40 ,70
    plot(heatmap_lpips, "LPIPS", "Metric_line_LPIPS1", [1, 4, 7]) 
    plot(heatmap_psnrs, "PSNR", "Metric_line_PSNR1", [1, 4, 7])
    plot(heatmap_losses, "Loss", "Metric_line_Loss1", [1, 4, 7])

def _plot_polar_heatmap(heatmap, metric, filename):
        data = np.array(heatmap)
        np.savez(f'plots/plots/{filename}.npz', data)
        # Make the last point the same as the first to ensure continuity
        data = np.vstack([data, data[0, :]])
        # Angle (theta) setup, ensuring it wraps around by repeating the first angle at the end
        theta = np.linspace(0, 2 * np.pi, data.shape[0])
        # Radius (r) setup
        r = np.linspace(0, 1, data.shape[1])
        # Create meshgrid for theta and r
        theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')
        # Create the polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Create the heatmap
        # if "LPIPS" in metric or "Loss" in metric:
        #     cmap = 'viridis_r'
        # else:
        cmap = 'viridis'
        heatmap = ax.contourf(theta_grid, r_grid, data, 50, cmap=cmap, extend='both')
        # Add a color bar
        cbar = fig.colorbar(heatmap, pad=0.1)
        cbar.set_label(metric)

        # Set the title of the plot
        ax.set_title(f"{metric} w.r.t. Elevation and Azimuth of Evaluation Pose")

        # Customizing the plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # Clockwise
        # Customizing radial labels
        radial_ticks = np.linspace(0, 1, 10) +0.0001 # Creates radial ticks from 0 to 1
        ax.set_rticks(radial_ticks)  # Set radial positions
        yticks = [f'{str(int(tick * 90))+"°"}' for tick in radial_ticks]
        ax.set_yticklabels(yticks)  # Set tick labels to degrees
        ax.set_rlabel_position(0)  # Moves the radial labels to 135 degrees
        plt.savefig(f'plots/plots/{filename}.png')
     
def plot_polar_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses):
    _plot_polar_heatmap(heatmap_ssims, "SSIM", "Metric_polar_SSIM")
    _plot_polar_heatmap(heatmap_lpips, "LPIPS", "Metric_polar_LPIPS")
    _plot_polar_heatmap(heatmap_psnrs, "PSNR", "Metric_polar_PSNR")
    _plot_polar_heatmap(heatmap_losses, "Loss", "Metric_polar_Loss")

def plot_heatmaps(heatmap_ssims, heatmap_lpips, heatmap_psnrs, heatmap_losses, total_rotation, step_x, step_z):
    def plot(data, plot_title, filename):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax = sns.heatmap(data.T, linewidth=0.5, cbar=True)

        x_ticks = np.arange(0, data.shape[0])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in range(0, total_rotation, step_z)])
        ax.set_xlabel("Azimuth")

        y_ticks = np.arange(0, data.shape[1])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(reversed([str(i) for i in range(0, 90, step_x)]))
        ax.set_ylabel("Elevation")
        
        ax.set_title(f"{plot_title} w.r.t. Elevation and Azimuth of conditioning image")

        fig.savefig(f"{filename}.png")

    plot(np.array(heatmap_ssims), "SSIM", "Metric SSIM")
    plot(np.array(heatmap_lpips), "LPIPS", "Metric LPIPS")
    plot(np.array(heatmap_psnrs), "PSNR", "Metric PSNR")
    plot(np.array(heatmap_losses), "Loss", "Metric Loss")