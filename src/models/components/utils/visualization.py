import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms.functional import rgb_to_grayscale
from src.models.components.utils.post_process import to_gray_0_1, ssim_for_batch

def min_max_normalize(self, x, dim=(0,2,3)):
    min_val = x.amin(dim=dim, keepdim=True)
    max_val = x.amax(dim=dim, keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)
        
def class_reconstructs_2ch(self, x, reconstructs, target, plot_ids, ood=None, fs=12):
    x = to_gray_0_1(x).cpu()
    # x = self.min_max_normalize(x, dim=(2,3)).cpu()
    
    ssim_orig_vs_reconstruct = []
    for i, reconstruct in enumerate(reconstructs):
        reconstructs[i] = to_gray_0_1(reconstruct).cpu()
        # reconstructs[i] = self.min_max_normalize(reconstruct, dim=(2,3)).cpu()

        # Calculate SSIM between original sample and all reconstructed labels
        _, ssim_img = ssim_for_batch(x, reconstructs[i], self.win_size)
        ssim_orig_vs_reconstruct.append(ssim_img) # (ssim_img > -0.1).astype(int)
        
    _, ssim_l0_vs_l1 = ssim_for_batch(reconstructs[0], reconstructs[1], self.win_size)

    extent = [0,4,0,4]
    for i in plot_ids:
        fig = plt.figure(constrained_layout=False, figsize=(15,17))
        fig.suptitle(f"OOD-score is: {ood[i]}")
        gs = GridSpec(4, 4, figure=fig, width_ratios=[1.08,1,1.08,1.08], height_ratios=[1,1,1,1], hspace=0.2, wspace=0.2)
        
        # RGB images
        # Span whole column
        ax1 = fig.add_subplot(gs[0:2,0])
        ax6 = fig.add_subplot(gs[0:2,3])

        # Regular grid
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[0,2])
        ax5 = fig.add_subplot(gs[1,2])

        # Height images
        # Span whole column
        ax7 = fig.add_subplot(gs[2:4,0])
        ax12 = fig.add_subplot(gs[2:4,3])

        # Regular grid
        ax8  = fig.add_subplot(gs[2,1])
        ax9  = fig.add_subplot(gs[3,1])
        ax10 = fig.add_subplot(gs[2,2])
        ax11 = fig.add_subplot(gs[3,2])

        # Plot rgb
        im1 = ax1.imshow(x[i,0], extent=extent, vmin=0, vmax=1)
        ax1.set_yticks([0,1,2,3,4])
        # ax1.tick_params(axis='both', which='both', labelbottom=False, labelleft=True)
        ax1.set_title("Original sample", fontsize =fs)
        ax1.set_ylabel("Y [mm]")
        ax1.set_xlabel("X [mm]")
        ax1.text(-0.3, 0.5, f"Gray-scale {target[i]}", fontsize= fs*2, rotation=90, va="center", ha="center", transform=ax1.transAxes)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax1)

        for j, ax in enumerate([ax2, ax3]):
            ax.imshow(reconstructs[j][i,0], extent=extent, vmin=0, vmax=1)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax.set_title(f"Reconstructed sample label {j}", fontsize =fs)

        for j, ax in enumerate([ax4, ax5]):
            im = ax.imshow(ssim_orig_vs_reconstruct[j][i,0], extent=extent)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_title(f"SSIM label {j} recon vs orig", fontsize =fs)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        im6 = ax6.imshow(ssim_l0_vs_l1[i,0], extent=extent, vmin=0)
        ax6.set_yticks([0,1,2,3,4])
        ax6.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
        ax6.set_xlabel("X [mm]")
        ax6.set_title(f"SSIM label 0 vs label 1 recon", fontsize =fs)
        divider = make_axes_locatable(ax6)
        cax6 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im6, cax=cax6)

        # Plot height
        im7 = ax7.imshow(x[i,1], extent=extent, vmin=0, vmax=1)
        ax7.set_yticks([0,1,2,3,4])
        ax7.set_title("Original sample", fontsize =fs)
        ax7.set_ylabel("Y [mm]")
        ax7.set_xlabel("X [mm]")
        ax7.text(-0.3, 0.5, f"Height {target[i]}", fontsize= fs*2, rotation=90, va="center", ha="center", transform=ax7.transAxes)
        divider = make_axes_locatable(ax7)
        cax7 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im7, cax=cax7)

        for j, ax in enumerate([ax8, ax9]):
            ax.imshow(reconstructs[j][i,1], extent=extent, vmin=0, vmax=1)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax.set_title(f"Reconstructed sample label {j}", fontsize =fs)

        for j, ax in enumerate([ax10, ax11]):
            im = ax.imshow(ssim_orig_vs_reconstruct[j][i,1], extent=extent)
            ax.set_yticks([0,1,2,3,4])
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_title(f"SSIM label {j} recon vs orig", fontsize =fs)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

        im12 = ax12.imshow(ssim_l0_vs_l1[i,1], extent=extent)
        ax12.set_yticks([0,1,2,3,4])
        ax12.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
        ax12.set_xlabel("X [mm]")
        ax12.set_title(f"SSIM label 0 vs label 1 recon", fontsize =fs)
        divider = make_axes_locatable(ax12)
        cax12 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im12, cax=cax12)

        plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs_{i}_target_{target[i]}.png")
        fig.savefig(plt_dir)
        plt.close()

def visualize_reconstructs_2ch(self, x, reconstruct, target, plot_ids, ood=None):

        x           = to_gray_0_1(x).cpu()
        reconstruct = to_gray_0_1(reconstruct).cpu()
            
        # Calculate pixel-wise squared error per channel + normalize

        _, error_idv = ssim_for_batch(x, reconstruct, self.win_size)
        # error_idv = self.min_max_normalize(error_idv, dim=(2,3))

        # Calculate pixel-wise squared error combined + normalize
        error_comb = self.reconstruction_loss(x, reconstruct, reduction=None).cpu()
        # error_comb = self.min_max_normalize(error_comb, dim=(2,3))
        
        if ood is None:
            ood = [None] * len(plot_ids)

        img = [self.min_max_normalize(x, dim=(2,3)).cpu(), self.min_max_normalize(reconstruct, dim=(2,3)).cpu(), error_idv, error_comb]
        extent = [0,4,0,4]
        for i in plot_ids:
            fig = plt.figure(constrained_layout=True, figsize=(15,7))
            fig.suptitle(f"OOD-score is: {ood[i]} | True label is: {target[i]}")
            gs = GridSpec(2, 4, figure=fig, width_ratios=[1.08,1,1.08,1.08], height_ratios=[1,1], hspace=0.05, wspace=0.2)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[0,2])
            ax4 = fig.add_subplot(gs[1,0])
            ax5 = fig.add_subplot(gs[1,1])
            ax6 = fig.add_subplot(gs[1,2])
            # Span whole column
            ax7 = fig.add_subplot(gs[:,3])
            axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            # Plot
            im1 = ax1.imshow(img[0][i,0], extent=extent, vmin=0, vmax=1)
            ax1.set_yticks([0,1,2,3,4])
            ax1.tick_params(axis='both', which='both', labelbottom=False, labelleft=True)
            ax1.set_title("Original sample", fontsize =self.fs)
            ax1.set_ylabel("Y [mm]")
            ax1.text(-0.3, 0.5, "Gray-scale", fontsize= self.fs, rotation=90, va="center", ha="center", transform=ax1.transAxes)
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im1, cax=cax1)

            im2 = ax2.imshow(img[1][i,0], extent=extent, vmin=0, vmax=1)
            ax2.set_yticks([0,1,2,3,4])
            ax2.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            ax2.set_title("Reconstructed sample", fontsize =self.fs)
            
            im3 = ax3.imshow(img[2][i,0], extent=extent)
            ax3.set_yticks([0,1,2,3,4])
            ax3.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            ax3.set_title("Anomaly map individual", fontsize =self.fs)
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im3, cax=cax3)

            im4 = ax4.imshow(img[0][i,1], extent=extent, vmin=0, vmax=1)
            ax4.set_yticks([0,1,2,3,4])
            ax4.set_xlabel("X [mm]")
            ax4.set_ylabel("Y [mm]")
            ax4.text(-0.3, 0.5, "Height", fontsize= self.fs, rotation=90, va="center", ha="center", transform=ax4.transAxes)
            divider = make_axes_locatable(ax4)
            cax4 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im4, cax=cax4)

            im5 = ax5.imshow(reconstruct[i,1].cpu(), extent=extent)
            ax5.set_yticks([0,1,2,3,4])
            ax5.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax5.set_xlabel("X [mm]")

            im6 = ax6.imshow(img[2][i,1], extent=extent)
            ax6.set_yticks([0,1,2,3,4])
            ax6.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax6.set_xlabel("X [mm]")
            divider = make_axes_locatable(ax6)
            cax6 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im6, cax=cax6)

            # Span whole column
            im7 = ax7.imshow(img[3][i,0], extent=extent, vmin=0)
            ax7.set_title("Anomaly map combined", fontsize =self.fs)
            ax7.set_yticks([0,1,2,3,4])
            ax7.set_xlabel("X [mm]")
            ax7.set_ylabel("Y [mm]")

            # for ax in axs:
            #     ax.axis("off")

            plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs_{i}.png")
            fig.savefig(plt_dir)
            plt.close()
            # Send figure as artifact to logger
            # if self.logger.__class__.__name__ == "MLFlowLogger":
            #     self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)

def visualize_reconstructs_1ch(self, x, reconstruct, plot_ids):
        # Convert back to [0,1] for plotting
        x = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        # Convert rgb to grayscale for plotting
        if self.mode == 'rgb':
            x              = rgb_to_grayscale(x)
            reconstruct    = rgb_to_grayscale(reconstruct)
            
        # Calculate pixel-wise squared error per channel + normalize
        error = ((x - reconstruct)**2)

        img = [x.cpu(), reconstruct.cpu(), error.cpu()]

        title = ["Original sample", "Reconstructed Sample", "Anomaly map"]

        fig, axes = plt.subplots(nrows=len(plot_ids), ncols=3, 
                                 width_ratios=[1.08,1,1.08], 
                                 figsize=(9, 3*len(plot_ids)))
        
        plt.subplots_adjust(wspace=0.2, hspace=-0.2)
        extent = [0,4,0,4]
        for i, id in enumerate(plot_ids):
            for j in range(3):
                if i == 0:
                     axes[i, j].set_title(title[j], fontsize=self.fs-1)
                # plot images
                if j == 2:
                     im = axes[i, j].imshow(img[j][i,0], extent=extent, vmin=0)
                else:
                    im = axes[i, j].imshow(img[j][i,0], extent=extent, vmin=0, vmax=1)
                # plot colorbars
                if j != 1:
                    divider = make_axes_locatable(axes[i,j])
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    plt.colorbar(im, cax=cax)
                if i == len(plot_ids) - 1:
                     axes[i,j].set_xlabel("X [mm]")
                else:
                     axes[i,j].tick_params(axis='both', which='both', labelbottom=False, labelleft=True)

                if j == 0:
                     axes[i,j].set_ylabel("Y [mm]")
                     axes[i,j].text(-0.4, 0.5, f"Sample {id}", fontsize= self.fs, rotation=90, va="center", ha="center", transform=axes[i,j].transAxes)
                elif (i < len(plot_ids) - 1) & (j > 0):
                     axes[i,j].tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
                else:
                     axes[i,j].tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
                
                        
        plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_reconstructs.png")
        fig.savefig(plt_dir)
        plt.close()
        # Send figure as artifact to logger
        # if self.logger.__class__.__name__ == "MLFlowLogger":
        #     self.logger.experiment.log_artifact(local_path=plt_dir, run_id=self.logger.run_id)