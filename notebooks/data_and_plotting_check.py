# %%
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import GaussianBlur

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
# %% Functions


# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto",
                                    batch_size = 32,
                                    rgb_transform = diffuser_normalize(),
                                    height_transform = diffuser_normalize_height_idv()
                                    )
lightning_data.setup()
test_loader = lightning_data.test_dataloader()
# %% Run for 1 batch
avg_diff_gray   = torch.zeros(7)
avg_diff_height = torch.zeros(7)
for i, (gray, height, id) in enumerate(test_loader):
    avg_diff_gray[i]    = gray.max() - gray.min()
    avg_diff_height[i]  = height.max() - height.min()
print(f"mean diff gray: {torch.mean(avg_diff_gray):.2f}")
print(f"mean diff height: {torch.mean(avg_diff_height):.2f}")

# %%

def visualize_reconstructs_1ch(x, plot_ids, fs=16):
        # Convert back to [0,1] for plotting
        # x = (x + 1) / 2
        # reconstruct = (reconstruct + 1) / 2

        # if self.DDPM_param.mode == "rgb":
        x_gray = rgb_to_grayscale(x)
        
        # reconstruct_gray = rgb_to_grayscale(reconstruct)

        Blur = GaussianBlur(kernel_size=9)
        reconstruct_gray = Blur(x_gray)

        error = (x_gray-reconstruct_gray)**2  
        
        img = [x_gray, reconstruct_gray, error]

        title = ["Original sample", "Reconstructed Sample", "Anomaly map"]

        fig, axes = plt.subplots(nrows=len(plot_ids), ncols=3, 
                                 width_ratios=[1.08,1,1.08], 
                                 figsize=(9, 3*len(plot_ids)))
        plt.subplots_adjust(wspace=0.2, hspace=-0.2)
        extent = [0,4,0,4]
        for i, id in enumerate(plot_ids):
            for j in range(3):
                if i == 0:
                     axes[i, j].set_title(title[j], fontsize=fs-1)
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
                     axes[i,j].text(-0.4, 0.5, f"Sample {id}", fontsize= fs, rotation=90, va="center", ha="center", transform=axes[i,j].transAxes)
                elif (i < len(plot_ids) - 1) & (j > 0):
                     axes[i,j].tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
                else:
                     axes[i,j].tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
                     
            

def visualize_reconstructs_2ch(x, plot_ids, fs=12):
        # Convert back to [0,1] for plotting
        # x = (x + 1) / 2
        # reconstruct = (reconstruct + 1) / 2
        Blur = GaussianBlur(kernel_size=9)
        reconstruct = Blur(x)

        # Calculate pixel-wise squared error per channel + normalize
        error_idv = (x - reconstruct)**2
        #error_idv = self.min_max_normalize(error_idv, dim=(2,3))

        # Calculate pixel-wise squared error combined + normalize
        # error_comb = self.reconstruction_loss(x, reconstruct, reduction=None)
        error_comb = error_idv[:,0] + error_idv[:,1] #self.min_max_normalize(error_comb, dim=(2,3))
        print(error_comb.shape)
        img = [x, reconstruct, error_idv, error_comb]

        extent = [0,4,0,4]
        for i in plot_ids:
            fig = plt.figure(constrained_layout=True, figsize=(15,7))
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
            ax1.set_title("Original sample", fontsize =fs)
            ax1.set_ylabel("Y [mm]")
            ax1.text(-0.3, 0.5, "Gray-scale", fontsize= fs, rotation=90, va="center", ha="center", transform=ax1.transAxes)
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im1, cax=cax1)

            im2 = ax2.imshow(img[1][i,0], extent=extent, vmin=0, vmax=1)
            ax2.set_yticks([0,1,2,3,4])
            ax2.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            ax2.set_title("Reconstructed sample", fontsize =fs)
            
            im3 = ax3.imshow(img[2][i,0], extent=extent, vmin=0)
            ax3.set_yticks([0,1,2,3,4])
            ax3.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
            ax3.set_title("Anomaly map individual", fontsize =fs)
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im3, cax=cax3)

            im4 = ax4.imshow(img[0][i,1], extent=extent, vmin=0, vmax=1)
            ax4.set_yticks([0,1,2,3,4])
            ax4.set_xlabel("X [mm]")
            ax4.set_ylabel("Y [mm]")
            ax4.text(-0.3, 0.5, "Height", fontsize= fs, rotation=90, va="center", ha="center", transform=ax4.transAxes)
            divider = make_axes_locatable(ax4)
            cax4 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im4, cax=cax4)

            im5 = ax5.imshow(reconstruct[i,1].cpu(), extent=extent)
            ax5.set_yticks([0,1,2,3,4])
            ax5.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax5.set_xlabel("X [mm]")

            im6 = ax6.imshow(img[2][i,1], extent=extent, vmin=0)
            ax6.set_yticks([0,1,2,3,4])
            ax6.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
            ax6.set_xlabel("X [mm]")
            divider = make_axes_locatable(ax6)
            cax6 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im6, cax=cax6)

            # Span whole column
            im7 = ax7.imshow(img[3][i], extent=extent, vmin=0)
            ax7.set_title("Anomaly map combined", fontsize =fs)
            ax7.set_yticks([0,1,2,3,4])
            ax7.set_xlabel("X [mm]")
            ax7.set_ylabel("Y [mm]")

x = torch.cat((gray,height), dim=1)
visualize_reconstructs_1ch(gray, [0,1,2,5,6])
# %%
