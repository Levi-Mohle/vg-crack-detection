# %%
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto",
                                    batch_size = 32,
                                    rgb_transform = diffuser_to_grayscale_idv(),
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
def visualize_reconstructs_2ch(x, reconstruct, plot_ids, fs=12):
        # Convert back to [0,1] for plotting
        x = (x + 1) / 2
        reconstruct = (reconstruct + 1) / 2

        # Calculate pixel-wise squared error per channel + normalize
        error_idv = (x - reconstruct)**2
        #error_idv = self.min_max_normalize(error_idv, dim=(2,3))

        # Calculate pixel-wise squared error combined + normalize
        # error_comb = self.reconstruction_loss(x, reconstruct, reduction=None)
        error_comb = x[:,0] #self.min_max_normalize(error_comb, dim=(2,3))
        
        img = [x, reconstruct, error_idv, error_comb]

        for i in plot_ids:
            fig = plt.figure(constrained_layout=True, figsize=(15,7))
            gs = GridSpec(2, 4, figure=fig, width_ratios=[1.08,1,1.08,1.08], height_ratios=[1,1], hspace=0.05, wspace=0.1)
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
            im1 = ax1.imshow(img[0][i,0], vmin=0, vmax=1)
            ax1.set_title("Original sample", fontsize =fs)
            ax1.text(-0.1, 0.5, "Gray-scale", fontsize= fs, rotation=90, va="center", ha="center", transform=ax1.transAxes)
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im1, cax=cax1)

            im2 = ax2.imshow(img[1][i,0], vmin=0, vmax=1)
            ax2.set_title("Reconstructed sample", fontsize =fs)
            
            im3 = ax3.imshow(img[2][i,0], vmin=0, vmax=1)
            ax3.set_title("Anomaly map individual", fontsize =fs)
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im3, cax=cax3)

            im4 = ax4.imshow(img[0][i,1], vmin=0, vmax=1) 
            ax4.text(-0.1, 0.5, "Height", fontsize= fs, rotation=90, va="center", ha="center", transform=ax4.transAxes)
            divider = make_axes_locatable(ax4)
            cax4 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im4, cax=cax4)

            im5 = ax5.imshow(img[1][i,1], vmin=0, vmax=1)

            im6 = ax6.imshow(img[2][i,1], vmin=0, vmax=1)
            divider = make_axes_locatable(ax6)
            cax6 = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im6, cax=cax6)

            # Span whole column
            im7 = ax7.imshow(img[3][i,0], vmin=0, vmax=1)
            ax7.set_title("Anomaly map combined", fontsize =fs)

            for ax in axs:
                ax.axis("off")

visualize_reconstructs_2ch(gray, height, [0])
# %%
