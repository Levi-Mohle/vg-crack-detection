import sys
from pandas import DataFrame
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import os
import h5py
import torch
from datetime import datetime
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms.functional import rgb_to_grayscale
from skimage.filters import sobel
import skimage.morphology as morphology

def plot_loss(self, skip):

    epochs = [i for i in range(1, self.current_epoch + 1)]
    plt.plot(epochs, [t.cpu().numpy() for t in self.train_epoch_loss], marker='o', linestyle = '-', label = "Training")
    plt.plot(epochs, [t.cpu().numpy() for t in self.val_epoch_loss][skip:], marker='o', linestyle = '-', label = "Validation")
    plt.xlabel('Epochs', fontsize = self.fs)
    plt.ylabel('Loss [-]', fontsize = self.fs)
    plt.legend()
    plt.title('Training and Validation Loss', fontsize = self.fs)

    plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_loss.png")
    plt.savefig(plt_dir)
    plt.close()

def plot_loss_VQGAN(self, skip, fs=16):
    epochs = [i for i in range(1, self.current_epoch + 1)]
    
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].plot(epochs, [t.cpu().numpy() for t in self.train_epoch_aeloss], marker='o', linestyle = '-', label = "Training")
    axes[0].plot(epochs, [t.cpu().numpy() for t in self.val_epoch_aeloss][skip:], marker='o', linestyle = '-', label = "Validation")
    axes[0].set_xlabel('Epochs', fontsize = fs)
    axes[0].set_ylabel('Loss [-]', fontsize = fs)
    axes[0].legend()
    axes[0].set_title("Autoencoder Loss", fontsize = fs)

    axes[1].plot(epochs, [t.cpu().numpy() for t in self.train_epoch_discloss], marker='o', linestyle = '-', label = "Training")
    axes[1].plot(epochs, [t.cpu().numpy() for t in self.val_epoch_discloss][skip:], marker='o', linestyle = '-', label = "Validation")
    axes[1].set_xlabel('Epochs', fontsize = fs)
    axes[1].legend()
    axes[1].set_title('Discriminator Loss', fontsize = fs)

    axes[2].plot(epochs, [t.cpu().numpy() for t in self.train_epoch_recloss], marker='o', linestyle = '-', label = "Training")
    axes[2].plot(epochs, [t.cpu().numpy() for t in self.val_epoch_recloss][skip:], marker='o', linestyle = '-', label = "Validation")
    axes[2].set_xlabel('Epochs', fontsize = fs)
    axes[2].legend()
    axes[2].set_title('Reconstruction Loss', fontsize = fs)

    plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_VQGAN_loss.png")
    plt.savefig(plt_dir)
    plt.close()

def plot_classification_metrics(y_score, y_true, save_dir=None, fs=12):

    auc_score = roc_auc_score(y_true, y_score)
    if auc_score < 0.2:
        auc_score = 1. - auc_score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = fpr[np.argmax(tpr >= 0.95)]

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    fig, axes= plt.subplots(1,2, figsize=(10, 10))
    # plot precision recall
    axes[0].plot(recall, precision)
    axes[0].set_title('Precision-Recall curve', fontsize = fs)
    axes[0].set_ylabel('Precision', fontsize = fs)
    axes[0].set_xlabel('Recall', fontsize = fs)
    axes[0].set_box_aspect(1)

    axes[0].plot([0,1], [0.5,0.5], ls="--")

    # plot ROC
    axes[1].plot(fpr, tpr)
    axes[1].set_title('ROC', fontsize = fs)
    axes[1].set_ylabel('True Positive Rate', fontsize = fs)
    axes[1].set_xlabel('False Positive Rate', fontsize = fs)
    axes[1].legend([f"AUC {auc_score:.3f}"], fontsize = 12)
    axes[1].set_box_aspect(1)

    axes[1].plot([0,1], [0,1], ls="--")

    plt.tight_layout()

    if save_dir is not None:
        # time    = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        plt_dir = os.path.join(save_dir, "0_PR_ROC.png")
        fig.savefig(plt_dir)
        plt.close()

        save_loc = os.path.join(save_dir, "0_classification_metrics.txt")
        # Print confusion matrix
        with open(save_loc, "w") as f:
            sys.stdout = f
            print_confusion_matrix(y_score, y_true, thresholds)
        sys.stdout = sys.__stdout__
    else:
        print_confusion_matrix(y_score, y_true, thresholds)


def print_confusion_matrix(y_score, y_true, thresholds):

    auc_score   = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr95       = fpr[np.argmax(tpr >= 0.95)]

    np.append(thresholds, -np.inf)

    best_f1         = 0
    best_threshold  = None
    accuracy        = 0

    for th in thresholds:
        y_pred      = (y_score >= th).astype(int)
        f1          = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_y_pred     = y_pred
            accuracy        = np.mean(y_pred == y_true)
            best_threshold  = th
            best_f1         = f1

    cm = confusion_matrix(y_true, best_y_pred)
    name_true = ["No crack true", "Crack true"]
    name_pred = ["No crack pred", "Crack pred"]
    cm_df = DataFrame(cm, index=name_true, columns=name_pred)

    print("##############################################")
    print(f"Confusion Matrix for best f1-score {best_f1:.3f}:")
    print(cm_df)
    print("")
    print(f"{'AUC:':<20}{auc_score:.3f}")
    print(f"{'FPR @ 95% Recall:':<20}{fpr95:.3f}")
    print("")
    print(f"Given threshold value @ best f1-score: {best_threshold}")
    print(f"{'Accuracy:':<20}{accuracy:.3f}")
    print(f"{'Precision:':<20}{cm[1,1]/(cm[0,1]+cm[1,1]):.3f}")
    print(f"{'Recall:':<20}{cm[1,1]/(cm[1,0]+cm[1,1]):.3f}")
    print(f"{'Misclassification:':<20}{cm[0,1]+cm[1,0]}")
    print("##############################################")

def classify_metrics(y_score, y_true, save_dir=None):
    auc_score           = roc_auc_score(y_true, y_score)
    _, _, thresholds    = roc_curve(y_true, y_score)

    if save_dir is not None:
        save_loc = os.path.join(save_dir, "0_classification_metrics.txt")
        # Print confusion matrix
        with open(save_loc, "w") as f:
            sys.stdout = f
            print_confusion_matrix(y_score, y_true, thresholds)
            print(f"AUC score: {auc_score:.3f}")
            print(f"true labels: {y_true}")
            print(f"OOD scores: {y_score}")
        sys.stdout = sys.__stdout__
    else:
        print_confusion_matrix(y_score, y_true, thresholds)
        print(f"AUC score: {auc_score:.3f}")
        print(f"true labels: {y_true}")
        print(f"OOD scores: {y_score}")

def threshold_mover(y_score, y_true, step_backward=0):

    auc_score           = roc_auc_score(y_true, y_score)
    _, _, thresholds    = roc_curve(y_true, y_score)
    np.append(thresholds, -np.inf)

    best_accuracy = 0
    best_threshold = None

    for i, th in enumerate(thresholds):
        y_pred      = (y_score >= th).astype(int)
        accuracy    = np.mean(y_pred == y_true)

        if accuracy > best_accuracy:
            best_y_pred     = y_pred
            best_accuracy   = accuracy
            best_threshold  = th
            best_i          = i

    y_pred      = (y_score >= thresholds[best_i+step_backward]).astype(int)
    accuracy    = np.mean(y_pred == y_true)

    cm = confusion_matrix(y_true, y_pred)
    name_true = ["No crack true", "Crack true"]
    name_pred = ["No crack pred", "Crack pred"]
    cm_df = DataFrame(cm, index=name_true, columns=name_pred)

    print("##############################################")
    print(f"Confusion Matrix for best accuracy {accuracy:.3f}:")
    print(cm_df)
    print("")
    print(f"Given best threshold value: {thresholds[best_i+step_backward]}")
    print(f"AUC score: {auc_score:.3f}")
    print(f"Recall: {cm[1,1]/(cm[1,0]+cm[1,1])}")
    print("##############################################")   

def plot_histogram(y_score, y_true, save_dir=None, fs=16):

    # Separate ID and OOD samples
    y_id = y_score[np.where(y_true == 0)]
    y_ood = y_score[np.where(y_true != 0)]

    fig, axes= plt.subplots(1,1, figsize=(10,5))
    
    axes.hist(y_id, bins=50, alpha=0.5, label='In-distribution', density=True)
    axes.hist(y_ood, bins=50, alpha=0.5, label='Out-of-distribution', density=True)
    axes.legend()
    axes.set_title('Outlier Detection', fontsize = fs)
    axes.set_ylabel('Counts', fontsize = fs)
    axes.set_xlabel('Loss', fontsize = fs)

    plt.tight_layout()
    
    if save_dir is not None:
        # time    = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        plt_dir = os.path.join(save_dir, f"0_histogram.png")
        fig.savefig(plt_dir)
        plt.close()
    
    # Logging plot as figure to mlflow
    # if self.logger.__class__.__name__ == "MLFlowLogger":
    #     self.logger.experiment.log_artifact(local_path = self.image_dir,
    #                                         run_id=self.logger.run_id)
    # Remove image from folder (saved to logger)
    # os.remove(image_path)

def save_reconstructions_to_h5(output_file_name, batch, cfg=False):
    if not os.path.exists(output_file_name):
        # Creating new h5 file
        create_h5f_reconstruct(output_file_name, batch, cfg)
    else:
        # Appending h5 file
        append_h5f_reconstruct(output_file_name, batch, cfg)
    return 0

def create_h5f_reconstruct(output_filename_full_h5, batch, cfg):    
    """
    Create and save h5 file to store crack and normal tiny patches in

    This function creates h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_0
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_1
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)

    Args:
        output_filename_full_h5 (str): filename + location of h5 file you want to create and save
        rgb_cracks (torch.Tensor): rgb tiny patches containing cracks (N,3,height,width)
        height_cracks (torch.Tensor): height tiny patches containing cracks (N,1,height,width)
        rgb_normal (torch.Tensor): rgb tiny patches containing normal samples (N,3,height,width)
        height_normal (torch.Tensor): height tiny patches containing normal samples (N,1,height,width)

    Returns:
        
    """
    with h5py.File(output_filename_full_h5, 'w') as h5f:
        h5f.create_dataset('meas_capture/height',
                            data = batch[0][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
        h5f.create_dataset('meas_capture/rgb',
                            data = batch[0][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
        if cfg:
            # Reconstruction with label 0 ("non-cracks")
            h5f.create_dataset('reconstructed_0/height',
                            data = batch[1][0][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
            h5f.create_dataset('reconstructed_0/rgb',
                            data = batch[1][0][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
            # Reconstruction with label 1 ("cracks")
            h5f.create_dataset('reconstructed_1/height',
                            data = batch[1][1][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
            h5f.create_dataset('reconstructed_1/rgb',
                            data = batch[1][1][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
        else:
            h5f.create_dataset('reconstructed_0/height',
                            data = batch[1][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float32')
            h5f.create_dataset('reconstructed_0/rgb',
                            data = batch[1][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float32')
            
        h5f.create_dataset('extra/OOD',
                           data = batch[2],
                           maxshape= (None,),
                           dtype= 'uint8')
        # Close the Keyence file for reading and the Keyence file for writing
        h5f.close()   

def append_h5f_reconstruct(output_filename_full_h5, batch, cfg=False):
    """
    Open and append a h5 file to store crack and normal tiny patches in

    This function opens h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_0
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed_1
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        └───extra
            └────OOD                  (8-bit uint array size num_images)
    Args:
        output_filename_full_h5 (str): filename + location of h5 file you want to open and append
        rgb_cracks (torch.Tensor): rgb tiny patches containing cracks (N,3,height,width)
        height_cracks (torch.Tensor): height tiny patches containing cracks (N,1,height,width)
        rgb_normal (torch.Tensor): rgb tiny patches containing normal samples (N,3,height,width)
        height_normal (torch.Tensor): height tiny patches containing normal samples (N,1,height,width)

    Returns:
    """
    with h5py.File(output_filename_full_h5, 'a') as hdf5:
        rgbs        = hdf5['meas_capture/rgb']
        heights     = hdf5['meas_capture/height']
        r0_rgbs      = hdf5['reconstructed_0/rgb']
        r0_heights   = hdf5['reconstructed_0/height']
        OODs        = hdf5['extra/OOD']

        original_size = rgbs.shape[0]

        if cfg:
            r1_rgbs      = hdf5['reconstructed_1/rgb']
            r1_heights   = hdf5['reconstructed_1/height']

            r0_rgb      = batch[1][0][:,:3]
            r0_height   = batch[1][0][:,3:]
            r1_rgb      = batch[1][1][:,:3]
            r1_height   = batch[1][1][:,3:]

            r1_rgbs.resize(original_size + r1_rgb.shape[0], axis=0)
            r1_heights.resize(original_size + r1_height.shape[0], axis=0)

            r1_rgbs[original_size:]      = r1_rgb
            r1_heights[original_size:]   = r1_height
        else:
            r0_rgb      = batch[1][:,:3]
            r0_height   = batch[1][:,3:]

        rgb         = batch[0][:,:3]
        height      = batch[0][:,3:]
        id          = batch[2]
 
        rgbs.resize(original_size + rgb.shape[0], axis=0)
        heights.resize(original_size + height.shape[0], axis=0)
        r0_rgbs.resize(original_size + r0_rgb.shape[0], axis=0)
        r0_heights.resize(original_size + r0_height.shape[0], axis=0)
        OODs.resize(original_size + id.shape[0], axis=0)

        rgbs[original_size:]        = rgb
        heights[original_size:]     = height
        r0_rgbs[original_size:]      = r0_rgb
        r0_heights[original_size:]   = r0_height

        OODs[original_size:]     = id

        # Close the Keyence file for reading and the Keyence file for writing
        hdf5.close()

def ssim_for_batch(batch, r_batch, win_size=5):
    batch   = batch.cpu().numpy()
    r_batch = r_batch.cpu().numpy()
    bs = batch.shape[0]
    
    ssim_batch     = np.zeros((batch.shape[0],batch.shape[1]))
    ssim_batch_img = np.zeros_like(batch)
    for i in range(bs):
        for j in range(batch.shape[1]):
            ssim,  img_ssim = structural_similarity(batch[i,j], 
                                r_batch[i,j],
                                win_size=win_size,
                                data_range=1,
                                full=True)
            ssim_batch_img[i, j] = img_ssim * -1
            ssim_batch[i, j]     = np.sum(ssim_batch_img[i, j] > 0)
    
    return ssim_batch, ssim_batch_img

def to_gray_0_1(x):
     # Convert first 3 channels (rgb) to gray-scale
     x_gray = rgb_to_grayscale(x[:,:3])
     # Concatentate result with height channel
     x = torch.cat((x_gray, x[:,3:]), dim=1)
     # Normalize back to [0,1]
     x = (x+1)/2
     return x

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

def post_process_ssim(x0, ssim_img):
    """
    Given the input sample x0 and anomaly maps produced with SSIM,
    this function filters the anomaly maps of noise and non-crack 
    related artifacts. It derives an OOD-score from the filtered map.

    Args:
        x0 (2D tensor) : input sample. 2 channels contain grayscale
                            and height (Bx2xHxW)
        ssim_img (2D tensor) : reconstruction of x0 (Bx2xHxW)
        
    Returns:
        ano_maps (2D tensor) : filtered anomaly map (Bx1xHxW)
        ood_score (1D tensor) : out-of-distribution scores (Bx1)
    """
    # Create empty tensor for filtered ssim and anomaly maps
    ssim_filt = np.zeros_like(ssim_img)
    ano_maps  = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))
    # sobel_filt  = np.zeros((ssim_img.shape[0],ssim_img.shape[2],ssim_img.shape[3]))

    # Loop over images in batch and both channels. Necessary since
    # skimage has no batch processing
    for idx in range(ssim_img.shape[0]):

        # Sobel filter on height map
        # sobel_filt[idx] = sobel(x0[idx,1].cpu().numpy())
        # sobel_filt[idx] = (sobel_filt[idx] > .02).astype(int)
        for i in range(ssim_img.shape[1]):
            
            # Thresholding
            ssim_filt[idx,i] = (ssim_img[idx,i] > np.percentile(ssim_img[idx,i], q=95)).astype(int)
            
            # Morphology filters
            ssim_filt[idx,i] = morphology.binary_erosion(ssim_filt[idx,i])


        # Boolean masks: if pixel is present in ssim height, ssim rgb
        # and sobel filter, it is accounted as crack pixel  
        ano_maps[idx] = (
                        (ssim_filt[idx,0]   == 1) & 
                        (ssim_filt[idx,1]   == 1) 
                        # (sobel_filt[idx]    == 1)
                        ).astype(int)
        
        # Opening (Erosion + Dilation) to remove noise + connect shapes
        # ano_maps[idx] = morphology.binary_opening(ano_maps[idx])
    
    # Calculate OOD-score, based on total number of crack pixels
    ood_score = np.sum(ano_maps, axis=(1,2))
                
    return ano_maps, ood_score

def OOD_score(x0, x1, x2):
    """
    Given the original sample x0 and its reconstructions x1 and x2, 
    this function returns the filtered anomaly map and OOD-score to be
    used in classification. If comparison is made between x0 and x1 or x2,
    provide x1 = x0.

    Args:
        x0 (2D tensor) : input sample (Bx2xHxW)
        x1 (2D tensor) : reconstruction of x0 (Bx2xHxW)
        x2 (2D tensor) : reconstruction of x0 (Bx2xHxW)
        

    Returns:
        ano_maps (2D tensor) : filtered anomaly map (Bx1xHxW)
        ood_score (1D tensor) : out-of-distribution scores (Bx1)
    
    """
    # Obtain SSIM between x1 and x2
    _, ssim_img             = ssim_for_batch(x1, x2)
    # Calculate anomaly maps and OOD-score
    ano_maps, ood_score    = post_process_ssim(x0, ssim_img)
    return ano_maps, ood_score