from pandas import DataFrame
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import h5py

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

def plot_confusion_matrix(y_scores, y_true, thresholds):

        accuracies = []
        for th in thresholds:
            y_pred = (y_scores >= th).astype(int)
            acc = (y_pred == y_true).sum() / len(y_true)
            accuracies.append(acc)

        best_index = np.argmax(accuracies) 
        best_th = thresholds[best_index]
        best_acc = accuracies[best_index]
        y_pred = (y_scores >= best_th).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        name_true = ["No crack true", "Crack true"]
        name_pred = ["No crack pred", "Crack pred"]
        cm_df = DataFrame(cm, index=name_true, columns=name_pred)

        print("##############################################")
        print(f"Confusion Matrix for best accuracy {best_acc:.3f}:")
        print(cm_df)
        print("##############################################")

def plot_histogram(self):

    y_score = np.concatenate([t.cpu().numpy() for t in self.test_losses])
    y_true = np.concatenate([t.cpu().numpy() for t in self.test_labels])

    auc_score = roc_auc_score(y_true, y_score)
    if auc_score < 0.2:
        auc_score = 1. - auc_score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = fpr[np.argmax(tpr >= 0.95)]
    
    # Print confusion matrix
    plot_confusion_matrix(y_score, y_true, thresholds)

    # Separate ID and OOD samples
    y_id = y_score[np.where(y_true == 0)]
    y_ood = y_score[np.where(y_true != 0)]

    fig, axes= plt.subplots(2,1, figsize=(10, 10))
    
    # plot histograms of scores in same plot
    axes[0].hist(y_id, bins=50, alpha=0.5, label='In-distribution', density=True)
    axes[0].hist(y_ood, bins=50, alpha=0.5, label='Out-of-distribution', density=True)
    axes[0].legend()
    axes[0].set_title('Outlier Detection', fontsize = self.fs)
    axes[0].set_ylabel('Counts', fontsize = self.fs)
    axes[0].set_xlabel('Loss', fontsize = self.fs)

    # plot roc
    axes[1].plot(fpr, tpr)
    axes[1].set_title('ROC', fontsize = self.fs)
    axes[1].set_ylabel('True Positive Rate', fontsize = self.fs)
    axes[1].set_xlabel('False Positive Rate', fontsize = self.fs)
    axes[1].legend([f"AUC {auc_score:.2f}"], fontsize = 12)
    axes[1].set_box_aspect(1)

    axes[1].plot([0,1], [0,1], ls="--")

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    
    plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_hist_ROC.png")
    fig.savefig(plt_dir)
    
    plt.close()
    
    # Logging plot as figure to mlflow
    # if self.logger.__class__.__name__ == "MLFlowLogger":
    #     self.logger.experiment.log_artifact(local_path = self.image_dir,
    #                                         run_id=self.logger.run_id)
    # Remove image from folder (saved to logger)
    # os.remove(image_path)

def save_anomaly_maps(output_file_name, batch):
    if not os.path.exists(output_file_name):
        # Creating new h5 file
        create_h5f_reconstruct(output_file_name, batch)
    else:
        # Appending h5 file
        append_h5f_reconstruct(output_file_name, batch)
    return 0

def create_h5f_reconstruct(output_filename_full_h5, batch):    
    """
    Create and save h5 file to store crack and normal tiny patches in

    This function creates h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed
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
                            dtype='float')
        h5f.create_dataset('meas_capture/rgb',
                            data = batch[0][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float')
        h5f.create_dataset('reconstructed/height',
                            data = batch[1][:,3:],
                            maxshape = (None, 1, 512, 512),
                            dtype='float')
        h5f.create_dataset('reconstructed/rgb',
                            data = batch[1][:,:3],
                            maxshape = (None, 3, 512, 512),
                            dtype='float')
        h5f.create_dataset('extra/OOD',
                           data = batch[2],
                           maxshape= (None,),
                           dtype= 'float')
        # Close the Keyence file for reading and the Keyence file for writing
        h5f.close()   

def append_h5f_reconstruct(output_filename_full_h5, batch):
    """
    Open and append a h5 file to store crack and normal tiny patches in

    This function opens h5 files which are structured like:
        
        dataset_file.h5
        ├───meas_capture
        |   ├────height               (16-bit uint array size num_images x num_rows x num_cols)
        |   └────rgb                  (16-bit uint array size num_images x num_rows x num_cols x 3)
        ├───reconstructed
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
        r_rgbs      = hdf5['reconstructed/rgb']
        r_heights   = hdf5['reconstructed/height']
        OODs        = hdf5['extra/OOD']

        original_size = rgbs.shape[0]

        rgb         = batch[0][:,:3]
        height      = batch[0][:,3:]
        r_rgb       = batch[1][:,:3]
        r_height    = batch[1][:,3:]
        id          = batch[2]
 
        rgbs.resize(original_size + rgb.shape[0], axis=0)
        heights.resize(original_size + height.shape[0], axis=0)
        r_rgbs.resize(original_size + r_rgb.shape[0], axis=0)
        r_heights.resize(original_size + r_height.shape[0], axis=0)
        OODs.resize(original_size + id.shape[0], axis=0)

        rgbs[original_size:]        = rgb
        heights[original_size:]     = height
        r_rgbs[original_size:]      = r_rgb
        r_heights[original_size:]   = r_height

        OODs[original_size:]     = id

        # Close the Keyence file for reading and the Keyence file for writing
        hdf5.close()

def ssim_for_batch(batch, r_batch, win_size):
    batch   = batch.cpu().numpy()
    r_batch = r_batch.cpu().numpy()
    bs = batch.shape[0]
    ssim_batch_img = np.zeros_like(batch)
    for i in range(bs):
        for j in range(batch.shape[1]):
            ssim,  img_ssim = structural_similarity(batch[i,j], 
                                r_batch[i,j],
                                win_size=win_size,
                                data_range=1,
                                full=True)
            ssim_batch_img[i, j] = (img_ssim - img_ssim.max()) * -1
    
    return ssim_batch_img