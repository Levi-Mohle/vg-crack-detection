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

# def threshold_mover(y_score, y_true, step_backward=0):

#     auc_score           = roc_auc_score(y_true, y_score)
#     _, _, thresholds    = roc_curve(y_true, y_score)
#     np.append(thresholds, -np.inf)

#     best_accuracy = 0
#     best_threshold = None

#     for i, th in enumerate(thresholds):
#         y_pred      = (y_score >= th).astype(int)
#         accuracy    = np.mean(y_pred == y_true)

#         if accuracy > best_accuracy:
#             best_y_pred     = y_pred
#             best_accuracy   = accuracy
#             best_threshold  = th
#             best_i          = i

#     y_pred      = (y_score >= thresholds[best_i+step_backward]).astype(int)
#     accuracy    = np.mean(y_pred == y_true)

#     cm = confusion_matrix(y_true, y_pred)
#     name_true = ["No crack true", "Crack true"]
#     name_pred = ["No crack pred", "Crack pred"]
#     cm_df = DataFrame(cm, index=name_true, columns=name_pred)

#     print("##############################################")
#     print(f"Confusion Matrix for best accuracy {accuracy:.3f}:")
#     print(cm_df)
#     print("")
#     print(f"Given best threshold value: {thresholds[best_i+step_backward]}")
#     print(f"AUC score: {auc_score:.3f}")
#     print(f"Recall: {cm[1,1]/(cm[1,0]+cm[1,1])}")
#     print("##############################################")   

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