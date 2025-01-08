from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os

def plot_loss(self):

    epochs = [i for i in range(1, self.current_epoch + 1)]
    plt.plot(epochs, [t.cpu().numpy() for t in self.train_epoch_loss], marker='o', linestyle = '-', label = "Training")
    plt.plot(epochs, [t.cpu().numpy() for t in self.val_epoch_loss][1:], marker='o', linestyle = '-', label = "Validation")
    plt.xlabel('Epochs', fontsize = self.fs)
    plt.ylabel('Loss [-]', fontsize = self.fs)
    plt.legend()
    plt.title('Training and Validation Loss', fontsize = self.fs)

    plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_loss.png")
    plt.savefig(plt_dir)
    plt.close()

def plot_loss_VQGAN(self):
    epochs = [i for i in range(1, self.current_epoch + 1)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].plot(epochs, [t.cpu().numpy() for t in self.train_epoch_aeloss], marker='o', linestyle = '-', label = "Training")
    axes[0].plot(epochs, [t.cpu().numpy() for t in self.val_epoch_aeloss][1:], marker='o', linestyle = '-', label = "Validation")
    axes[0].set_xlabel('Epochs', fontsize = self.fs)
    axes[0].set_ylabel('Loss [-]', fontsize = self.fs)
    axes[0].legend()
    axes[0].set_title('Training and Validation Autoencoder Loss', fontsize = self.fs)

    axes[1].plot(epochs, [t.cpu().numpy() for t in self.train_epoch_discloss], marker='o', linestyle = '-', label = "Training")
    axes[1].plot(epochs, [t.cpu().numpy() for t in self.val_epoch_discloss][1:], marker='o', linestyle = '-', label = "Validation")
    axes[1].set_xlabel('Epochs', fontsize = self.fs)
    axes[1].legend()
    axes[1].set_title('Training and Validation Discriminator Loss', fontsize = self.fs)

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