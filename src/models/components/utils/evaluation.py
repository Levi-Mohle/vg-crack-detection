import sys
from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import os

def plot_loss(self, skip):
    """
    Plot the training and validation loss over epochs and save the plot as an image file.

    Args:
        skip (int): Number of initial epochs to skip for the validation loss plot.

    """
    # Generate a list of epoch numbers from 1 to the current epoch
    epochs = [i for i in range(1, self.current_epoch + 1)]
    
    # Plot the training loss for each epoch
    plt.plot(epochs, [t.cpu().numpy() for t in self.train_epoch_loss], marker='o', linestyle='-', label="Training")
    
    # Plot the validation loss for each epoch, skipping the initial epochs as specified
    plt.plot(epochs, [t.cpu().numpy() for t in self.val_epoch_loss][skip:], marker='o', linestyle='-', label="Validation")
    
    # Set the x-axis label
    plt.xlabel('Epochs', fontsize=self.fs)
    
    # Set the y-axis label
    plt.ylabel('Loss [-]', fontsize=self.fs)
    
    # Add a legend to the plot
    plt.legend()
    
    # Set the title of the plot
    plt.title('Training and Validation Loss', fontsize=self.fs)
    
    # Define the directory and filename for saving the plot
    plt_dir = os.path.join(self.image_dir, f"{self.current_epoch}_loss.png")
    
    # Save the plot as an image file
    plt.savefig(plt_dir)
    
    # Close the plot to free up memory
    plt.close()

def plot_classification_metrics(y_score, y_true, save_dir=None, fs=12):
    """
    Plot classification metrics including Precision-Recall curve and ROC curve, 
    and save the plots and metrics to files if a save directory is provided.

    Args:
        y_score (array-like): Predicted scores or probabilities.
        y_true (array-like): True binary labels.
        save_dir (str, optional): Directory to save the plots and metrics. Default is None.
        fs (int, optional): Font size for the plots. Default is 12.

    """
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_score)
    if auc_score < 0.2:
        auc_score = 1. - auc_score
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = fpr[np.argmax(tpr >= 0.95)]
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Create subplots for Precision-Recall and ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    
    # Plot Precision-Recall curve
    axes[0].plot(recall, precision)
    axes[0].set_title('Precision-Recall curve', fontsize=fs)
    axes[0].set_ylabel('Precision', fontsize=fs)
    axes[0].set_xlabel('Recall', fontsize=fs)
    axes[0].set_box_aspect(1)
    axes[0].plot([0, 1], [0.5, 0.5], ls="--")
    
    # Plot ROC curve
    axes[1].plot(fpr, tpr)
    axes[1].set_title('ROC', fontsize=fs)
    axes[1].set_ylabel('True Positive Rate', fontsize=fs)
    axes[1].set_xlabel('False Positive Rate', fontsize=fs)
    axes[1].legend([f"AUC {auc_score:.4f}"], fontsize=12)
    axes[1].set_box_aspect(1)
    axes[1].plot([0, 1], [0, 1], ls="--")
    
    # Adjust layout
    plt.tight_layout()
    
    if save_dir is not None:
        # Save the plots to the specified directory
        plt_dir = os.path.join(save_dir, "0_PR_ROC.png")
        fig.savefig(plt_dir)
        plt.close()
        
        # Save classification metrics to a text file
        save_loc = os.path.join(save_dir, "0_classification_metrics.txt")
        with open(save_loc, "w") as f:
            sys.stdout = f
            print_confusion_matrix(y_score, y_true, thresholds)
        sys.stdout = sys.__stdout__
    else:
        # Print classification metrics to the console
        print_confusion_matrix(y_score, y_true, thresholds)

def print_confusion_matrix(y_score, y_true, thresholds):
    """
    Calculate and print the confusion matrix and various classification metrics for the best F1-score.

    Args:
        y_score (array-like): Predicted scores or probabilities.
        y_true (array-like): True binary labels.
        thresholds (array-like): Thresholds for converting scores to binary predictions.

    """
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_score)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = fpr[np.argmax(tpr >= 0.95)]
    fpr95_threshold = thresholds[np.argmax(tpr >= 0.95)]
    
    # Append -inf to thresholds
    np.append(thresholds, -np.inf)
    
    # Initialize variables to track the best F1-score and corresponding threshold
    best_f1 = 0
    best_threshold = None
    accuracy = 0
    
    # Iterate over thresholds to find the best F1-score
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_y_pred = y_pred
            accuracy = np.mean(y_pred == y_true)
            best_threshold = th
            best_f1 = f1
    
    # Calculate confusion matrix for the best F1-score predictions
    cm = confusion_matrix(y_true, best_y_pred)
    name_true = ["No crack true", "Crack true"]
    name_pred = ["No crack pred", "Crack pred"]
    cm_df = DataFrame(cm, index=name_true, columns=name_pred)
    
    # Print the confusion matrix and various classification metrics
    print("##############################################")
    print(f"Confusion Matrix for best F1-score {best_f1:.4f}:")
    print(cm_df)
    print("")
    print(f"{'AUC:':<20}{auc_score:.4f}")
    print(f"{'FPR @ 95% Recall:':<20}{fpr95:.4f}")
    print(f"{'Threshold at FPR @ 95% Recall: ':<40}{fpr95_threshold:.1f}")
    print("")
    print(f"Given threshold value @ best F1-score: {best_threshold}")
    print(f"{'Accuracy:':<20}{accuracy:.4f}")
    print(f"{'Precision:':<20}{cm[1,1]/(cm[0,1]+cm[1,1]):.4f}")
    print(f"{'Recall:':<20}{cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    print(f"{'Misclassification:':<20}{cm[0,1]+cm[1,0]}")
    print("##############################################")

def classify_metrics(y_score, y_true, save_dir=None):
    """
    Calculate and print classification metrics, including the confusion matrix and AUC score. 
    Optionally save the metrics to a file.

    Args:
        y_score (array-like): Predicted scores or probabilities.
        y_true (array-like): True binary labels.
        save_dir (str, optional): Directory to save the classification metrics. Default is None.

    """
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_score)
    
    # Calculate ROC curve to obtain thresholds
    _, _, thresholds = roc_curve(y_true, y_score)
    
    if save_dir is not None:
        # Define the location to save the classification metrics
        save_loc = os.path.join(save_dir, "0_classification_metrics.txt")
        
        # Print confusion matrix and other metrics to a file
        with open(save_loc, "w") as f:
            sys.stdout = f
            print_confusion_matrix(y_score, y_true, thresholds)
            print(f"AUC score: {auc_score:.3f}")
            print(f"True labels: {y_true}")
            print(f"OOD scores: {y_score}")
        sys.stdout = sys.__stdout__
    else:
        # Print confusion matrix and other metrics to the console
        print_confusion_matrix(y_score, y_true, thresholds)
        print(f"AUC score: {auc_score:.3f}")
        print(f"True labels: {y_true}")
        print(f"OOD scores: {y_score}")

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_histogram(y_score, y_true, save_dir=None, fs=16):
    """
    Plot histograms of in-distribution (ID) and out-of-distribution (OOD) samples based on their scores, 
    and save the plot as an image file if a save directory is provided.

    Args:
        y_score (array-like): Predicted scores or probabilities.
        y_true (array-like): True binary labels.
        save_dir (str, optional): Directory to save the histogram plot. Default is None.
        fs (int, optional): Font size for the plot labels and title. Default is 16.

    """
    # Separate ID and OOD samples based on true labels
    y_id    = y_score[np.where(y_true == 0)]
    y_ood   = y_score[np.where(y_true != 0)]
    
    # Create a figure and axis for the histogram plot
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    # Plot histogram for in-distribution samples
    axes.hist(y_id, bins=50, alpha=0.5, label='In-distribution', density=True)
    
    # Plot histogram for out-of-distribution samples
    axes.hist(y_ood, bins=50, alpha=0.5, label='Out-of-distribution', density=True)
    
    # Add legend to the plot
    axes.legend()
    
    # Set the title of the plot
    axes.set_title('Outlier Detection', fontsize=fs)
    
    # Set the y-axis label
    axes.set_ylabel('Counts', fontsize=fs)
    
    # Set the x-axis label
    axes.set_xlabel('Loss', fontsize=fs)
    
    # Adjust layout to fit elements properly
    plt.tight_layout()
    
    if save_dir is not None:
        # Define the directory and filename for saving the plot
        plt_dir = os.path.join(save_dir, "0_histogram.png")
        
        # Save the plot as an image file
        fig.savefig(plt_dir)
        
        # Close the plot to free up memory
        plt.close()
    
    # Logging plot as figure to mlflow
    # if self.logger.__class__.__name__ == "MLFlowLogger":
    #     self.logger.experiment.log_artifact(local_path = self.image_dir,
    #                                         run_id=self.logger.run_id)
    # Remove image from folder (saved to logger)
    # os.remove(image_path)