import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

def hungarian_evaluate(head, predictions, class_names, compute_confusion_matrix=False, confusion_matrix_file=None):
    # Combine predictions from all heads
    all_predictions = torch.cat([head['predictions'] for head in predictions], dim=1)
    targets = predictions[0]['targets']
    # calculate accuracy
    correct = torch.sum(all_predictions == targets.view(-1, 1), dim=0)
    accuracy = correct.float() / targets.size(0)
    # calculate the confusion matrix
    if compute_confusion_matrix:
        # Compute the confusion matrix as a pandas DataFrame
        confusion = confusion_matrix(targets.cpu().numpy(), all_predictions.cpu().numpy())
        confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_df, annot=True, fmt='d')
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.title('Confusion Matrix')
        if confusion_matrix_file is not None:
            plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
    return accuracy.tolist()
