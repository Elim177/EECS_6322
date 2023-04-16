import numpy as np
import matplotlib.pyplot as plt

# for the confusion matrix
def confusion_matrix(predictions, ground_truth, class_names, output_file=None):
    # calculate the conf_matrix
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum((predictions == j) & (ground_truth == i))
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    #this is the plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    # Set ticks and ticklabels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    # add the colument names
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, '{:.2f}%'.format(cm[i, j]*100),
                    ha="center", va="center", color="blue" if cm[i, j] > thresh else "white")
    #this will be to get the proper layout
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
