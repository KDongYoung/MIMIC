import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(targets, preds, save_path, metric):

    plt.figure()
    cm = confusion_matrix(targets, preds)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v2}\n({v3})" for v2, v3 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{save_path}/confusion_matrix/{metric}.png")
    
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    
    for i in range(cm.shape[0]):  # Iterate over classes
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity_i = tn / (tn + fp)
        specificities.append(specificity_i)
    
    return np.mean(specificities) # , specificities