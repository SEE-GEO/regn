import numpy as np

def compute_roc(p,
                y_true,
                thresholds=np.linspace(0, 1, 11)):
    n_true = y_true.sum()
    tpr = np.zeros(thresholds.size)
    fpr = np.zeros(thresholds.size)
    y_true_not = np.logical_not(y_true)

    for i in range(thresholds.size):
        t = thresholds[i]
        tpr[i] = np.logical_and(y_true, p >= t).sum() / y_true.sum()
        fpr[i] = np.logical_and(y_true_not, p >= t).sum() / y_true_not.sum()
    return fpr, tpr




