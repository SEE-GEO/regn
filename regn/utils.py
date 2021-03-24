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

SURFACE_TYPE_NAMES = [
    "Ocean",
    "Sea-Ice",
    "Vegetation 1",
    "Vegetation 2",
    "Vegetation 3",
    "Vegetation 4",
    "Vegetation 5",
    "Snow 1",
    "Snow 2",
    "Snow 3",
    "Snow 4",
    "Standing Water",
    "Land Coast",
    "Mixed land/ocean o. water",
    "Ocean or water Coast",
    "Sea-ice edge",
    "Mountain Rain",
    "Mountain Snow"
]

def surface_type_to_name(surface_index):
    return SURFACE_TYPE_NAMES[int(surface_index) - 1]

CHANNEL_NAMES = [
    "10.6 GHz, V",
    "10.6 GHz, H",
    "18.7 GHz, V",
    "18.7 GHz, H",
    "23 GHz, V",
    "Missing 1",
    "37 GHz, V",
    "37 GHz, H",
    "89 GHz, V",
    "89 GHz, H",
    "166 GHz, V",
    "166 GHz, H",
    "Missing 2",
    "183 +/- 3 GHz, V",
    "183 +/- 7 GHz, V",
]

def channel_to_name(index):
    return CHANNEL_NAMES[int(index)]




