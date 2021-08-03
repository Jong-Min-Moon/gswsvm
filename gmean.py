import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

def gmean(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return np.sqrt(sensitivity * specificity)

gmeanScorer = make_scorer(gmean, greater_is_better=True)

def gmeanReporter(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return sensitivity, specificity, np.sqrt(sensitivity * specificity)