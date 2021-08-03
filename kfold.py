import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])




print(skf)

def gswsvmKfoldCV(X, y, n_splits = 5):
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 0)
    skf.get_n_splits(X, y)

    for trainIdx, validIdx in skf.split(X, y):
        X_train, X_valid = X[trainIdx], X[validIdx]
        y_train, y_valid = y[trainIdx], y[validIdx]

        #apply gmc-smote to 

        