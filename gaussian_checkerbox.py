import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import GridSearchCV

Xtrain, Ytrain, Wtrain = generateCheckerBoardforBayes(n = 1000, pnRatio = 4/1, random_state = 4)
print(Wtrain[0], Wtrain[-1])
Xvalid, Yvalin,_ = generateCheckerBoard(n = 3000, pnRatio = 1/1)
Xtest, Ytest,_ = generateCheckerBoard(n = 20000, pnRatio = 1/1)

from sklearn.utils.multiclass import type_of_target
print(type_of_target(Ytrain))

C_range = [2**i for i in range(-10, 10 + 1)]
gamma_range = [2**i for i in range(-10, 3 + 1)]
param_grid = [
    {'C': C_range, 'gamma': gamma_range}
    ]
    
#gs = GridSearchCV(estimator = SVC(random_state=1), param_grid=param_grid, scoring='accuracy', n_jobs = -1, verbose = True)
#gs = gs.fit(Xvalid, Yvalid)
#print(gs.best_score_)
#print(gs.best_params_)


#sample_weight = np.ones(len(X))

#sample_weight[:nPos] *= wPos
#sample_weight[nPos : nPos + nNeg] *= wNeg
#sample_weight[nPos + nNeg : 2 * nPos + nNeg ] *= wPos
#sample_weight[2 * nPos + nNeg : 2 * nPos + 2 * nNeg] *= wNeg

no_weight = np.ones(len(Xtrain))
no_weight *= 0.5

# for reference, first fit without sample weights

# fit the model
clf_no_weights = SVC(C = 0.0009765625, gamma = 0.001953125)
clf_no_weights.fit(Xtrain, Ytrain)

clf_weights = SVC(C = 0.125, gamma = 0.015625, class_weight = {-1:Wtrain[-1], 1: Wtrain[1]})
clf_weights.fit(Xtrain, Ytrain)

#fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#plot_decision_function(X, y, clf_no_weights, no_weight, axes[0],
#                       "Constant weights")
#plot_decision_function(X, y, clf_weights, no_weight, axes[1],
#                       "Modified weights")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plotBoundary(Xtrain, clf_no_weights, axes[0], "Constant weights")
plotBoundary(Xtrain, clf_weights, axes[1], "Modified weights")
plt.show()