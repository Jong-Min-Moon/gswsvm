import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import GridSearchCV

def plot_decision_function(X, y, classifier, sample_weight, axis, title):
    # plot the decision function
    x1Min = np.min(X[:,0]) 
    x1Max = np.max(X[:,0])
    x2Min = np.min(X[:,1])
    x2Max = np.max(X[:,1])

    #plot datapoint
   # axis.scatter(X[:, 0], X[:, 1], c=y, s=60 * np.array(sample_weight), alpha=0.6, cmap=plt.cm.bone, edgecolors='black', marker = 's')

    #plot support vectors
    SV = classifier.support_vectors_
    alphas = classifier.dual_coef_.flatten()
    #alphas_abs = np.abs(alphas)
    #alphas_normalized = (alphas_abs - np.min(alphas_abs)) / (np.max(alphas_abs) - np.min(alphas_abs))
    SVlabel = np.sign(alphas).astype(np.int64)
    #axis.scatter(
    #    SV[:, 0], SV[:, 1],
    #   c = SVlabel,
    #   s = 10 + 20 * alphas_normalized, alpha=0.9, cmap=plt.cm.bone, edgecolors='black')
    
    #plot decision hyperplane
    x1GridBoundary, x2GridBoundary = np.meshgrid(np.linspace(x1Min, x1Max, 1000), np.linspace(x2Min, x2Max, 1000))
    ZBoundary = classifier.decision_function(np.c_[x1GridBoundary.ravel(), x2GridBoundary.ravel()])
    ZBoundary = ZBoundary.reshape(x1GridBoundary.shape)
    #idxNearZero = (ZBoundary > -1e-2)&(ZBoundary < 1e-2)
    #axis.scatter(
    #    x1GridBoundary[idxNearZero],
    #    x2GridBoundary[idxNearZero],
    #    c = 'red', marker = '.', s = 10
    #    )

    #color positive and negative area
    idxPositive = (ZBoundary > 0)
    idxNegative = (ZBoundary < 0)
    axis.scatter(
        x1GridBoundary[idxPositive],
        x2GridBoundary[idxPositive],
        c = 'blue', marker = '.', s = 10
        )
    axis.scatter(
        x1GridBoundary[idxNegative],
        x2GridBoundary[idxNegative],
        c = 'red', marker = '.', s = 10
        )

    # plot the line, the points, and the nearest vectors to the plane
    #axis.contourf(x1GridBoundary, x2GridBoundary, ZBoundary, alpha=0.75, cmap=plt.cm.bone)
    print("contour and hyperplane")




    

    nSupVecPos = np.sum(SVlabel > 0)
    nSupVecNeg = np.sum(SVlabel < 0)

    print(f"# positive support vectors:{nSupVecPos}")
    print(f"# negative support vectors:{nSupVecNeg}")

    axis.axis('off')
    axis.set_title(title)


Xtrain, Ytrain, Wtrain = generateCheckerBoard(n = 3000, pnRatio = 4/1)
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
plot_decision_function(Xtrain, Ytrain, clf_no_weights, no_weight, axes[0],
                      "Constant weights")
plot_decision_function(Xtrain, Ytrain, clf_weights, Wtrain, axes[1],
                      "Modified weights")
plt.show()