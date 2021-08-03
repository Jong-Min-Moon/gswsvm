from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import gmean
from triprob import getY2d
from functions import *

#samples
np.random.seed(1)
n = 1000
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
x = np.c_[x1,x2]
y = [getY2d(x1, x2) for (x1,x2) in zip(x1,x2)]

fig, ax = plt.subplots(1,1, figsize = (6,6))
ax.scatter(x1, x2, c = y, cmap = 'bwr_r')

counts = np.unique(y, return_counts = True)
print(counts)
nMaj = counts[1][0]
nMin = counts[1][1]
wMaj = nMin/n
wMin = nMaj/n


C_range = [2**i for i in range(-10, 12 + 1)]
gamma_range = [2**i for i in range(-10, 3 + 1)]
param_grid = [
    {'C': C_range, 'gamma': gamma_range}
    ]

#standard svm
gridSearcherStan = GridSearchCV(estimator = SVC(random_state=1), param_grid=param_grid, n_jobs = -1, verbose = True,
    scoring = gmean.gmeanScorer
)
gridSearcherStan.fit(x, y)
print(gridSearcherStan.best_score_)
print(gridSearcherStan.best_params_)

#weighted svm

gridSearcherWeighted = GridSearchCV(estimator = SVC(random_state=1, class_weight = {-1:wMaj, 1:wMin}), param_grid=param_grid, n_jobs = -1, verbose = True,
    scoring = gmean.gmeanScorer
)
gridSearcherWeighted.fit(x, y)
print(gridSearcherWeighted.best_score_)
print(gridSearcherWeighted.best_params_)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plotDecisionFunction(x, gridSearcherStan.best_estimator_,  axes[0],
                      "Constant weights")
plotDecisionFunction(x, gridSearcherWeighted.best_estimator_, axes[1],
                      "Modified weights")
plt.show()


#testing
np.random.seed(1)
nTest = 3000
x1Test = np.random.uniform(0,1,nTest)
x2Test = np.random.uniform(0,1,nTest)
xTest = np.c_[x1Test,x2Test]
yTest = [getY2d(x1, x2) for (x1,x2) in zip(x1Test,x2Test)]

predStan = gridSearcherStan.best_estimator_.predict(xTest)
sens, spec, gm = gmean.gmeanReporter(predStan, yTest)
print(f"""
    standard
    sensitivity: {sens}
    specificity: {spec}
    g-mean: {gm}
    """)


predWeighted = gridSearcherWeighted.best_estimator_.predict(xTest)
sens, spec, gm = gmean.gmeanReporter(predWeighted, yTest)
print(f"""
    weighted
    sensitivity: {sens}
    specificity: {spec}
    g-mean: {gm}
    """)