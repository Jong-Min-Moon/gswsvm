import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import GridSearchCV

Xtrain, Ytrain, Wtrain  = generateCheckerBoard(n = 3000, pnRatio = 4/1)



C_range = [2**i for i in range(-10, 10 + 1)]
gamma_range = [2**i for i in range(-9, 3 + 1)]
param_grid = [
    {'C': C_range, 'gamma': gamma_range}
    ]

#tune for standard svm    
gs = GridSearchCV(estimator = SVC(random_state=1), param_grid=param_grid, scoring='fowlkes_mallows_score', n_jobs = -1, verbose = True)
gs = gs.fit(Xtrain, Ytrain)
print("standard")
print(gs.best_score_)
print(gs.best_params_)

#tune for w svm    
gs = GridSearchCV(estimator = SVC(random_state=1, class_weight = {-1:Wtrain[-1], 1: Wtrain[1]}), param_grid=param_grid, scoring='fowlkes_mallows_score', n_jobs = -1, verbose = True)
gs = gs.fit(Xtrain, Ytrain)
print("weighted")
print(gs.best_score_)
print(gs.best_params_)