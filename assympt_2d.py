from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import gmean

#2d
def triDensity2d(x1, x2):
    return (1 - np.abs(1 - 2*x1)) * (1 - np.abs(1 - 2*x2))
    
def getY2d(x1, x2):
    prob1 = triDensity2d(x1, x2)
    randomNumber = np.random.uniform(0,1)
    if randomNumber>=prob1:
        return -1
    if randomNumber < prob1:
        return 1

#samples
x1 = np.random.uniform(0,1,1000)
x2 = np.random.uniform(0,1,1000)
x = np.c_[x1,x2]
y = [getY2d(x1, x2) for (x1,x2) in zip(x1,x2)]

fig, ax = plt.subplots(1,1, figsize = (6,6))
ax.scatter(x1, x2, c = y)
plt.show()

np.unique(y, return_counts = True)

#nonstandard
x1Grid, x2Grid = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
xGrid = np.c_[x1Grid.ravel(), x2Grid.ravel()]

fig = plt.figure(figsize = (10,10))

two_n_lambda_range = [2**-i for i in range(-10, 10 + 1)]
gamma_range = np.array([[2**(i + j * (1/5)) for j in range(5)] for i in range(5, 8 + 1)]).ravel()
param_grid = [
    {'C': two_n_lambda_range}
    ]

for i in range(5*4):
    ax = fig.add_subplot(5, 4, i+1, projection='3d')
    gamma_now = gamma_range[i]
    gs = GridSearchCV(
        estimator = SVC(random_state=1, gamma = gamma_now),
        param_grid=param_grid,
        scoring=gmean.gmeanScorer,
        n_jobs = -1, refit = True)
    gs.fit(x, y)
    print(gs.best_params_)
    print(gs.best_score_)
    yGrid = gs.best_estimator_.decision_function(xGrid)
    ax.plot_surface(x1Grid, x2Grid, yGrid.reshape(200,200))
plt.show()



