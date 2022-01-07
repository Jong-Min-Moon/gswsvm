from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
x1 = np.random.uniform(0,1,5000)
x2 = np.random.uniform(0,1,5000)
x = np.c_[x1,x2]
y = [getY2d(x1, x2) for (x1,x2) in zip(x1,x2)]


np.unique(y, return_counts = True)


x1Grid, x2Grid = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
xGrid = np.c_[x1Grid.ravel(), x2Grid.ravel()]

fig = plt.figure(figsize = (10,10))

#fix gamma
two_n_lambda_range = np.array([[2**-(i + j * (1/5)) for j in range(5)] for i in range(-1, 3 + 1)]).ravel()
gamma_range = np.array([[2**(i + j * (1/5)) for j in range(5)] for i in range(5, 8 + 1)]).ravel()
gamma_fixed = gamma_range[5]


for i, param in enumerate(two_n_lambda_range):
    print(i)
    print(param)
    ax = fig.add_subplot(5, 5, i+1, projection='3d')
    model = SVC(random_state = 1, gamma = gamma_fixed, C = param)
    model.fit(x, y)
    yGrid = model.decision_function(xGrid)
    ax.plot_surface(x1Grid, x2Grid, yGrid.reshape(200,200))
plt.show()