import numpy as np
from sklearn.mixture import GaussianMixture
from functions import *
import matplotlib.pyplot as plt

X, Y, W = generateCheckerBoardforBayes(n = 1000, pnRatio = 4/1, random_state = 4)


bic = []
for k in range(1,30):
    print(k)
    gm = GaussianMixture(n_components=k, random_state=0).fit(X)
    bic.append(gm.bic(X))

plt.plot(range(1,30), bic)
plt.show()

#print(gm.predict([[0, 0], [12, 3]]))
#print(gm.means_)
#print(gm.bic(X))
