import numpy as np

#2d
def triPosterior2d(x1, x2):
    return (1 - np.abs(1 - 2*x1)) * (1 - np.abs(1 - 2*x2))
    
def getY2d(x1, x2):
    prob1 = triPosterior2d(x1, x2)
    randomNumber = np.random.uniform(0,1)
    if randomNumber>=prob1:
        return -1
    if randomNumber < prob1:
        return 1