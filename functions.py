import numpy as np
import matplotlib.pyplot as plt




def plotBoundary(X, classifier, axis, title):

    x1Min = np.min(X[:,0])
    x1Max = np.max(X[:,0])
    x2Min = np.min(X[:,1])
    x2Max = np.max(X[:,1])

    #plot datapoint
   # axis.scatter(X[:, 0], X[:, 1], c=y, s=60 * np.array(sample_weight), alpha=0.6, cmap=plt.cm.bone, edgecolors='black', marker = 's')



    #color positive and negative area
    x1GridBoundary, x2GridBoundary = np.meshgrid(np.linspace(x1Min, x1Max, 1000), np.linspace(x2Min, x2Max, 1000))
    ZBoundary = classifier.decision_function(np.c_[x1GridBoundary.ravel(), x2GridBoundary.ravel()])
    ZBoundary = ZBoundary.reshape(x1GridBoundary.shape)

    idxPositive = (ZBoundary > 0)
    idxNegative = (ZBoundary < 0)
    axis.scatter(
        x1GridBoundary[idxPositive],
        x2GridBoundary[idxPositive],
        c = 'lightsteelblue', marker = '.', s = 10
        )
    axis.scatter(
        x1GridBoundary[idxNegative],
        x2GridBoundary[idxNegative],
        c = 'lightpink', marker = '.', s = 10
        )
    
    #plot support vectors
    SV = classifier.support_vectors_
    alphas = classifier.dual_coef_.flatten()
    #alphas_abs = np.abs(alphas)
    #alphas_normalized = (alphas_abs - np.min(alphas_abs)) / (np.max(alphas_abs) - np.min(alphas_abs))
    SVlabel = np.sign(alphas).astype(np.int64)
    axis.scatter(
        SV[:, 0], SV[:, 1], c = SVlabel, cmap = 'bwr_r',
        #s = 10 + 20 * alphas_normalized,
        alpha=0.9,
        edgecolors='black')

    nSupVecPos = np.sum(SVlabel > 0)
    nSupVecNeg = np.sum(SVlabel < 0)

    print(f"# positive support vectors:{nSupVecPos}")
    print(f"# negative support vectors:{nSupVecNeg}")

    axis.axis('off')
    axis.set_title(title)



def generateCheckerBoard(seedNumber = 1, n = 100, pnRatio = 4/1):
    mu_maj_1 = [5,5]
    mu_maj_2 = [15,5]
    mu_maj_3 = [10,10]
    mu_maj_4 = [20,10]
    mu_maj_5 = [5,15]
    mu_maj_6 = [15,15]
    mu_maj = [mu_maj_1, mu_maj_2, mu_maj_3, mu_maj_4, mu_maj_5, mu_maj_6]
    cov_maj = [[2, 0], [0, 2]]

    mu_min_1 = [10,5]
    mu_min_2 = [20,5]
    mu_min_3 = [5,10]
    mu_min_4 = [15,10]
    mu_min_5 = [10,15]
    mu_min_6 = [20,15]
    mu_min = [mu_min_1, mu_min_2, mu_min_3, mu_min_4, mu_min_5, mu_min_6]
    cov_min = [[2.5, 0], [0, 2.5]]

    np.random.seed(seedNumber)
    nMin = int(n / (pnRatio + 1))
    nMaj = n - nMin

    #generate minority
    membership = np.random.randint(6, size = nMin)
    indices, frequency = np.unique(membership, return_counts =True)
    Xmin = [np.random.multivariate_normal(mu_min[i], cov_min, frequency[i]) for i in indices]
    Xmin = np.vstack(Xmin)

    #generate majority
    membership = np.random.randint(6, size = nMaj)
    indices, frequency = np.unique(membership, return_counts =True)
    Xmaj = [np.random.multivariate_normal(mu_maj[i], cov_maj, frequency[i]) for i in indices]
    Xmaj = np.vstack(Xmaj)

    X = np.r_[Xmin, Xmaj]
    y = [1] * nMin + [-1] * nMaj
    w = [n/nMin] * nMin + [n/nMaj] * nMaj
    print(nMin, nMaj)

    return X, y, w

def generateCheckerBoardforBayes(random_state = 1, n = 100, pnRatio = 4/1):
    mu_maj_1 = [5,5]
    mu_maj_2 = [15,5]
    mu_maj_3 = [10,10]
    mu_maj_4 = [20,10]
    mu_maj_5 = [5,15]
    mu_maj_6 = [15,15]
    mu_maj = [mu_maj_1, mu_maj_2, mu_maj_3, mu_maj_4, mu_maj_5, mu_maj_6]
    cov_maj = [[2, 0], [0, 2]]

    mu_min_1 = [10,5]
    mu_min_2 = [20,5]
    mu_min_3 = [5,10]
    mu_min_4 = [15,10]
    mu_min_5 = [10,15]
    mu_min_6 = [20,15]
    mu_min = [mu_min_1, mu_min_2, mu_min_3, mu_min_4, mu_min_5, mu_min_6]
    cov_min = [[2.5, 0], [0, 2.5]]

    np.random.seed(random_state)

    y = np.random.uniform(0,1,n) < 1 / (1 + pnRatio)
    nMin = sum(y)
    nMaj = len(y) - nMin

    #generate minority
    membership = np.random.randint(6, size = nMin)
    indices, frequency = np.unique(membership, return_counts =True)
    Xmin = [np.random.multivariate_normal(mu_min[i], cov_min, frequency[i]) for i in indices]
    Xmin = np.vstack(Xmin)

    #generate majority
    membership = np.random.randint(6, size = nMaj)
    indices, frequency = np.unique(membership, return_counts =True)
    Xmaj = [np.random.multivariate_normal(mu_maj[i], cov_maj, frequency[i]) for i in indices]
    Xmaj = np.vstack(Xmaj)

    X = np.r_[Xmin, Xmaj]
    y = [1] * nMin + [-1] * nMaj
    w = [n/nMin] * nMin + [n/nMaj] * nMaj
    print(nMin, nMaj)

    return X, y, w

    
def generateUniformChecker(seedNumber = 1, n = 100, pnRatio = 4/1):
    np.random.seed(seedNumber)
    nMin = int(n / (pnRatio + 1))
    nMaj = n - nMin

    #generate minority
    min1 = np.c_[
        np.random.uniform(0,100, int(int(nMin/4))),
        np.random.uniform(0,100, int(nMin/4))
    ]

    min2 = np.c_[
        np.random.uniform(200,300, int(nMin/4)),
        np.random.uniform(0,100, int(nMin/4))
    ]
    
    min3 = np.c_[
        np.random.uniform(0,100, int(nMin/4)),
        np.random.uniform(200,300, int(nMin/4))
    ]

    min4 = np.c_[
        np.random.uniform(200,300, int(nMin/4)),
        np.random.uniform(200,300, int(nMin/4))
    ]

    #generate majority
    maj1 = np.c_[
        np.random.uniform(100,200, int(nMaj/5)),
        np.random.uniform(0,100, int(nMaj/5))
    ]

    maj2 = np.c_[
        np.random.uniform(0,300, int(3*nMaj/5)),
        np.random.uniform(100,200, int(3*nMaj/5))
    ]
    
    maj3 = np.c_[
        np.random.uniform(100,200, int(nMaj/5)),
        np.random.uniform(200,300, int(nMaj/5))
    ]

    Xmin = np.r_[min1, min2, min3, min4]
    Xmaj = np.r_[maj1, maj2, maj3]
    nMinReal = Xmin.shape[0]
    nMajReal = Xmaj.shape[0]

    X = np.r_[Xmin, Xmaj]
    y = [1] * nMinReal + [-1] * nMajReal
    w = [n/nMin] * nMinReal + [n/nMaj] * nMajReal

    return X, y, w


import matplotlib.pyplot as plt

#X, y, w = generateUniformChecker(n=3000, pnRatio = 10/1)
#fig, ax = plt.subplots(1,1, figsize = (10,10))
#ax.scatter(X[:,0], X[:,1], c = y)
#plt.show()