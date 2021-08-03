import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_function(X, y, classifier, sample_weight, axis, title):
    # plot the decision function
    x1Min = np.min(X[:,0])
    x1Max = np.max(X[:,0])
    x2Min = np.min(X[:,1])
    x2Max = np.max(X[:,1])


    #plot decision hyperplane
    x1GridBoundary, x2GridBoundary = np.meshgrid(np.linspace(x1Min, x1Max, 1000), np.linspace(x2Min, x2Max, 1000))
    ZBoundary = classifier.decision_function(np.c_[x1GridBoundary.ravel(), x2GridBoundary.ravel()])
    ZBoundary = ZBoundary.reshape(x1GridBoundary.shape)
    idxNearZero = (ZBoundary>-3*1e-1)&(ZBoundary<3*1e-1)
    axis.scatter(
        x1GridBoundary[idxNearZero],
        x2GridBoundary[idxNearZero],
        c = 'red', marker = '.', s = 20
        )
    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(x1GridBoundary, x2GridBoundary, ZBoundary, alpha=0.75, cmap=plt.cm.bone)
    print("contour and hyperplane")

    #plot support vectors
    SV = classifier.support_vectors_
    alphas = classifier.dual_coef_.flatten()
    SVlabel = np.sign(alphas).astype(np.int64)
    #axis.scatter(
     #   SV[:, 0], SV[:, 1],
     #   c = SVlabel,
     #   s = 50 * np.abs(alphas), alpha=0.9, cmap=plt.cm.bone, edgecolors='black')
    
    #plot datapoint
    #axis.scatter(X[:, 0], X[:, 1], c=y, s=60 * sample_weight, alpha=0.6, cmap=plt.cm.bone, edgecolors='black', marker = 's')

    

    nSupVecPos = np.sum(SVlabel > 0)
    nSupVecNeg = np.sum(SVlabel < 0)

    print(f"# positive support vectors:{nSupVecPos}")
    print(f"# negative support vectors:{nSupVecNeg}")

    axis.axis('off')
    axis.set_title(title)


# we create 20 points
np.random.seed(0)
pnRatio = 100
nPos = 500

nNeg = nPos*pnRatio
wPos = 90/100
wNeg = 10/100
q1 = np.c_[
    np.random.uniform(100,200, nPos),
    np.random.uniform(100,200, nPos)
]
q2 = np.c_[
    np.random.uniform(100,200, nNeg),
    np.random.uniform(0,100, nNeg)
]
q3 = np.c_[
    np.random.uniform(0,100,nPos),
    np.random.uniform(0,100, nPos)
]
q4 = np.c_[
    np.random.uniform(0,100, nNeg),
    np.random.uniform(100,200, nNeg)
]

X = np.r_[q1, q2, q3, q4]
y = [1] * nPos + [-1] * nNeg + [1] * nPos + [-1] * nNeg

sample_weight = np.ones(len(X))

sample_weight[:nPos] *= wPos
sample_weight[nPos : nPos + nNeg] *= wNeg
sample_weight[nPos + nNeg : 2 * nPos + nNeg ] *= wPos
sample_weight[2 * nPos + nNeg : 2 * nPos + 2 * nNeg] *= wNeg

no_weight = np.ones(len(X))
no_weight *= 0.5

# for reference, first fit without sample weights

# fit the model
clf_weights = svm.SVC(gamma = 1 / 31.62**2, C = 2*1000)
clf_weights.fit(X, y, sample_weight=sample_weight)

clf_no_weights = svm.SVC(gamma = 1 / 31.62**2, C = 2*1000)
clf_no_weights.fit(X, y, sample_weight = no_weight)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(X, y, clf_no_weights, no_weight, axes[0],
                       "Constant weights")
plot_decision_function(X, y, clf_weights, sample_weight, axes[1],
                       "Modified weights")

plt.show()