import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_function(X, y, classifier, sample_weight, axis, title):
    # plot the decision function
    x1Min = np.min(X[:,0])
    x1Max = np.max(X[:,0])
    x2Min = np.min(X[:,1])
    x2Max = np.max(X[:,1])
    xx, yy = np.meshgrid(np.linspace(x1Min, x1Max, 500), np.linspace(x2Min, x2Max, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)

    #plot support vectors
    SV = classifier.support_vectors_
    alphas = classifier.dual_coef_.flatten()
    SVlabel = np.sign(alphas).astype(np.int64)
    axis.scatter(
        SV[:, 0], SV[:, 1],
        c = SVlabel,
        s = 50 * np.abs(alphas), alpha=0.9, cmap=plt.cm.bone, edgecolors='black')
    
    #plot datapoint
    axis.scatter(X[:, 0], X[:, 1], c=y, s=60 * sample_weight, alpha=0.6, cmap=plt.cm.bone, edgecolors='black', marker = 's')

    nSupVecPos = np.sum(SVlabel > 0)
    nSupVecNeg = np.sum(SVlabel < 0)

    print(f"# positive support vectors:{nSupVecPos}")
    print(f"# negative support vectors:{nSupVecNeg}")

    axis.axis('off')
    axis.set_title(title)


# we create 20 points
np.random.seed(0)
X = np.r_[
    np.random.multivariate_normal([3.5,3.5], [[2,0], [0,2]], 50),
    np.random.multivariate_normal([1,1], [[2,0], [0,2]], 500)
    ]
y = [1] * 50 + [-1] * 500

sample_weight = np.ones(len(X))

sample_weight[:51] *= 0.95
sample_weight[51:] *= 0.05

no_weight = np.ones(len(X))
no_weight *=0.5

# for reference, first fit without sample weights

# fit the model
clf_weights = svm.SVC(gamma=1, C=10)
clf_weights.fit(X, y, sample_weight=sample_weight)

clf_no_weights = svm.SVC(gamma=1, C=10)
clf_no_weights.fit(X, y, sample_weight = no_weight)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(X, y, clf_no_weights, no_weight, axes[0],
                       "Constant weights")
plot_decision_function(X, y, clf_weights, sample_weight, axes[1],
                       "Modified weights")

plt.show()