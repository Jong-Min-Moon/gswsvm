from sklearn.base import ClassifierMixin
from sklearn import svm
from sklearn.mixture import GaussianMixture

def _setWeight(self, X, Y):
    Xpos = X[Y == 1, :]
    nMinOG = Xpos.shape[0]
    nMinSyn = 2 * nMinOG
    normalizingConstant = (nMinOG * nMinSyn) + (nMinSyn * nMaj) + (nMinOG * nMaj)
    wMaj = nMinOG * nMinSyn / normalizingConstant
    wMinOG = nMinSyn * nMaj / normalizingConstant
    wMinSyn = nMinOG * nMaj / normalizingConstant #합이 1이 되게 normalize했으므로, cv에서 이 weight 그대로 써도 됨.

    W_GMC = [wMinOG] * nMinOG + [wMaj] * nMaj + [wMinSyn] * nMinSyn

class smoteWeightSVM(ClassifierMixin):
    def __init__(self, random_state, C, gamma, sample_weight):
        self.random_state = random_state
        self.C = C
        self.gamma = gamma
        self.sample_weight = sample_weight
        self.baseSVM = svm.SVC(random_state = self.random_state, C = self.C, gamma = self.gamma)

    def fit(self, X, y):
        X_GMC, Y_GMC = self._smote(X, y)
        self.baseSVM.fit(X_GMC, Y_GMC, sample_weight = self.sample_weight)
    
    def predict(self, X):
        return self.baseSVM.predict(X)

    def _smote(self, X, Y):
        Xpos = X[Y == 1, :]
        nMinOG = Xpos.shape[0]
        nMinSyn = 2 * nMinOG
        
        gm = GaussianMixture(n_components = 6, random_state=0).fit(Xpos)
        gmm_sample = gm.sample(nMinSyn)[0]
        X_GMC = np.r_[Xtrain, gmm_sample]
        Y_GMC = np.r_[Ytrain, [1] * nMinSyn]

        return X_GMC, Y_GMC
    
