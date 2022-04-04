import numpy as np
from tqdm import tqdm

class KRR:

    def __init__(self, kernel, class_id, lambd, verbose):
        self.kernel = kernel
        self.lambd = lambd
        self.class_id = class_id
        self.alpha = None
        self.verbose = verbose

    def fit(self, y, K):
        if self.verbose: print("Fitting...")
        n = len(y)
        # self.X = X
        self.y = (y == self.class_id)
        self.alpha = np.linalg.inv(K + self.lambd * n * np.eye(n)) @ self.y

    def predict(self, K):
        if self.verbose: print("Predicting...")
        pred = K @ self.alpha
        return pred

class MultiKRR:

    def __init__(self, kernel, lambd=1e-3, verbose=False):
        self.kernel = kernel
        self.lambd = lambd
        self.verbose = verbose
        self.KRR_list = [KRR(self.kernel, class_id, self.lambd, self.verbose) for class_id in range(10)]

    def fit(self, X, y):
        self.X = X
        K = self.kernel(X,X)
        for krr in tqdm(self.KRR_list, desc="Fitting"):
            krr.fit(y, K)

    def predict(self, x):
        K = self.kernel(x,self.X)
        pred = np.array([krr.predict(K) for krr in self.KRR_list]).T
        pred = np.argmax(pred, axis=1)
        return pred