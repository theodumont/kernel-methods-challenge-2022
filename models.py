"""Models implemented:
- One-vs-rest KRR (Kernel Ridge Regression)
- One-vs-rest KSVC (Kernel Support Vector Classifier)
- Kernel PCA
"""
import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh
from tqdm import tqdm
from scipy import optimize

# KRR ====================================================================================

class KRR:
    """Kernel Ridge Regression."""

    def __init__(self, kernel, class_id, lambd, verbose):
        self.kernel = kernel
        self.lambd = lambd
        self.class_id = class_id
        self.alpha = None
        self.verbose = verbose

    def fit(self, y, K):
        if self.verbose: print("Fitting...")
        n = len(y)
        self.y = (y == self.class_id)
        self.alpha = np.linalg.solve(K + self.lambd * n * np.eye(n), self.y)

    def predict(self, K):
        if self.verbose: print("Predicting...")
        pred = K @ self.alpha
        return pred

class OneVsRestKRR:
    """One-vs-rest KRR."""

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


# KSVC ===================================================================================

class KSVC:
    """Kernel Support Vector Classifier."""

    def __init__(self, kernel, C, class_id, epsilon, verbose):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.verbose = verbose
        self.class_id = class_id

    def fit(self, X, y, K):
        if self.verbose: print("Fitting...")
        N = len(y)
        self.X = X
        self.y = (y == self.class_id)

        # Lagrange dual problem
        def loss(alpha):
            return 1/2 * alpha.T @ np.diag(y) @ K @ np.diag(y) @ alpha - alpha.sum()
        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return np.diag(y) @ K @ np.diag(y) @ alpha - 1

        fun_eq = lambda alpha: - alpha.T @ y
        jac_eq = lambda alpha: - y
        fun_ineq = lambda alpha: self.C - alpha
        jac_ineq = lambda alpha: - np.eye(N)
        fun_ineq_pos = lambda alpha: alpha
        jac_ineq_pos = lambda alpha: np.eye(N)

        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq , 'jac': jac_ineq},
                       {'type': 'ineq', 'fun': fun_ineq_pos, 'jac': jac_ineq_pos})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        self.alpha = optRes.x

        self.supportIndices = np.where((self.alpha > self.epsilon) & (self.alpha < self.C*np.ones(N) - self.epsilon))
        self.support = X[self.supportIndices]
        self.b = (self.y - self.alpha.T @ np.diag(self.y) @ K)[self.supportIndices].mean()
        self.norm_f = self.alpha.T @ np.diag(self.y) @ K @ np.diag(self.y) @ self.alpha

    def predict(self, K):
        if self.verbose: print("Predicting...")
        d = self.alpha.T @ np.diag(self.y) @ K
        return d + self.b

class OneVsRestKSVC:
    """One-vs-rest KSVC."""

    def __init__(self, kernel, C, epsilon=1e-3, verbose=False):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.verbose = verbose
        self.KSVC_list = [KSVC(self.kernel, self.C, self.epsilon, class_id, self.verbose) for class_id in range(10)]

    def fit(self, X, y):
        self.X = X
        K = self.kernel(X,X)
        for ksvc in tqdm(self.KSVC_list, desc="Fitting"):
            ksvc.fit(X, y, K)

    def predict(self, x):
        K = self.kernel(self.X, x)
        pred = np.array([ksvc.predict(K) for ksvc in self.KSVC_list]).T
        pred = np.argmax(pred, axis=1)
        return pred


# KPCA ===================================================================================

def kernel_pca(K, n_components):
    """Following slide 219/1013 of the course."""
    # 1. center Gram matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    # 2. compute the first eigenvectors (u_i, \Delta_i)
    Delta, u = eigh(K)  # ascending order
    Delta, u = Delta[::-1], u[:,::-1]  # descending order
    Delta, u = Delta[:n_components], u[:,:n_components]  # first eigs
    # 3. normalize the eigenvectors \alpha_i = u_i / \sqrt{\Delta_i}
    alpha = u / Delta
    # 4. return projections K @ alpha
    Xtr_proj = K @ alpha
    return Xtr_proj