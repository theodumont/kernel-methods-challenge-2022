import numpy as np
import torch
from scipy.spatial import distance_matrix


class GaussianKernel_old:
    """Gaussian kernel.

        K(x, y) = exp(-||x-y||^2/(2*sigma^2))
    """
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        return np.exp(-distance_matrix(X,Y)/(2*self.sigma**2))

class GaussianKernel:
    """Gaussian kernel using torch for faster computation.

        K(x, y) = exp(-||x-y||/(2*sigma^2))
    """
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(torch.exp(-torch.cdist(X_,Y_)/(2*self.sigma**2)))

class RBFKernel:
    """RBF kernel using torch for faster computation.

        K(x, y) = exp(-g||x-y||^2)
    """
    def __init__(self, g=1.):
        self.g = g
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(torch.exp(-torch.cdist(X_,Y_)**2/(2*self.sigma**2)))

class TStudentKernel:
    """T-Student kernel "Alternative Kernels for Image Recognition".

        K(x, y) = 1 / (1 + ||x - y||^p)
    """
    def __init__(self, p=2):
        self.p = p
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(1 / (1 + torch.cdist(X_,Y_)**self.p))


class LinearKernel:
    """Linear kernel.

        K(x, y) = <x, y>
    """
    def kernel(self,X,Y):
        return X @ Y.T

class PolynomialKernel:
    """Polynomial kernel.

        K(x, y) = (a<x, y> + b)^p
    """
    def __init__(self, a=1, b=0, p=2):
        self.a = a
        self.b = b
        self.p = p
    def kernel(self, X, Y):
        return (self.a * X @ Y.T + self.b) ** self.p

class Chi2Kernel:
    """
    Chi^2 kernel,
        K(x, y) = exp( -gamma * SUM_i (x_i - y_i)^2 / (x_i + y_i) )
    as defined in:
    "Local features and kernels for classification
     of texture and object categories: A comprehensive study"
    Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
    International Journal of Computer Vision 2007
    http://eprints.pascal-network.org/archive/00002309/01/Zhang06-IJCV.pdf
    """

    def __init__(self, gamma=1.):
        self._gamma = gamma

    def kernel(self, X, Y):

        if np.any(X < 0) or np.any(Y < 0):
            print('Chi^2 kernel requires data to be strictly positive!')

        kernel = np.zeros((X.shape[0], Y.shape[0]))

        for d in range(X.shape[1]):
            column_1 = X[:, d].reshape(-1, 1)
            column_2 = Y[:, d].reshape(-1, 1)
            kernel += (column_1 - column_2.T)**2 / (column_1 + column_2.T)

        return np.exp(-self._gamma * kernel)

class WaveletKernel:
    """http://see.xidian.edu.cn/faculty/zhangli/publications/WSVM.pdf"""
    def __init__(self, a=1):
        self.a = a
        self.h = lambda t: np.cos(1.75*t) * np.exp(-t**2/2)
    def kernel(self,X,Y):
        return np.prod(self.h((X-Y)/self.a))

class LogKernel:
    """Log kernel using torch for faster computation.

        K(x, y) = - log(1+||x-y||^d)
    """
    def __init__(self, d=1):
        self.d = d
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(- torch.log(1 + torch.cdist(X_,Y_)**self.d))

class GHIKernel:
    """Generalized Histogram Intersection kernel.

        K(x, y) = sum_i min(|x_i|^beta,|y_i|^beta|)

    http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
    """
    def __init__(self, beta=1.):
        self.beta = beta
    def kernel(self,X,Y):
        K = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            K += np.minimum(
                np.power(np.abs(X[:, d].reshape(-1, 1)), self.beta),
                np.power(np.abs(Y[:, d].reshape(-1, 1)), self.beta).T
            )
        return K