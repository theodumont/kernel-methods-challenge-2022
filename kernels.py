"""Kernels implemented:
- Gaussian kernel
- RBF kernel
- TStudent kernel
- linear kernel
- polynomial kernel
- Chi2 kernel
- wavelet kernel
- log kernel
- GHI kernel
"""
import numpy as np
import torch


class GaussianKernel:
    """Gaussian kernel.

        K(x, y) = exp(-||x-y||/(2*sigma^2))
    """
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(torch.exp(-torch.cdist(X_,Y_)/(2*self.sigma**2)))

class RBFKernel:
    """RBF kernel.

        K(x, y) = exp(-g||x-y||^2)
    """
    def __init__(self, g=1.):
        self.g = g
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(torch.exp(-torch.cdist(X_,Y_)**2/(2*self.sigma**2)))

class TStudentKernel:
    """T-Student kernel.

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
    Chi^2 kernel.

        K(x, y) = exp( -gamma * sum_i (x_i - y_i)^2 / (x_i + y_i) )

    "Local features and kernels for classification of texture and object categories: A
    comprehensive study", Zhang et al, 2006
    http://eprints.pascal-network.org/archive/00002309/01/Zhang06-IJCV.pdf
    """

    def __init__(self, gamma=1.):
        self._gamma = gamma
    def kernel(self, X, Y):
        if np.any(X < 0) or np.any(Y < 0):
            print('Chi^2 kernel requires data to be strictly positive!')
        kernel = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            kernel += (X[:, d].reshape(-1, 1) - Y[:, d].reshape(-1, 1).T)**2 / (X[:, d].reshape(-1, 1) + Y[:, d].reshape(-1, 1).T)
        return np.exp(-self._gamma * kernel)

class WaveletKernel:
    """
    Wavelet kernel.

        K(x, y) = prod_i h( (x_i - y_i) / a)
        with h(t) = cos(1.75t) exp(-t^2/2)

    "Wavelet support vector machine", Zhang et al, 2004
    http://see.xidian.edu.cn/faculty/zhangli/publications/WSVM.pdf
    """
    def __init__(self, a=1):
        self.a = a
        self.h = lambda t: np.cos(1.75*t) * np.exp(-t**2/2)
    def kernel(self,X,Y):
        K = np.ones((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            K *= self.h( (X[:, d].reshape(-1, 1) - Y[:, d].reshape(-1, 1).T) / self.a )
        return K

class LogKernel:
    """Log kernel.

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

    "Generalized histogram intersection kernel for image recognition", Boughorbel et al, 2005
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