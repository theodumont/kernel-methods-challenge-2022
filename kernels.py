import numpy as np
import torch
from scipy.spatial import distance_matrix


class GaussianKernel_old:
    """Gaussian kernel."""
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        return np.exp(-distance_matrix(X,Y)/(2*self.sigma**2))

class GaussianKernel:
    """Gaussian kernel using torch for faster computation."""
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        X_ = torch.tensor(X)
        Y_ = torch.tensor(Y)
        return np.array(torch.exp(-torch.cdist(X_,Y_)/(2*self.sigma**2)))

class LinearKernel:
    """Linear kernel."""
    def kernel(self,X,Y):
        return X @ Y.T