import numpy as np
from scipy.spatial import distance_matrix

class GaussianKernel:
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        return np.exp(-distance_matrix(X,Y)/(2*self.sigma**2))

class LinearKernel:
    def kernel(self,X,Y):
        return X @ Y.T