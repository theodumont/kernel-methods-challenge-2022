"""Implement HOG (Histogram of Oriented Gradients) using numpy."""
import numpy as np


def gradient(image):
    """Compute gradient of image while preserving the shape."""
    grad_y = np.zeros_like(image)
    grad_x = np.zeros_like(image)
    # borders
    grad_y[0,:] = 0
    grad_x[:,0] = 0
    grad_y[-1,:] = 0
    grad_x[:,-1] = 0
    # gradient
    grad_y[1:-1,:] = image[2:,:] - image[:-2,:]
    grad_x[:,1:-1] = image[:,2:] - image[:,:-2]
    return grad_x, grad_y

def select_biggest_magnitude(grad_x, grad_y):
    """For an 3-channel image, return the gradients that have the biggest magnitude along
    the channel axis."""
    magnitude = grad_magnitude(grad_x, grad_y)
    max_magnitude_idx = magnitude.argmax(axis=2)
    yy, xx = np.meshgrid(np.arange(grad_x.shape[0]), np.arange(grad_x.shape[1]), indexing='ij', sparse=True)
    grad_x, grad_y = grad_x[yy,xx,max_magnitude_idx], grad_y[yy,xx,max_magnitude_idx]
    return grad_x, grad_y

def grad_magnitude(grad_x, grad_y):
    """Return magnitude of the gradients."""
    return np.sqrt(grad_x ** 2 + grad_y ** 2)

def grad_orientation(grad_x, grad_y, eps=1e-8):
    """Return orientation of the gradients."""
    return np.rad2deg(np.arctan(grad_y/(grad_x+eps))) % 180

def cell_histogram(magnitude, orientation, orient_min, orient_max):
    """Compute the value of the histogram for one orientation range for a cell of the image.

    Parameters
    ----------
    magnitude : magnitude of the cell.
    orientation : orientations of the cell.
    orient_min, orient_max : range of the orientation. We will select the orientations `or`
        that verify orient_min <= or < orient_max.

    Returns
    -------
    total : value of the histogram.
    """
    total = 0
    Y, X = magnitude.shape[:2]
    for y in range(Y):
        for x in range(X):
            current_orientation = orientation[y,x]
            if orient_min <= current_orientation and current_orientation < orient_max:
                total += magnitude[y,x]
    return total / (Y * X)

def normalize_block(block, norm_method, eps=1e-5):
    """Normalize a block."""
    if norm_method == 'L1':
        norm = block / (np.sum(np.abs(block)) + eps)
    elif norm_method == 'L1-sqrt':
        norm = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif norm_method == 'L2':
        norm = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif norm_method == 'L2-Hys':
        norm = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        norm = np.minimum(norm, 0.2)
        norm = norm / np.sqrt(np.sum(norm ** 2) + eps ** 2)
    else:
        raise ValueError('Block normalization method invalid.')

    return norm

def hog(image, orientations=9, pix_per_cell=8, cells_per_block=3, normalization='L2-Hys'):
    """Perform HOG (Histograms Of Gradients) on an image."""

    ### 1. compute the gradients of the image ============================================
    grad_x, grad_y = gradient(image)                           # compute gradients
    grad_x, grad_y = select_biggest_magnitude(grad_x, grad_y)  # select the channel with biggest magnitude
    magnitude = grad_magnitude(grad_x, grad_y)
    orientation = grad_orientation(grad_x, grad_y)

    ### 2. compute the histogram cell by cell ============================================
    s = image.shape[0]
    n_cells = s // pix_per_cell
    histogram = np.zeros((n_cells, n_cells, orientations))
    for orient_idx in range(histogram.shape[2]):               # for each orientation
        for y_idx in range(histogram.shape[0]):
            for x_idx in range(histogram.shape[1]):            # for each cell, compute the histogram of the cell
                histogram[y_idx,x_idx,orient_idx] = cell_histogram(
                    magnitude=magnitude[
                        y_idx*pix_per_cell:(y_idx+1)*pix_per_cell,
                        x_idx*pix_per_cell:(x_idx+1)*pix_per_cell],
                    orientation=orientation[
                        y_idx*pix_per_cell:(y_idx+1)*pix_per_cell,
                        x_idx*pix_per_cell:(x_idx+1)*pix_per_cell],
                    orient_min=180/orientations*orient_idx,
                    orient_max=180/orientations*(orient_idx+1),
                )

    ### 3. normalize the blocks ==========================================================
    n_blocks = (n_cells - cells_per_block) + 1
    normalized_blocks = np.zeros((n_blocks, n_blocks, cells_per_block, cells_per_block, orientations))

    for bloc_y in range(n_blocks):
        for bloc_x in range(n_blocks):
            block = histogram[
                bloc_y:bloc_y+cells_per_block,
                bloc_x:bloc_x+cells_per_block,
                :]
            normalized_blocks[bloc_y,bloc_x,:] = normalize_block(block, norm_method=normalization)

    ### 4. flatten the feature vector ====================================================
    return normalized_blocks.ravel()