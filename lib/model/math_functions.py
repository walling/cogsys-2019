
import numpy as np

def distance(location1, location2):
    location1 = np.asarray(location1, dtype=np.float64)
    location2 = np.asarray(location2, dtype=np.float64)
    return np.linalg.norm(location1 - location2)

def gaussian(n, sigma):
    return np.exp(n**2 * (-0.5 / sigma**2))

def gaussian_field(width, height, depth, mean, sigma):
    # Calculate distance field around mean. The mean (x,y) location will be 0.
    d_x, d_y = np.meshgrid(
        np.arange(width)  - mean[0],
        np.arange(height) - mean[1])
    d = np.sqrt(d_x * d_x + d_y * d_y).T

    # Repeat the distance field to create the depth, ie. for a map of weights.
    d = np.repeat(np.expand_dims(d, 2), depth, axis=2)

    # Calculate and return the gaussian distribution on the distance field.
    return gaussian(d, sigma)

def gaussian_filter(tensor, axis, mean, sigma):
    mean        = np.copy(mean)
    w, h, depth = tensor.shape
    other_axis  = 1 - axis
    axis_size   = tensor.shape[other_axis]
    updates     = np.zeros(tensor.shape)
    for i in range(axis_size):
        mean[other_axis] = i
        updates += tensor * gaussian_field(w, h, depth, mean, sigma)
    return updates
