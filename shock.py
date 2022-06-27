#!/usr/bin/env python
__authors__ = ["Tobias Uelwer", "Thomas Germer"]
__date__ = "2022/06/27"
__license__ = "MIT"

import scipy
import scipy.ndimage
import numpy as np


def laplacian(img):
    """
    Calculates the Laplace-filtered image.
    
    Parameters
    ----------
    img: numpy.ndarray
        Grayscale image with shape (h, w)
    
    Returns
    -------
    img: numpy.ndarray
        Filtered image
    """
    f = 0.25 * np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return scipy.ndimage.convolve(img, f)


def gradient_norm(x):
    """
    Calculates the norm of the gradient at each pixel.
    
    Parameters
    ----------
    img: numpy.ndarray
        Grayscale image with shape (h, w)
    
    Returns
    -------
    img: numpy.ndarray
        Gradient image
    """
    assert (
        len(x.shape) == 2
    ), "To many color-channels. Image must be grayscale with shape (h, w)."
    gx, gy = np.gradient(x)
    grad = np.hypot(gx, gy)  # same as: np.sqrt(np.sum(gx**2+gy**2))
    return grad


def blur(x, sigma=0.5):
    """
    Applies a Gaussian blur to the input image
    
    Parameters
    ----------
    img: numpy.ndarray
        Grayscale image with shape (h, w) or color image with shape (h, w, 3)
    sigma: float
        Standard deviation. Larger values correspond to stronger blur. Defaults to 0.5.
    
    Returns
    -------
    img: numpy.ndarray
        Blurred image
    """
    assert (
        len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[2] == 3)
    ), "Invalid input. Input must be grayscale image with shape (h, w) or color image with shape (h, w, 3)."
    if len(x.shape) == 2:
        return scipy.ndimage.gaussian_filter(x, sigma)
    if len(x.shape) == 3 and x.shape[2] == 3:
        return scipy.ndimage.gaussian_filter(x, (sigma, sigma, 0))
        


def shock(img, steps=20, step_size=0.25, sigma=0.5):
    """
    Shock filter [1] for grayscale images.
    
    Parameters
    ----------
    img: numpy.ndarray
        Grayscale image with shape (h, w)
    steps: int
        Number of iterations to apply. Usually, a small number 
        of iterations is sufficient.
    step_size: float
        Step size. Should be larger than zero.
    sigma: float
        Standard deviation of the intermediate Gaussian blur. Defaults to 0.5.
    
    Returns
    -------
    img: numpy.ndarray
        Filtered image having the same shape as the input image.
    
    References
    ----------
    [1] Osher, Stanley, and Leonid I. Rudin. "Feature-oriented image enhancement using shock filters."
    SIAM Journal on numerical analysis 27.4 (1990): 919-940.
    """
    img = img.copy()
    for i in range(steps):
        img = blur(img, sigma=0.5)
        lap = laplacian(img)
        grad = gradient_norm(img)
        img -= step_size * np.sign(lap) * grad
    return img


def chromatic_shock(img, steps=30, step_size=0.25, sigma=0.5):
    """
    Shock filter for color images. Based on [1]. Note, that this implementation uses
    the laplacian (similar to the original shock filter) instead of the second derivative 
    in the direction of the gradient.
    
    Parameters
    ----------
    img: numpy.ndarray
        Grayscale image with shape (h, w, 3)
    steps: int
        Number of iterations to apply. Usually, a small number 
        of iterations is sufficient.
    step_size: float
        Step size. Should be larger than zero.
    sigma: float
        Standard deviation of the intermediate Gaussian blur. Defaults to 0.5.
    
    Returns
    -------
    img: numpy.ndarray
        Filtered image having the same shape as the input image.
    
    References
    [1] Schuler, Christian J., et al. "Blind correction of optical aberrations." 
    European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2012.
    """
    img = img.transpose(2, 0, 1)
    for i in range(steps):
        for channel in img:
            channel[...] = blur(channel, sigma)
        z = img.mean(axis=0)
        s = np.sign(laplacian(z))
        for channel in img:
            grad = gradient_norm(channel)
            channel -= step_size * s * grad
    return img.transpose(1, 2, 0)
