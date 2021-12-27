""" Canny Edge Detection is based on the following five steps:
    1. Gaussian filter
    2. Gradient Intensity
    3. Non-maximum suppression
    4. Double threshold
    5. Edge tracking
    This module contains these five steps as five separate Python functions.
"""

from scipy import ndimage
import numpy as np
import cv2


def round_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle


def gs_filter(img, size, sigma):
    """ Step 1: Gaussian filter"""
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return cv2.GaussianBlur(img, (size, size), sigma)


def gradient_intensity(img):
    """ Step 2: Find gradients"""

    # Kernel for Gradient in x-direction
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype='float64')
    # Kernel for Gradient in y-direction
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype='float64')
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return G, D


def suppression(img, D):
    """ Step 3: Non-maximum suppression"""

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i, j] = img[i, j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i, j] = img[i, j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    return Z


def threshold(img, t, T):
    """ Step 4: Thresholding"""
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(15),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return img, cf.get('WEAK')


def tracking(img, weak, strong=255):
    """ Step 5:Edge tracking"""

    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                            or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                            or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def Canny(img, size, sigma, low_th, high_th):
    img = gs_filter(img, size, sigma)
    gradient, D = gradient_intensity(img)
    nonMaxImg = suppression(gradient, D)
    thresholdImg, weak = threshold(nonMaxImg, 255 * low_th, 255 * high_th)
    img = tracking(thresholdImg, weak, 255)
    return img
