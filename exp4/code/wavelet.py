import cv2
from scipy import signal
import numpy as np


def dwt2(image, h, g, delta):
    image = image.astype('float64')
    A = signal.convolve(h, image)
    A = signal.convolve(h.T, A)

    H = signal.convolve(g, image)
    H = signal.convolve(h, H)

    V = signal.convolve(g.T, image)
    V = signal.convolve(h.T, V)

    D = signal.convolve(delta, image)
    D = signal.convolve(delta.T, D)

    return A, (H, V, D)


if __name__ == '__main__':
    # load the image
    img = cv2.imread('../images/demo-2.tif', 0)

    # set the parameters of wavelet
    h = np.array([[0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758]])
    g = np.array([[0.5, -0.5]])
    delta = np.array([[1.0, 0.0, 0.0]])

    # calculate the wavelet transform
    cA, (cH, cV, cD) = dwt2(img, h, g, delta)

    # save the image
    cv2.imwrite('./3cA.png', cA)
    cv2.imwrite('./3cH.png', 5 * cH + 127)
    cv2.imwrite('./3cV.png', 5 * cV + 127)
    cv2.imwrite('./3cD.png', 5 * cD + 127)
