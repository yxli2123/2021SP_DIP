import cv2
import numpy as np


if __name__ == '__main__':
    image = cv2.imread('../images/Fig1038(a)(noisy_fingerprint).tif', 0)
    T = 120
    dT = 10
    cnt = 0
    while dT > 0.001:
        cnt += 1
        g = np.copy(image)
        G1 = np.where(image > T)
        G2 = np.where(image <= T)
        m1 = image[G1].mean()
        m2 = image[G2].mean()
        dT = abs(T - (m1 + m2) / 2)
        T = (m1 + m2) / 2
    image[image > T] = 255
    image[image <= T] = 0
    cv2.imwrite('./septagon_220.png', image)

