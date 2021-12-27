import cv2
import numpy as np


def element(r):
    ele = np.zeros((2*r, 2*r), dtype='uint8')
    for h in range(2*r):
        for w in range(2*r):
            if ((h-r)**2 + (w-r)**2) <= r**2:
                ele[h, w] = 255
    return ele


if __name__ == '__main__':
    img = cv2.imread('../images/rice_image_with_intensity_gradient.tif', 0)
    img1 = np.copy(img)
    img1[img > 127] = 255
    img1[img <= 127] = 0
    mask = element(40)
    img2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, mask)
    img3 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, mask)
    img4 = np.copy(img3)
    img4[img3 > 50] = 255
    img4[img3 <= 50] = 0
    cv2.imwrite('../images/tophat0.png', img)
    cv2.imwrite('../images/tophat1.png', img1)
    cv2.imwrite('../images/tophat2.png', img2)
    cv2.imwrite('../images/tophat3.png', img3)
    cv2.imwrite('../images/tophat4.png', img4)
