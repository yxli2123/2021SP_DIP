import numpy as np
import cv2


def element(r):
    ele = np.zeros((2*r, 2*r), dtype='uint8')
    for h in range(2*r):
        for w in range(2*r):
            if ((h-r)**2 + (w-r)**2) <= r**2:
                ele[h, w] = 1
    return ele


if __name__ == '__main__':
    img = cv2.imread('../images/dark_blobs_on_light_background.tif', 0)
    cv2.imwrite('../images/texture0.png', img)
    img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element(30))
    img2 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, element(60))
    img3 = cv2.morphologyEx(img2, cv2.MORPH_GRADIENT, np.ones((3, 3)))
    img3 = img + img3

    cv2.imwrite('../images/texture1.png', img1)
    cv2.imwrite('../images/texture2.png', img2)
    cv2.imwrite('../images/texture4.png', img3)
