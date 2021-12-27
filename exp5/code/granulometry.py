import numpy as np
import cv2
import matplotlib.pyplot as plt

def element(r):
    ele = np.zeros((2*r, 2*r), dtype='uint8')
    for h in range(2*r):
        for w in range(2*r):
            if ((h-r)**2 + (w-r)**2) <= r**2:
                ele[h, w] = 255
    return ele


if __name__ == '__main__':
    img = cv2.imread('../images/wood_dowels.tif', 0)
    cv2.imwrite('../images/gra00.png', img)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element(5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element(5))
    cv2.imwrite('../images/gra01.png', img)
    radius = np.arange(3, 35)
    img_ = []
    di = []
    for r in radius:
        img_.append(cv2.morphologyEx(img, cv2.MORPH_OPEN, element(r)))
    for i in range(len(img_) - 1):
        di.append(np.sum(img_[i].astype('float64') - img_[i+1].astype('float64')))
    x = np.array(radius[1:len(radius)])
    y = np.array(di)
    plt.plot(x, y)
    plt.xlabel('r')
    plt.ylabel('Differences in surface area')
    plt.savefig('../image/diff.png')

