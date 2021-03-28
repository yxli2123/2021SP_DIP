import cv2
import numpy as np
import math
import time


# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0


# Padding the image to prevent overflow
def padding(img, H, W, C):
    z_img = np.zeros((H + 4, W + 4, C))
    z_img[2:H + 2, 2:W + 2, :C] = img
    # Pad the first/last two col and row
    z_img[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    z_img[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    z_img[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    z_img[0:2, 2:W + 2, :C] = img[0:1, :, :C]
    # Pad the missing eight points
    z_img[0:2, 0:2, :C] = img[0, 0, :C]
    z_img[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    z_img[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    z_img[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]
    return z_img


# Bicubic operation
def bicubic(img, ratio, a):
    # fetch image height, width and channel
    H, W, C = img.shape
    # padding the image
    img = padding(img, H, W, C)
    # create a new blank image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1 / ratio

    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.array([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.array([[img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c],
                                   img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]],
                                  [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c],
                                   img[int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]],
                                  [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c],
                                   img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]],
                                  [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c],
                                   img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]])
                mat_r = np.array([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst


if __name__ == '__main__':
    # open an image
    image = cv2.imread('../images/lenna.jpg')
    # set scale factor
    scale_factor = 2

    # bicubic interpolation
    print('Ready to interpolate, it takes a while!')
    start = time.time()
    image_re = bicubic(image, scale_factor, -0.5)
    end = time.time()
    print('It costs {} seconds!'.format(end - start))

    # save the image
    cv2.imwrite('../interpolation/bicubic_lenna_x{}.jpg'.format(scale_factor), image_re)
    print('Save the interpolated image successfully!')
