import numpy as np
import cv2
import time


# Padding the image to prevent overflow
def padding(img, H, W, C):
    z_img = np.zeros((H + 1, W + 1, C))
    z_img[0:H, 0:W, :] = img
    # pad the first/last col and row
    z_img[0:H, W, :] = img[:, W - 1, :]
    z_img[H, 0:W, :] = img[H - 1, :, :]
    # pad the 4 corners
    z_img[H, W, :] = img[H - 1, W - 1, :]
    return z_img


def bilinear(img, ratio):
    H, W, C = img.shape

    # create the canvas for expected resized image
    H_new = int(ratio * H)
    W_new = int(ratio * W)
    img_new = np.zeros((C, H_new, W_new), dtype='uint8')

    # pad the image
    img = padding(img, H, W, C)

    # transpose the original image shape (H, W, C) to (C, H, W)
    img = np.transpose(img, (2, 0, 1))

    # paint the pixel row by row and column by column
    for h_new in range(H_new):
        for w_new in range(W_new):
            h = h_new * H / H_new  # h is a float number
            w = w_new * W / W_new  # w is a float number
            tl = img[:, int(h) - 1, int(w) - 1]
            tr = img[:, int(h) - 1, int(w)]
            bl = img[:, int(h), int(w) - 1]
            br = img[:, int(h), int(w)]
            p1 = (bl - tl) * (h - int(h)) + tl
            p2 = (br - tr) * (h - int(h)) + tr
            p = (p2 - p1) * (w - int(w)) + p1
            img_new[:, h_new, w_new] = p
    img_new = np.transpose(img_new, (1, 2, 0))
    return img_new


if __name__ == '__main__':
    # open an image
    image = cv2.imread('../images/lenna.jpg')
    # set scale factor
    scale_factor = 2

    # bicubic interpolation
    print('Ready to interpolate, it takes a while!')
    start = time.time()
    image_re = bilinear(image, scale_factor)
    end = time.time()
    print('It costs {} seconds!'.format(end - start))

    # save the image
    cv2.imwrite('../interpolation/bilinear_lenna_x{}.jpg'.format(scale_factor), image_re)
    print('Save the interpolated image successfully!')
