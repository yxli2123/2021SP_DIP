import numpy as np
import cv2
import time


def nearest(img, ratio):
    # read same metadata from the original image, i.e., height, width, and channel
    H, W, C = img.shape
    # transpose the original image shape (H, W, C) to (C, H, W)
    img = np.transpose(img, (2, 0, 1))

    # create the canvas for expected resized image
    H_new = int(ratio * H)
    W_new = int(ratio * W)
    img_new = np.zeros((C, H_new, W_new), dtype='uint8')

    # paint the pixel row by row and column by column
    for h_new in range(H_new):
        for w_new in range(W_new):
            h = h_new * H / H_new  # h is a float number
            w = w_new * W / W_new  # w is a float number
            h_nearest = round(h) if round(h) < H else H - 1
            w_nearest = round(w) if round(w) < W else W - 1
            img_new[:, h_new, w_new] = img[:, h_nearest, w_nearest]
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
    image_re = nearest(image, scale_factor)
    end = time.time()
    print('It costs {} seconds!'.format(end - start))

    # save the image
    cv2.imwrite('../interpolation/nearest_lenna_x{}.jpg'.format(scale_factor), image_re)
    print('Save the interpolated image successfully!')
