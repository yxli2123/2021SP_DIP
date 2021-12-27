import cv2
import numpy as np


def reconstruct(image, element, marker, iteration, method='erosion'):
    if method == 'erosion':
        for _ in range(iteration):
            image = np.maximum(cv2.erode(image, element), marker)
        return image

    if method == 'dilation':
        for _ in range(iteration):
            image = np.minimum(cv2.dilate(image, element), marker)
        return image

    return image


def boarder_0(image_):
    image = np.copy(image_)
    H, W = image.shape
    image[0:H, 0] = 255 - image[0:H, 0]
    image[0:H, W - 1] = 255 - image[0:H, W - 1]
    image[0, 1:W - 1] = 255 - image[0, 1:W - 1]
    image[H - 1, 1:W - 1] = 255 - image[H - 1, 1:W - 1]
    image[1:H - 1, 1:W - 1] = 0
    return image


def boarder_1(image_):
    image = np.copy(image_)
    H, W = image.shape
    image[1:H - 1, 1:W - 1] = 0
    return image


def converse(image):
    img_c = np.copy(image)
    img_c[image == 255] = 0
    img_c[image == 0] = 255
    return img_c


if __name__ == '__main__':
    # read the image
    img = cv2.imread('../images/text_image.tif', 0)

    """Detect long character"""

    # structuring element for erosion
    kernel_erosion = np.ones((51, 1), dtype='uint8')

    # structuring element for geodesic dilation
    kernel_open = np.ones((3, 3), dtype='uint8')

    # erosion
    img_e = cv2.erode(img, kernel_erosion)

    img_o = cv2.dilate(img_e, kernel_erosion)

    # morphological reconstruction by dilation
    img_r = reconstruct(img_e, kernel_open, img, 100, 'dilation')

    # write the answer
    cv2.imwrite('../images/Re_text_0.png', img)
    cv2.imwrite('../images/Re_text_1.png', img_e)
    cv2.imwrite('../images/Re_text_2.png', img_o)
    cv2.imwrite('../images/Re_text_3.png', img_r)

    """Fill the hole"""
    F_xy = boarder_0(img)
    img_converse = converse(img)

    kernel = np.ones((3, 3), dtype='uint8')

    img_h = reconstruct(F_xy, kernel, img_converse, 600, method='dilation')
    img_h = converse(img_h)
    cv2.imwrite('../images/Fill_text_0.png', img)
    cv2.imwrite('../images/Fill_text_1.png', img_converse)
    cv2.imwrite('../images/Fill_text_2.png', img_h)

    """Clear the boarder"""
    F_xy = boarder_1(img)
    img_x_r = reconstruct(F_xy, kernel, img, 200, 'dilation')
    img_x = img - img_x_r
    cv2.imwrite('../images/Clear_text_0.png', img)
    cv2.imwrite('../images/Clear_text_1.png', img_x_r)
    cv2.imwrite('../images/Clear_text_2.png', img_x)
