import numpy as np
from scipy import ndimage
import math
import cv2
import matplotlib.pyplot as plt
from Canny import Canny


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def draw_hough_line(accumulator, thetas, rhos):
    plt.imshow(accumulator, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    plt.savefig('./hogh.png')
    plt.show()


if __name__ == '__main__':
    # read image
    image = cv2.imread('../images/marion_airport.tif')
    # averaging filter
    image_smoothed = ndimage.filters.convolve(image, np.ones(5 * 5).reshape((5, 5)) / 25)
    # binary image
    image_smoothed[image_smoothed > 127] = 255
    image_smoothed[image_smoothed <= 127] = 0
    # canny
    image_canny = Canny(image_smoothed, 13, 2, 0.05, 0.15)
    # hough transform
    accumulator, thetas, rhos = hough_line(image)
    rhos = 0.1 * rhos
    # draw hough curves
    draw_hough_line(accumulator, thetas, rhos)
    # find a vertical line and draw the detected lines
    for i in range(0, len(accumulator)):
        rho = accumulator[i][0][0]
        theta = accumulator[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./hogh_ot.png', image)
