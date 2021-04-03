import numpy as np
import cv2


def averageBlur(img):
    # Create an empty canvas to fill
    img_average = np.zeros_like(img, dtype='int64')

    # Define the kernel, here we choose 3x3 Laplace
    kernel = np.ones((3, 3)) / 9

    # Symmetrically pad the image to extend the boundary
    img_pad = np.pad(img, 1, mode='symmetric')

    # Compute each pixel's value window by window
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            window = img_pad[row:row + 3, col:col + 3]
            value = np.sum(window * kernel)
            img_average[row, col] = int(value)
    return img_average


if __name__ == '__main__':
    # Read the image
    img = cv2.imread('../images/circuit.jpg', 0)

    for _ in range(20):
        img = averageBlur(img)

    # Save the processed image
    cv2.imwrite('../result/circuit_average_20.jpg', img)
