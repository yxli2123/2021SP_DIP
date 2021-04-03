import numpy as np
import cv2


def sharpen(image):
    # Create an empty canvas to fill
    img_grad = np.zeros_like(image, dtype='int64')

    # Define the kernel, here we choose 3x3 Laplace
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])

    # Symmetrically pad the image to extend the boundary
    img_pad = np.pad(image, 1, mode='symmetric')

    # Compute each pixel's value window by window
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = img_pad[row:row + 3, col:col + 3]
            value = np.sum(np.array(window, dtype='int64') * np.array(kernel, dtype='int64'))
            img_grad[row, col] = value

    img_sharpen = image - img_grad
    return img_sharpen


if __name__ == '__main__':
    # Read the image
    img = cv2.imread('../images/moon.jpg', 0)

    img = sharpen(img)

    # Save the processed image.
    cv2.imwrite('../result/moon_sharpen_20.jpg', img)
