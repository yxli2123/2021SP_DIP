import numpy as np
import cv2


def median(image):
    # Create an empty canvas to fill
    img_median = np.zeros_like(image)

    # Symmetrically pad the image to extend the boundary
    img_pad = np.pad(image, 1, mode='symmetric')

    # Compute each pixel's value window by window
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = img_pad[row:row + 3, col:col + 3]
            value = np.median(window)
            img_median[row, col] = value
    return img_median


if __name__ == '__main__':
    # Read the image
    img = cv2.imread('../images/circuit.jpg', 0)

    for _ in range(200):
        img = median(img)

    # Save the processed image.
    cv2.imwrite('../result/circuit_median_200.jpg', img)
