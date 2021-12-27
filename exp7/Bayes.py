import numpy as np
import cv2


def prob(X, mean, cov):
    X = X.reshape(X.size, 1)
    exp = np.exp(-0.5 * (X - mean).T.dot(np.linalg.inv(cov)).dot(X - mean)).item()
    den = np.sqrt((2 * np.pi)**4 * np.linalg.det(cov))
    return exp / den


if __name__ == '__main__':
    # read 4 images, i.e., 1 for blue, 2 for green, 3 for red, and 4 for infrared
    # flatten them into 1-D arrays
    img_1 = cv2.imread('./image/Fig1213(a)(WashingtonDC_Band1_512).tif', 0).flatten()
    img_2 = cv2.imread('./image/Fig1213(b)(WashingtonDC_Band2_512).tif', 0).flatten()
    img_3 = cv2.imread('./image/Fig1213(c)(WashingtonDC_Band3_512).tif', 0).flatten()
    img_4 = cv2.imread('./image/Fig1213(d)(WashingtonDC_Band4_512).tif', 0).flatten()

    # normalize these images
    img_1 = (img_1 - img_1.mean()) / img_1.std()
    img_2 = (img_2 - img_2.mean()) / img_2.std()
    img_3 = (img_3 - img_3.mean()) / img_3.std()
    img_4 = (img_4 - img_4.mean()) / img_4.std()

    # stack these 1-D arrays into 4-dimension samples
    img = np.stack((img_1, img_2, img_3, img_4), axis=1)

    # read the masks, i.e, 1 for water, 2 for urban, and 3 for vegetation
    # flatten them into 1-D arrays
    img_mask_1 = cv2.imread('./image/Fig1213(e)(Mask_B1_without_numbers).tif', 0).flatten()
    img_mask_2 = cv2.imread('./image/Fig1213(e)(Mask_B2_without_numbers).tif', 0).flatten()
    img_mask_3 = cv2.imread('./image/Fig1213(e)(Mask_B3_without_numbers).tif', 0).flatten()

    # extract the index
    mask_1 = np.where(img_mask_1 == 255)
    mask_2 = np.where(img_mask_2 == 255)
    mask_3 = np.where(img_mask_3 == 255)

    # extract the interested area of each pattern
    area_1 = img[mask_1]
    area_2 = img[mask_2]
    area_3 = img[mask_3]

    # train the mean vectors and covariance matrices
    mean_1 = np.mean(area_1, axis=0).reshape(4, 1)
    mean_2 = np.mean(area_2, axis=0).reshape(4, 1)
    mean_3 = np.mean(area_3, axis=0).reshape(4, 1)
    cov_1 = np.cov(area_1.T)
    cov_2 = np.cov(area_2.T)
    cov_3 = np.cov(area_3.T)

    # calculate the posterior probability
    score_1 = np.array([prob(pix.T, mean_1, cov_1) for pix in img])
    score_2 = np.array([prob(pix.T, mean_2, cov_2) for pix in img])
    score_3 = np.array([prob(pix.T, mean_3, cov_3) for pix in img])

    # choose the pattern that has the maximal posterior probability
    result = np.argmax((score_1, score_2, score_3), axis=0).reshape(512, 512)

    # classify these patterns
    img_result_1 = np.where(result == 0, 255, 0)
    img_result_2 = np.where(result == 1, 255, 0)
    img_result_3 = np.where(result == 2, 255, 0)

    # save the result
    cv2.imwrite('./01.png', img_result_1)
    cv2.imwrite('./02.png', img_result_2)
    cv2.imwrite('./03.png', img_result_3)

