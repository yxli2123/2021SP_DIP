import numpy as np
import cv2


if __name__ == '__main__':
    # we take the original image as high resolution image, dented as HR
    GT = cv2.imread('../images/lenna.jpg')

    # we down sample it into 0.5 scale, with bilinear interpolation
    LR_bilinear = cv2.resize(GT, (GT.shape[0]//4, GT.shape[1]//4), interpolation=cv2.INTER_LINEAR)
    LR_nearest = cv2.resize(GT, (GT.shape[0]//4, GT.shape[1]//4), interpolation=cv2.INTER_NEAREST)
    LR_bicubic = cv2.resize(GT, (GT.shape[0]//4, GT.shape[1]//4), interpolation=cv2.INTER_CUBIC)

    # we now interpolate them using different methods
    HR_linear2linear = cv2.resize(LR_bilinear, (GT.shape[0], GT.shape[1]), interpolation=cv2.INTER_LINEAR)
    HR_linear2nearest = cv2.resize(LR_bilinear, (GT.shape[0], GT.shape[1]), interpolation=cv2.INTER_NEAREST)
    HR_linear2cubic = cv2.resize(LR_bilinear, (GT.shape[0], GT.shape[1]), interpolation=cv2.INTER_CUBIC)

    # compute the PSNR
    MSE_linear2linear = np.mean((GT - HR_linear2linear)**2)
    PSNR_linear2linear = 10*np.log10(255 * 255 / MSE_linear2linear)
    print("PSNR = {}".format(PSNR_linear2linear))

    MSE_linear2nearest = np.mean((GT - HR_linear2nearest) ** 2)
    PSNR_linear2nearest = 10 * np.log10(255 * 255 / MSE_linear2nearest)
    print("PSNR = {}".format(PSNR_linear2nearest))

    MSE_linear2cubic = np.mean((GT - HR_linear2cubic) ** 2)
    PSNR_linear2cubic = 10 * np.log10(255 * 255 / MSE_linear2cubic)
    print("PSNR = {}".format(PSNR_linear2cubic))


