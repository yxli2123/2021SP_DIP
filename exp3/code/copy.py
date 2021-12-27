import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import cv2


def make_PSF(kernel_size=15, angle=60):
    PSF = np.diag(np.ones(kernel_size))  # 初始模糊核的方向是-45度
    angle = angle + 45  # 抵消-45度的影响
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)  # 生成旋转算子
    PSF = cv2.warpAffine(PSF, M, (kernel_size, kernel_size), flags=cv2.INTER_LINEAR)  # 实现旋转变换
    PSF = PSF / PSF.sum()  # 使模糊核的权重和为1
    return PSF


# 此函数扩展PSF0，使之与image0一样大小
def extension_PSF(image0, PSF0):
    [img_h, img_w] = image0.shape
    [h, w] = PSF0.shape
    PSF = np.zeros((img_h, img_w))
    PSF[0:h, 0:w] = PSF0
    return PSF


# 在频域对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)  # 对输入图像进行傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps # 对运动模糊核进行傅里叶变换，并加上一个很小的数
    cv2.imwrite('filter.png', 10*np.log10(np.abs(PSF_fft)).astype('uint8'))
    blurred = np.fft.ifft2(input_fft * PSF_fft)  # 在频域进行运动模糊
    blurred = np.abs(blurred)
    return blurred


def inverse(input, PSF, eps):  # 逆滤波
    input_fft = np.fft.fft2(input)  # 对退化图像进行傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps  # 对运动模糊核进行傅里叶变换，并加上一个很小的数
    Output_fft = input_fft / PSF_fft  # 在频域进行逆滤波
    result = np.fft.ifft2(Output_fft)  # 进行傅里叶反变换
    result = np.abs(result)
    return result


def wiener(input, PSF, eps, K=0.01):  # 维纳滤波
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF)
    PSF_fft_1 = (np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)) * input_fft  # 根据公式写就够了
    result = np.fft.ifft2(PSF_fft_1)
    result = np.abs(result)
    return result


if __name__ == "__main__":
    image = cv2.imread('../image/demo-2.jpg', 0)  # 这是你要搞得图片

    eps = 1e-3
    plt.figure(1)
    # 进行运动模糊处理
    PSF = make_PSF(15, 60)
    # 扩展PSF，使其与图像一样大小
    PSF = extension_PSF(image, PSF)
    blurred = make_blurred(image, PSF, eps)  # 在频域对图像进行运动模糊

    # 添加噪声,standard_normal产生随机的函数
    blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    # numpy.random.normal(loc=0.0, scale=1.0, size=None)
    plt.figure(figsize=(8, 6))
    plt.subplot(131)
    plt.axis("off")
    plt.gray(), plt.title("motion & noisy blurred"), plt.imshow(blurred_noisy)  # 显示添加噪声且运动模糊的图像

    result = inverse(blurred_noisy, PSF, eps)  # 对添加噪声的图像进行逆滤波
    plt.subplot(132)
    plt.axis("off"), plt.title("inverse deblurred"), plt.imshow(result)

    result = wiener(blurred_noisy, PSF, eps, K=0.01)  # 对添加噪声的图像进行维纳滤波
    plt.subplot(133)
    plt.axis("off"), plt.title("wiener deblurred(k=0.01)"), plt.imshow(result)
    plt.show()
