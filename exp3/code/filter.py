import cv2
import numpy as np


def D_func(H, W, k=0.0025):
    """
    :param H: the height of image ready to degrade
    :param W: the width of image ready to degrade
    :param k: the turbulence factor, 0.0025(severe), 0.001(mild), 0.00025(low)
    :return: the system function in frequency domain. Note it is not shifted to center!!!
    """
    D_uv = np.zeros((H, W))
    for u in range(H):
        for v in range(W):
            D_uv[u, v] = np.exp(-1 * k * ((u - H / 2)**2 + (v - W / 2)**2)**(5/6))
    return D_uv


def degradation(img, k=0.0025):
    F_uv = np.fft.fftshift(np.fft.fft2(img))
    H_uv = D_func(img.shape[0], img.shape[1], k=k)
    G_uv = F_uv * H_uv
    img_d = np.fft.ifft2(np.fft.fftshift(G_uv))
    noise = np.random.normal(scale=0.5, size=img.shape)
    img_d = img_d + noise
    img_d = np.abs(img_d).astype('uint8')
    return img_d


def inverse_filter(img_d, eps=0.2):
    H_uv = D_func(img_d.shape[0], img_d.shape[1])
    H_uv = np.where(H_uv < eps, eps, H_uv)
    G_uv = np.fft.fftshift(np.fft.fft2(img_d))
    F_uv = G_uv / H_uv
    img = np.fft.ifft2(np.fft.ifftshift(F_uv))
    img = np.real(img)
    img = 255 * (img - img.min()) / (img.max() - img.min())
    return img.astype('uint8')


def Wiener_filter(img_d, K=0.008):
    H_uv = D_func(img_d.shape[0], img_d.shape[1])
    G_uv = np.fft.fftshift(np.fft.fft2(img_d))
    F_uv = (np.conj(H_uv) / (np.abs(H_uv) ** 2 + K)) * G_uv
    img = np.fft.ifft2(np.fft.ifftshift(F_uv))
    img = np.real(img)
    img = np.where(img > 255, 255, img)
    return img.astype('uint8')


if __name__ == '__main__':
    image = cv2.imread('../image/demo-1.jpg', 0)
    image_de = degradation(image)
    cv2.imwrite('../image/image_de.png', image_de)
    image_inv = inverse_filter(image_de)
    cv2.imwrite('../image/image_inv.png', image_inv)
    image_win = Wiener_filter(image_de, K=0.001)
    cv2.imwrite('../image/image_win.png', image_win)
