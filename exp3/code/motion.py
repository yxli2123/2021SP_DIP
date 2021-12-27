import numpy as np
import cv2


def H_func(H, W, T=1, a=0.1, b=0.1):
    H_uv = np.zeros((H, W), dtype='complex')
    for h in range(H):
        for w in range(W):
            H_uv[h, w] = T * np.sinc(a * (h - H // 2) + b * (w - W // 2)) * np.exp(
                -1j * np.pi * (a * (h - H // 2) + b * (w - W // 2)))
    return H_uv


def degradation(img, T=1, a=0.1, b=0.1, mean=0, std=0.001):
    H = img.shape[0]
    W = img.shape[1]
    F_uv = np.fft.fftshift(np.fft.fft2(img))
    H_uv = H_func(H, W, T, a, b)
    # noise = np.random.normal(loc=mean, scale=std, size=(H, W))
    G_uv = F_uv * H_uv
    img_d = np.fft.ifft2(np.fft.fftshift(G_uv))
    img_d = np.abs(img_d)
    img_d = np.where(img_d > 255, 255, img_d)
    img_d = np.where(img_d < 0, 0, img_d)
    return img_d.astype('uint8')


def inverse_filter(img_d):
    H_uv = H_func(img_d.shape[0], img_d.shape[1])
    G_uv = np.fft.fftshift(np.fft.fft2(img_d))
    F_uv = G_uv / H_uv
    img = np.fft.ifft2(np.fft.fftshift(F_uv))
    img = np.abs(img)
    img = 255 * (img - img.min()) / (img.max() - img.min())
    return img.astype('uint8')


def Wiener_filter(img_d, K=0.0001):
    H_uv = H_func(img_d.shape[0], img_d.shape[1])
    G_uv = np.fft.fftshift(np.fft.fft2(img_d))
    F_uv = (np.conj(H_uv) / (np.abs(H_uv) ** 2 + K)) * G_uv
    img = np.fft.ifft2(F_uv)
    img = np.abs(img)
    img = 255 * (img - img.min()) / (img.max() - img.min())
    return img.astype('uint8')


if __name__ == '__main__':
    image = cv2.imread('../image/demo-2.jpg', 0)
    image_de = degradation(image)
    cv2.imwrite('../image/2image_de.png', image_de)
    image_inv = inverse_filter(image_de)
    cv2.imwrite('../image/2image_inv.png', image_inv)
    image_win = Wiener_filter(image_de)
    cv2.imwrite('../image/2image_win.png', image_win)
