import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram(img, L, path):
    # Compute the density distribution function
    ddf = np.bincount(img.flatten(), minlength=L) / img.size
    # Compute the cumulative distribution function,
    cdf = np.cumsum(ddf)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(L), ddf)
    ax1.set_ylabel('DDF')
    ax1.set_title("DDF and CDF of the Image")
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.arange(L), cdf, 'r')
    ax2.set_xlim([0, L-1])
    ax2.set_ylabel('CDF')
    ax2.set_xlabel('Gray Scale')
    plt.savefig(path)


if __name__ == '__main__':
    image = cv2.imread('../images/Fig1046(a)(septagon_noisy_shaded).tif', 0)
    # histogram(image, 256, './histogram_.png')
    img_list = np.split(image, [651//2], axis=0)
    img_list = [np.split(img, [814//2], axis=1) for img in img_list]
    for im_v in img_list:
        for im in im_v:
            ddf = np.bincount(im.flatten(), minlength=256) / im.size
            cdf = np.cumsum(ddf)
            m = np.cumsum(np.arange(256) * ddf)
            sigma = (m[-1] * cdf - m) ** 2 / ((cdf + 0.001) * (1 - cdf - 0.001))  # add 0.001 to avoid 0 denominator
            k = np.argmax(sigma)
            eta = sigma[k] / im.var()
            im[im > k] = 255
            im[im <= k] = 0
    img = [np.hstack(im_v) for im_v in img_list]
    img = np.vstack(img)
    cv2.imwrite('./se.png', img)


