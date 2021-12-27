import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogramEqualize(img, L):
    # Compute the density distribution function
    ddf = np.bincount(img.flatten(), minlength=L) / img.size

    # Compute the cumulative distribution function,
    cdf = np.cumsum(ddf)

    # Uncomment code bellow to plot histogram for the original image
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(L), ddf)
    ax1.set_ylabel('DDF')
    ax1.set_title("DDF and CDF of Equalized Image")
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(np.arange(L), cdf, 'r')
    ax2.set_xlim([0, L-1])
    ax2.set_ylabel('CDF')
    ax2.set_xlabel('Gray Scale')
    plt.savefig('../result/histogram_equalized.png')
    '''

    # Use the general histogram equalization formula
    # h(v) = round((cdf[v] - min{cdf})*(L-1)/(M*N - min{cdf})),
    # where v is the value of original gray scale

    img_cdf = cdf[img]
    img_eq = np.round((img_cdf - cdf.min()) * (L - 1) / (cdf.max() - cdf.min()))
    img_eq = np.array(img_eq, dtype='int64')

    return 256*img_eq // L


if __name__ == '__main__':
    image = cv2.imread('../images/bridge.jpg', 0)
    image_e = histogramEqualize(image, L=256)
    cv2.imwrite('../result/bridge_eq_256.jpg', image_e)
