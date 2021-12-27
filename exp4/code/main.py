import cv2
from hw1 import Interpolation
from hw1 import GaussianBlur


def myPyrDown(image):
    image = GaussianBlur(image, (7, 7), 0)
    image[1::2, 1::2] = 0
    size = (image.shape[0]//2, image.shape[1]//2)
    image = Interpolation(image, size, interpolation=cv2.INTER_NEAREST)
    return image


def myPyrUp(image):
    size = (image.shape[0]*2, image.shape[1]*2)
    image = Interpolation(image, size, interpolation=cv2.INTER_CUBIC)
    return image


if __name__ == '__main__':
    img0 = cv2.imread('../images/demo-1.jpg', 0)
    img1 = myPyrDown(img0)
    img2 = myPyrDown(img1)
    img3 = myPyrDown(img2)

    img0_res = cv2.subtract(img0, myPyrUp(img1))
    img1_res = cv2.subtract(img1, myPyrUp(img2))
    img2_res = cv2.subtract(img2, myPyrUp(img3))

    cv2.imwrite('../images/img0.png', img0)
    cv2.imwrite('../images/img1.png', img1)
    cv2.imwrite('../images/img2.png', img2)
    cv2.imwrite('../images/img3.png', img3)

    cv2.imwrite('../images/img0_res.png', img0_res + 128)
    cv2.imwrite('../images/img1_res.png', img1_res + 128)
    cv2.imwrite('../images/img2_res.png', img2_res + 128)
    cv2.imwrite('../images/img3_res.png', img3)