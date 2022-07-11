import numpy as np
import cv2


def single_scale_retinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def multi_scale_retinex(img, variance_list):
    retinex = np.zeros(img.shape)
    for variance in variance_list:
        retinex += single_scale_retinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex


def normalise_color(img):
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
            (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
    img = np.uint8(img)
    return img


def ssr(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = single_scale_retinex(img, variance)
    img_retinex = normalise_color(img_retinex)
    return img_retinex


def msr(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, variance_list)
    img_retinex = normalise_color(img_retinex)
    return img_retinex


if __name__ == "__main__":
    variance_list = [15, 80, 250]
    variance = 100

    img_clear = cv2.imread(
        '/home/viet/OneDrive/Studying_Materials/Computer_Vision/data/outdoor/clear/0189.png')
    img_hazy = cv2.imread(
        '/home/viet/OneDrive/Studying_Materials/Computer_Vision/data/outdoor/hazy/0189_0.85_0.2.jpg')
    img_vanilla_ssr = ssr(img_hazy, variance)
    img_vanilla_msr = msr(img_hazy, variance_list)

    cv2.imshow('Hazy', img_hazy)
    cv2.imshow('Clear', img_clear)
    cv2.imshow('Vanilla SSR', img_vanilla_ssr)
    cv2.imshow('Vanilla MSR', img_vanilla_msr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
