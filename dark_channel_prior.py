import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
import matplotlib.pyplot as plt


def get_dark_channel(img, patch_size):
    dark_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(patch_size, patch_size))
    dark = cv2.erode(dark_channel,kernel)

    return dark


def estimate_atm_light(img, dark):
    darkvec = dark.reshape(img.shape[0] * img.shape[1])
    imvec = img.reshape(img.shape[0] * img.shape[1], 3)
    num_top_pixels = int(img.shape[0] * img.shape[1] / 1000)
    sort_indices = darkvec.argsort()[::-1]

    atm_light = imvec[sort_indices[:num_top_pixels]].mean(axis=0)
    # atm_light = imvec[sort_indices[0]] # Use max instead of mean

    return atm_light


def estimate_transmission(img, atm_light, omega, patch_size):
    normalized_img = np.empty(img.shape, img.dtype)

    for channel in range(3):
        normalized_img[:, :, channel] = img[:, :, channel] / atm_light[channel]

    transmission = 1 - omega*get_dark_channel(normalized_img, patch_size)
    return transmission


def soft_matting(img, transmission, radius=60):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    transmission = np.float32(transmission)
    gray = np.float32(gray)/255
    eps = 0.0001
    transmission_refined = guidedFilter(img, transmission, radius, eps)
    return transmission_refined


def recover(img, atm_light, transmission, t0=0.1):
    dehazed = np.empty(img.shape, img.dtype)

    for channel in range(3):
        dehazed[:, :, channel] = ((img[:, :, channel] - atm_light[channel]) / cv2.max(transmission, t0)
                                  + atm_light[channel])
    
    return dehazed

if __name__ == '__main__':
    src = cv2.imread('img/realworld/BJ_Bing_185.jpeg')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = src / 255
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    dark = get_dark_channel(img, 15)
    atm_light = estimate_atm_light(img, dark)
    transmission = estimate_transmission(img, atm_light, 0.95, 15)
    transmission_refined = soft_matting(src, transmission)
    dehazed = recover(img, atm_light, transmission_refined)
    plt.figure(figsize=(10, 10))
    plt.imshow(dehazed)
    plt.show()

    