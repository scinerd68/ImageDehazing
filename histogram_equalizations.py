import cv2
import numpy as np

def histEqual_BGR(img_path):
    '''
    perform vanilla histogram equalization seperately on each of the 3 B-G-R color channels.
    :param img_path: String, path to BGR Image
    :return: transformed BGR image
    '''
    img = cv2.imread(img_path)
    bgr_planes = cv2.split(img)
    for i in range(3):
        bgr_planes[i] = cv2.equalizeHist(bgr_planes[i])
    he = cv2.merge(bgr_planes)
    return he

def histEqual_HSV(img_path, equalize_S = True, equalize_V = True):
    '''
    Convert BGR image to HSV image, perform vanilla histogram equalization
    on Saturation and/or Value channel of HSV image, then convert back to BGR image.

    :param img_path: String, path to BGR Image
    :param equalize_S: Boolean, if True, perform HE on Saturation channel
    :param equalize_V:  Boolean, if True, perform HE on Value channel
    :return: transformed BGR image
    '''
    img = cv2.imread(IMG)
    hsv_planes = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    if equalize_S:
        hsv_planes[1] = cv2.equalizeHist(hsv_planes[1])
    if equalize_V:
        hsv_planes[2] = cv2.equalizeHist(hsv_planes[2])

    he_hsv = cv2.merge(hsv_planes)
    he_hsv = cv2.cvtColor(he_hsv, cv2.COLOR_HSV2BGR)
    return he_hsv

def clahe_BGR(img_path):
    '''
    perform contrast limited adaptive histogram equalization (CLAHE) seperately
    on each of the 3 B-G-R color channels.
    :param img_path: String, path to BGR Image
    :return: transformed BGR image
    '''
    img = cv2.imread(img_path)
    bgr_planes = cv2.split(img)
    for i in range(3):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        bgr_planes[i] = clahe.apply(bgr_planes[i])
    clh = cv2.merge(bgr_planes)
    return clh

def clahe_HSV(img_path, clipLimit = 3.0, tileGridSize = (8,8), equalize_S = True, equalize_V = True):
    '''
    Convert BGR image to HSV image, perform CLAHE on Saturation and/or Value
    channel of HSV image, then convert back to BGR image.

    :param img_path: String, path to BGR Image
    :param equalize_S: Boolean, if True, perform CLAHE on Saturation channel
    :param equalize_V:  Boolean, if True, perform CLAHE on Value channel
    :return: transformed BGR image
    '''
    img = cv2.imread(img_path)
    hsv_planes = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    if equalize_S:
        hsv_planes[1] = clahe.apply(hsv_planes[1])
    if equalize_V:
        hsv_planes[2] = clahe.apply(hsv_planes[2])

    clh_hsv = cv2.merge(hsv_planes)
    clh_hsv = cv2.cvtColor(clh_hsv, cv2.COLOR_HSV2BGR)

    return clh_hsv
