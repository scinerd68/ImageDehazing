import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter

def colorCorrect(img,u):
    '''
    perform color correction on BGR image (details in report)
    :param img: BGR Image
    :return: color corrected BGR image
    '''
    img = np.float64(img) / 255
    B_rmse = np.std(img[:,:,0])
    G_rmse = np.std(img[:,:,1])
    R_rmse = np.std(img[:,:,2])

    B_max = np.mean(img[:,:,0]) + u * B_rmse
    G_max = np.mean(img[:,:,1]) + u * G_rmse
    R_max = np.mean(img[:,:,2]) + u * R_rmse

    B_min = np.mean(img[:,:,0]) - u * B_rmse
    G_min = np.mean(img[:,:,1]) - u * G_rmse
    R_min = np.mean(img[:,:,2]) - u * R_rmse

    B_cr = (img[:,:,0]  - B_min) / (B_max - B_min)
    G_cr = (img[:,:,1]  - G_min) / (G_max - G_min)
    R_cr = (img[:,:,2]  - R_min) / (R_max - R_min)

    img_CR = cv2.merge([B_cr,G_cr,R_cr]) *255
    img_CR = np.clip(img_CR,0,255)
    img_CR = np.uint8(img_CR)

    return img_CR


def getMinChannel(img):
    '''
    get local min of single-channel image
    :param img: single-channel Image
    :return: local-min image that has the same shape as the input image
    '''
	imgGray = np.zeros((img.shape[0],img.shape[1]),np.float32)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			localMin = 255
			for k in range(2):
				if img.item((i,j,k)) < localMin:
					localMin = img.item((i,j,k))
			imgGray[i,j] = localMin
	return imgGray


def whiteBalance(img):
    '''
    perform white balancing on BGR Image (details in report)
    :param img: BGR Image
    :return: white balanced BGR image
    '''
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)

    for i in range(len(Ba)):
        for j in range(len(Ba[0])):
            Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
            Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
            Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


def visibilityRestore(img_path, s_v = 5, p = 0.95, color_correct = True):
    '''
    full pipeline of visibility restoration algorithm
    :param img_path: path to BGR Image
    :param s_v: Median filters' kernel size
    :param p: interpreted as the "percentage" of fog to be filtered, ranged 0 to 1, the higher, the
    higher the effect of restoration.
    :param color_correct: Boolean, whether to apply color correcting on the result image or not.
    :return: restored BGR image
    '''
    img = cv2.imread(img_path)

    img_wb =whiteBalance(img)

    W = getMinChannel(img_wb)

    A = cv2.medianBlur(np.uint8(W),s_v)
    B = W - A
    B = np.abs(B)
    B = A - cv2.medianBlur(np.uint8(B),s_v)
    max_255_img = np.ones(B.shape,dtype = np.uint8 ) * 255
    min_t = cv2.merge([np.uint8(p*B),np.uint8(W),max_255_img])
    min_t = getMinChannel(min_t)
    min_t[min_t<0] = 0
    V = np.uint8(min_t)
    # V = cv2.blur(V,(5,5))

    V = np.float32(V) / 255

    R_dehazy = np.zeros((V.shape[0],V.shape[1],3), dtype=np.float32)

    img_wb = np.float32(img_wb) / 255

    for i in range(3):
        R_dehazy[:,:,i] = (img_wb[:,:,i] - V) / (1 - V)
    R_dehazy = R_dehazy / R_dehazy.max()

    R_dehazy = np.clip(R_dehazy,0,1)
    R_dehazy = np.uint8(R_dehazy*255)

    if color_correct:
        R_dehazy = colorCorrect(R_dehazy,2)

    return R_dehazy
