import numpy as np
import pandas as pd
from skimage import filters

def mascaras(lab_img):
    L, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]
    L_m = np.mean(L)
    a_m = np.mean(a)
    b_m = np.mean(b)

    # R1(x,y) = 1 si el valor de L del pixel en esa posición era mayor al promedio de L* de la imagen, 0 si no.
    R_1 = L >= L_m

    # R2(x,y) = 1 si el valor de a* del pixel en esa posición era mayor al promedio de a* de la imagen, 0 si no.
    R_2 = a >= a_m

    # R3(x,y) = 1 si el valor de b* del pixel en esa posición era mayor al promedio de b* de la imagen, 0 si no.
    R_3 = b >= b_m

    # R4(x,y) = 1 si el valor de b* del pixel en esa posición era mayor al valor de a* del pixel en esa posición, 0 si no.
    R_4 = b >= a

    return R_1, R_2, R_3, R_4

def otsu_segmentos(img):
    thresh = filters.threshold_multiosu(img)
    S1 = img < thresh[0]
    S2 = (img >= thresh[0]) & (img < thresh[1])
    S3 = img >= thresh[1]
    return S1, S2, S3

def coincidencia(M1, M2):
    return np.sum(M1 == M2) / (M1.shape[0] * M1.shape[1])