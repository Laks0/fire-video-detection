import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage import color
import cv2

buckets = 24

def mascaras(lab_img, prob_la, prob_lb, prob_ab, alpha):
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

    limitesL = np.arange(100/24, 100, 100/24)
    limitesab = np.arange(-128, 127, 256/24)
    
    indiceL = np.searchsorted(limitesL, L).flatten()
    indicea = np.searchsorted(limitesab, a).flatten()
    indiceb = np.searchsorted(limitesab, b).flatten()

    R_5 = prob_la[indiceL, indicea] * prob_lb[indiceL, indiceb] * prob_ab[indicea, indiceb]
    R_5 = R_5.reshape(R_1.shape) >= alpha

    return R_1, R_2, R_3, R_4, R_5

def otsu_segmentos(img):
    thresh = threshold_multiotsu(img)
    S1 = img < thresh[0]
    S2 = (img >= thresh[0]) & (img < thresh[1])
    S3 = img >= thresh[1]
    return S1, S2, S3

def otsu_simple(img):
    thresh = threshold_otsu(img)
    return (img < thresh), (img >= thresh)

def coincidencia(M1, M2):
    return np.sum(M1 == M2) / (M1.shape[0] * M1.shape[1])

def video_otsu_a(video_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error: Could not open video.")
      exit()

  # Leer el primer frame e inicializar las estructuras
  ret, frame = cap.read()
  if not ret:
      print("Error: Could not read the first frame.")
      exit()

  M_t = []
  while True:
      a = color.rgb2lab(frame[:,:,:3])[:,:,1]
      _, F = otsu_simple(a)
      M_t.append(F)
      
      # Leer segundo frame
      ret, frame = cap.read()
      if not ret:  # Salir si terminó el video
          break

  return M_t
    
def video_rgb(video_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error: Could not open video.")
      exit()

  # Leer el primer frame e inicializar las estructuras
  ret, frame = cap.read()
  if not ret:
      print("Error: Could not read the first frame.")
      exit()

  M_t = []
  while True:
      M_t.append(frame[:,:,:3])
      
      # Leer segundo frame
      ret, frame = cap.read()
      if not ret:  # Salir si terminó el video
          break

  return M_t