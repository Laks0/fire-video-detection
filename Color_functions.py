import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage import color
import cv2

buckets = 24

def mascaras(lab_img, prob_la, prob_lb, prob_ab, alpha=0.0016):
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


# Genera una lista con los frames del video en formato RGB
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

  F_t = []
  while True:
      frame_rgb = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2RGB)
      F_t.append(frame_rgb)
      
      # Leer segundo frame
      ret, frame = cap.read()
      if not ret:  # Salir si terminó el video
          break

  return F_t


######### Funciones que devuelve distintas máscaras generadas sobre los frames del video

# Genera una lista con la segmentación de Otsu sobre la componente a de L*a*b* de la imagen
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

  F_t = []
  while True:
      a = color.rgb2lab(frame[:,:,:3])[:,:,1]
      _, F = otsu_simple(a)
      F_t.append(F)
      
      # Leer segundo frame
      ret, frame = cap.read()
      if not ret:  # Salir si terminó el video
          break

  return F_t


# Devuelve una lista F_t donde cada elemento es una lista: 
# F_t[0] = R1 & R2 & R3 & R4 & R5   y   F_t[1] = R1 & R2 & R3 & R4
def video_mascaras_R(video_path, prob_la, prob_lb, prob_ab):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error: Could not open video.")
      exit()

  # Leer el primer frame e inicializar las estructuras
  ret, frame = cap.read()
  if not ret:
      print("Error: Could not read the first frame.")
      exit()

  F_t = []
  while True:
      frame_rgb = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2RGB)
      frame_lab = color.rgb2lab(frame_rgb)
      R1, R2, R3, R4, R5 = mascaras(frame_lab, prob_la, prob_lb, prob_ab)
	  
      F_t.append([R1&R2&R3&R4&R5, R1&R2&R3&R4])
      
      # Leer segundo frame
      ret, frame = cap.read()
      if not ret:  # Salir si terminó el video
          break

  return F_t


# Devuelve una lista con máscaras que se componen de R5 junto con la segmentación de Otsu de la componente a
def video_otsu_a_R5(video_path, prob_la, prob_lb, prob_ab):      
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error: Could not open video.")
      exit()

  # Leer el primer frame e inicializar las estructuras
  ret, frame = cap.read()
  if not ret:
      print("Error: Could not read the first frame.")
      exit()

  F_t = []
  while True:
      frame_rgb = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2RGB)
      frame_lab = color.rgb2lab(frame_rgb)
      _, _, _, _, R5 = mascaras(frame_lab, prob_la, prob_lb, prob_ab)

      a = frame_lab[:,:,1]
      _, F = otsu_simple(a)

      F_t.append(F&R5)
      
      # Leer segundo frame
      ret, frame = cap.read()
      if not ret:  # Salir si terminó el video
          break

  return F_t


