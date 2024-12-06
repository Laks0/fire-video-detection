from Color_functions import coincidencia
import numpy as np
from skimage import color, measure
import cv2

#########################
# Píxeles en movimiento #
#########################

def threshold(D, cota_inferior=10):    # Basado en ecuación (12)
  return max(np.mean(D) + np.std(D), cota_inferior)

def computar_FD(L, L_prev):
  F = abs(L - L_prev)
  T_fd = threshold(F)
  return (F >= T_fd).astype('int8')

def computar_BD(L, BG_prev):
  B = abs(L - BG_prev)
  T_bd = threshold(B)
  return (B >= T_bd).astype('int8')
    
def actualizar_SI(fd, si_prev):
  return np.where(fd==0, si_prev+1, 0)

# T_si = cantidad de frames que el sistema puede procesar por segundo. En la función que procesa los videos, lo defino como los FPS del input
def actualizar_BG(L, BG_prev, SI, T_si):
  return np.where(SI == T_si, L, BG_prev)
    
def matriz_de_movimiento(video_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error: Could not open video.")
      exit()

  fps_video = cap.get(cv2.CAP_PROP_FPS)

  # Leer el primer frame e inicializar las estructuras
  ret, frame_1 = cap.read()
  if not ret:
      print("Error: Could not read the first frame.")
      exit()
      
  L_1 = color.rgb2lab(frame_1[:,:,:3])[:,:,0]
  SI_1 = np.zeros(L_1.shape)
  BG_1 = np.zeros(L_1.shape)

  M_t = []    # Armo como una listita para los M que se van calculando. 
  while True:
      # Leer segundo frame
      ret, frame_2 = cap.read()
      if not ret:  # Salir si terminó el video
          break

      frame_2 = frame_2.astype(np.float32)/255
      frame_2_lab = color.rgb2lab(frame_2[:,:,:3])
      L_2 = frame_2_lab[:,:,0]

      # Calculo el FD y BD
      FD = computar_FD(L_2, L_1)
      BD = computar_BD(L_2, BG_1)

      M = FD | BD
      M_t.append(M)

      SI_2 = actualizar_SI(FD, SI_1)
      BG_2 = actualizar_BG(L_2, BG_1, SI_2, fps_video)

      # Siguiente par de frames, actualizo frame2 para que sea el frame previo ahora.
      frame_1 = frame_2
      L_1 = L_2
      SI_1 = SI_2
      BG_1 = BG_2

  return M_t
    
#######################
# Componentes conexas #
#######################

# Recibe una imágen segmentada y una máscara y retorna el número del segmento más parecido a la máscara
def segmento_mas_parecido(segmentos, mascara):
    coincidencias = np.bincount(segmentos[mascara])
    if coincidencias.shape[0] == 1:
        return 0
    return np.argmax(coincidencias[1:]) - 1

def frames_en_crecimiento(seg, t, areas_t, segmentos_t, tiempo_buffer, coincidencia_minima, segmento_similar):
    id_segmento = seg
    frames_en_crecimiento = 0
    for dt in range(1, tiempo_buffer+1):
        mascara = segmentos_t[t-dt+1] == id_segmento
        nueva_id = segmento_similar[t-dt+1][id_segmento]

        # El segmento ya no existe más
        if nueva_id == 0 or coincidencia(segmentos_t[t-dt] == nueva_id, mascara) < coincidencia_minima:
            frames_en_crecimiento += 1
            break

        if areas_t[t-dt][nueva_id] < areas_t[t-dt+1][id_segmento]:
            frames_en_crecimiento += 1
        id_segmento = nueva_id
    return frames_en_crecimiento

# Recibe los frames de un video binarizados y filtra las componentes que crecieron en [min_crecimiento]% de los últimos [tiempo_buffer] frames
# Coincidera ruido todas las componentes de área menor a [min_area]
# Elimina los primeros [tiempo_buffer] frames del video
def componentes_que_crecen(img_t, tiempo_buffer=10, min_crecimiento=.4, min_area=5, coincidencia_minima=.5):
    img_shape = img_t.shape
    res_t = np.full((img_shape[0]-tiempo_buffer, img_shape[1], img_shape[2]), False)

    segmentos_t = np.array([measure.label(x) for x in img_t])
    areas_t = [np.array([np.count_nonzero(segmentos == seg) for seg in range(np.max(segmentos)+1)]) for segmentos in segmentos_t]

    # segmento_similar[t][seg] es el segmento equivalente a seg de tiempo t en el tiempo t-1
    segmento_similar = [
        np.array([
            segmento_mas_parecido(segmentos_t[t-1], segmentos_t[t] == seg) for seg in range(np.max(segmentos_t[t])+1)
        ]) 
        for t in range(1, len(segmentos_t))]
    segmento_similar.insert(0, [0])
    print("Preprocesados", len(segmento_similar), "frames")

    for t, segmentos in enumerate(segmentos_t):
        if t < tiempo_buffer:
            continue
        if t % 10 == 0:
            print("Procesado frame", t)
        for seg, area in enumerate(areas_t[t]):
            if seg == 0:
                continue
            if (area < min_area):
                continue
                
            crecimiento = frames_en_crecimiento(seg, t, areas_t, segmentos_t, tiempo_buffer, coincidencia_minima, segmento_similar)
            
            if crecimiento/tiempo_buffer > min_crecimiento:
                res_t[t-tiempo_buffer] = res_t[t-tiempo_buffer] | (segmentos == seg)

    print("Listo :)")
    return res_t
