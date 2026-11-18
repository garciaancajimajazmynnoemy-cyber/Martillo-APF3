import os
import re
import sys
import time
import cv2

# ---------------------------------------------------------------
# DETECTOR DE MARTILLO - V1 (simple, rectángulo exacto de la detección)
# Para un modelo entrenado con muestras de 48 x 24 (el martillo es más ancho que alto, horizontal).
# Si algo no funciona, usa Detector_Martillo_V2.py (tiene diagnóstico).
# ---------------------------------------------------------------

IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

ARCHIVO_MODELO = "cascade.xml"
ETIQUETA = "Martillo"
MUESTRA_ENTRENADA = (48, 24)   # (ancho, alto) de las positivas

# Parámetros de detección (ajústalos según tus pruebas):
SCALE_FACTOR = 1.05       # más cercano a 1 = más sensible pero más lento
MIN_NEIGHBORS = 3         # más alto = menos falsos positivos, pero puede perder detecciones
MIN_SIZE = MUESTRA_ENTRENADA   # nunca menor que la ventana del modelo

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def ventana_modelo(ruta):
    try:
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            cab = f.read(4000)
    except Exception:
        return None
    w = re.search(r"<width>\s*(\d+)\s*</width>", cab)
    h = re.search(r"<height>\s*(\d+)\s*</height>", cab)
    return (int(w.group(1)), int(h.group(1))) if (w and h) else None


def elegir_modelo():
    """cascade.xml más reciente cuya ventana coincide con la muestra; o el argumento dado."""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    rutas = []
    for raiz, _, archivos in os.walk(CARPETA_SCRIPT):
        if ARCHIVO_MODELO in archivos:
            rutas.append(os.path.join(raiz, ARCHIVO_MODELO))
    if not rutas:
        return None
    buenas = [r for r in rutas if ventana_modelo(r) == MUESTRA_ENTRENADA] or rutas
    buenas.sort(key=os.path.getmtime, reverse=True)
    return buenas[0]


ruta_modelo = elegir_modelo()
if ruta_modelo is None:
    print(f"ERROR: no se encontró '{ARCHIVO_MODELO}'. Se buscó en: {CARPETA_SCRIPT}")
    raise SystemExit
print("Modelo cargado desde:", ruta_modelo, "| ventana:", ventana_modelo(ruta_modelo))

clasificador = cv2.CascadeClassifier(ruta_modelo)
if clasificador.empty():
    print(f"ERROR: OpenCV no pudo leer '{ruta_modelo}'.")
    raise SystemExit

cap = None
for url in URLS_PRUEBA:
    print("Probando:", url)
    prueba = cv2.VideoCapture(url)
    if prueba.isOpened():
        ok, _ = prueba.read()
        if ok:
            print("CONECTADO con:", url)
            cap = prueba
            break
    prueba.release()

if cap is None:
    print("No se pudo conectar con DroidCam. Probando la webcam de la PC...")
    prueba = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if prueba.isOpened():
        cap = prueba

if cap is None:
    print("ERROR: no se pudo abrir ninguna cámara.")
    raise SystemExit

while True:

    ret, frame = cap.read()
    if not ret:
        print("No se recibe imagen de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    martillos = clasificador.detectMultiScale(gray,
                                            scaleFactor=SCALE_FACTOR,
                                            minNeighbors=MIN_NEIGHBORS,
                                            minSize=MIN_SIZE)

    for (x, y, w, h) in martillos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, ETIQUETA, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Detecciones: {len(martillos)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
