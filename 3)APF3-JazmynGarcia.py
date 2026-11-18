"""
APF3 - Visión Computacional aplicada a Ingeniería Civil (UTP)
Autor : Jazmyn Noemy Garcia Ancajima
Objeto: Martillo de uña (herramienta de obra)
Método: Haar Cascade entrenado con Cascade Trainer GUI (ventana 48x24, 15 etapas)

Uso:
    python APF3-JazmynGarcia.py            -> cámara DroidCam (celular)
    python APF3-JazmynGarcia.py 0          -> webcam de la PC
    python APF3-JazmynGarcia.py foto.jpg   -> imagen o video guardado

Teclas: ESC = salir | C = marcar detección Correcta | F = Falso positivo
        N = objeto No detectado (falso negativo) | G = guardar captura
Barras: minNeighbors (más alto = menos falsos positivos), scale x10.
"""
import os
import sys
import time
import threading

import cv2

# ---------------- CONFIGURACIÓN ----------------
IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_CAMARA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]
ARCHIVO_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascade.xml")
ETIQUETA = "Martillo"
MIN_NEIGHBORS_INICIAL = 3
SCALE_X10_INICIAL = 11
ANCHO_PROCESO = 960      # ancho al que se reduce el cuadro para detectar (más fluido)
MARGEN_X, MARGEN_Y = 0.05, 0.05   # margen visual del recuadro dibujado


def cargar_modelo(ruta=ARCHIVO_MODELO):
    modelo = cv2.CascadeClassifier(ruta)
    if modelo.empty():
        raise FileNotFoundError(f"No se pudo cargar '{ruta}'. Debe estar junto a este script.")
    return modelo


def tamano_ventana(ruta=ARCHIVO_MODELO):
    """Lee el tamaño de la ventana de entrenamiento (ancho, alto) desde cascade.xml."""
    fs = cv2.FileStorage(ruta, cv2.FILE_STORAGE_READ)
    ancho = int(fs.getNode("cascade").getNode("width").real())
    alto = int(fs.getNode("cascade").getNode("height").real())
    fs.release()
    return ancho, alto


def detectar(modelo, frame, min_neighbors, scale_factor, min_size, ancho_proceso=ANCHO_PROCESO):
    """Devuelve lista de (x, y, w, h) en coordenadas del cuadro original."""
    alto, ancho = frame.shape[:2]
    esc = min(1.0, ancho_proceso / ancho)
    peq = cv2.resize(frame, (int(ancho * esc), int(alto * esc))) if esc < 1 else frame
    gris = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
    ms = (max(int(min_size[0] * esc), 8), max(int(min_size[1] * esc), 8))
    cajas = modelo.detectMultiScale(gris, scaleFactor=scale_factor,
                                    minNeighbors=min_neighbors, minSize=ms)
    return [(int(x / esc), int(y / esc), int(w / esc), int(h / esc)) for x, y, w, h in cajas]


def dibujar(frame, cajas):
    alto, ancho = frame.shape[:2]
    for x, y, w, h in cajas:
        x1, y1 = max(x - int(w * MARGEN_X), 0), max(y - int(h * MARGEN_Y), 0)
        x2, y2 = min(x + w + int(w * MARGEN_X), ancho - 1), min(y + h + int(h * MARGEN_Y), alto - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, ETIQUETA, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


class Lector(threading.Thread):
    """Lee la cámara en un hilo aparte para evitar retraso acumulado (video fluido)."""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.frame, self.ok, self.activo = cap, None, False, True
        self.start()

    def run(self):
        while self.activo:
            self.ok, self.frame = self.cap.read()
            if not self.ok:
                time.sleep(0.05)


def abrir_fuente(arg):
    if arg is not None:
        fuente = int(arg) if arg.isdigit() else arg
        cap = cv2.VideoCapture(fuente)
        if cap.isOpened():
            return cap
        raise SystemExit(f"No se pudo abrir la fuente: {arg}")
    for url in URLS_CAMARA:
        print("Probando:", url)
        cap = cv2.VideoCapture(url)
        if cap.isOpened() and cap.read()[0]:
            print("CONECTADO con:", url)
            return cap
        cap.release()
    raise SystemExit("No se pudo conectar con DroidCam. Revise la IP, el WiFi y la app abierta.")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    modelo = cargar_modelo()
    ventana = tamano_ventana()
    print("Ventana del modelo (ancho, alto):", ventana)

    es_imagen = arg is not None and arg.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    nombre = "APF3 - Detector de Martillo"
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("minNeighbors", nombre, MIN_NEIGHBORS_INICIAL, 100, lambda _: None)
    cv2.createTrackbar("scale x10", nombre, SCALE_X10_INICIAL, 30, lambda _: None)

    lector = None
    if es_imagen:
        imagen = cv2.imread(arg)
        if imagen is None:
            raise SystemExit(f"No se pudo leer la imagen: {arg}")
    else:
        cap = abrir_fuente(arg)
        lector = Lector(cap)

    cont = {"C": 0, "F": 0, "N": 0}
    n_guardadas, t0, fps = 0, time.time(), 0.0
    while True:
        if es_imagen:
            frame = imagen.copy()
        else:
            if not lector.ok or lector.frame is None:
                time.sleep(0.01)
                continue
            frame = lector.frame.copy()

        mn = cv2.getTrackbarPos("minNeighbors", nombre)
        sc = max(cv2.getTrackbarPos("scale x10", nombre), 11) / 10
        cajas = detectar(modelo, frame, mn, sc, ventana)
        limpio = frame.copy()
        dibujar(frame, cajas)

        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - t0, 1e-6)
        t0 = ahora
        cv2.putText(frame, f"FPS {fps:.0f} | Det {len(cajas)} | C={cont['C']} F={cont['F']} N={cont['N']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(nombre, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k in (ord("c"), ord("C")):
            cont["C"] += 1
        elif k in (ord("f"), ord("F")):
            cont["F"] += 1
        elif k in (ord("n"), ord("N")):
            cont["N"] += 1
        elif k in (ord("g"), ord("G")):
            n_guardadas += 1
            cv2.imwrite(f"captura_{n_guardadas}.jpg", frame)
            print("Guardada captura", n_guardadas)

    print("RESUMEN  Correctas:", cont["C"], "| Falsos positivos:", cont["F"], "| No detectados:", cont["N"])
    if lector:
        lector.activo = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
