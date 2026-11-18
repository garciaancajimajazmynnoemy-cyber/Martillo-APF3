import os
import re
import sys
import threading
import time
import cv2
import numpy as np


def cargar_clasificador(ruta):
    """Carga el cascade.xml aunque la ruta tenga tildes o ñ (OpenCV en Windows falla con eso)."""
    clasif = cv2.CascadeClassifier(ruta)
    if not clasif.empty():
        return clasif
    import shutil
    import tempfile
    copia = os.path.join(tempfile.gettempdir(), "cascade_copia_ascii.xml")
    try:
        shutil.copyfile(ruta, copia)
        return cv2.CascadeClassifier(copia)
    except Exception:
        return clasif

# ---------------------------------------------------------------
# DETECTOR DE MARTILLO - V2 (con diagnóstico, sin barras de ajuste)
# Modelo entrenado con muestras de 48 x 24 (el martillo es más ancho que alto, horizontal).
#
# Base fluida: hilo de cámara + hilo de detección.
#
# Mejoras incluidas:
#   - Si hay varios cascade.xml en las carpetas, elige el MÁS RECIENTE cuya ventana coincide
#     con 48 x 24 (así no se carga por error un modelo viejo de otro objeto).
#     También puedes indicarlo: python Detector_Martillo_V2.py ruta\\cascade.xml
#   - Muestra en pantalla la ventana del modelo, el número de etapas y el tiempo de detección.
#   - Si el hilo de detección falla, el error se imprime y se ve en pantalla.
#   - Tecla W: cambia la resolución con la que se detecta (640 / 960 / 1280 / completa).
#   - Tecla D: modo diagnóstico (minNeighbors = 1 y muestra todos los candidatos).
#   - Tecla E: ecualizar histograma (ayuda con poca luz).
#
# Teclas:  R = girar | F = pantalla completa | + / - = minNeighbors | W = resolución
#          D = diagnóstico | E = ecualizar | ESC = salir
# ---------------------------------------------------------------

# ---------------- PARÁMETROS ----------------
IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

ARCHIVO_MODELO = "cascade.xml"
ETIQUETA = "Martillo"
MUESTRA_ENTRENADA = (48, 24)   # (ancho, alto) de las positivas; debe coincidir con el entrenamiento

SCALE_FACTOR = 1.05       # pasos de escala finos = más sensible (1.1 es más rápido)
MIN_NEIGHBORS = 3         # sube con '+' si salen falsos positivos, baja con '-' si no detecta
PESO_MIN = 0.0            # sin filtro por peso
MAX_CUADROS = 2           # máximo de detecciones mostradas (sube si hay varios martillos)
MARGEN = 0.05             # el recuadro se agranda 5 % alrededor de la detección

ANCHOS_PROCESO = [960, 640, 1280, None]   # resoluciones de detección (None = completa); tecla W
MIN_SIZE_REAL = None      # None = usa la ventana del modelo como mínimo; o (ancho, alto) en px reales

ECUALIZAR_INICIAL = False

PANTALLA_COMPLETA = True
GIROS = [
    ("sin girar", None),
    ("90 grados derecha", cv2.ROTATE_90_CLOCKWISE),
    ("90 grados izquierda", cv2.ROTATE_90_COUNTERCLOCKWISE),
]
# --------------------------------------------


def tamano_pantalla():
    """Tamaño de la pantalla de Windows; si falla, 1920 x 1080."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


SW, SH = tamano_pantalla()
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------
def info_modelo(ruta):
    """Lee del cascade.xml: (ancho, alto) de la ventana, número de etapas y tipo de rasgos."""
    try:
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return None, 0, "?"
    cab = txt[:4000]
    w = re.search(r"<width>\s*(\d+)\s*</width>", cab)
    h = re.search(r"<height>\s*(\d+)\s*</height>", cab)
    tipo = re.search(r"<featureType>\s*(\w+)\s*</featureType>", cab)
    etapas = len(re.findall(r"<stageThreshold>", txt))
    ventana = (int(w.group(1)), int(h.group(1))) if (w and h) else None
    return ventana, etapas, (tipo.group(1) if tipo else "?")


def elegir_modelo(nombre, muestra, explicito=None):
    """Elige el cascade.xml más reciente cuya ventana coincide con la muestra entrenada."""
    if explicito and os.path.isfile(explicito):
        return explicito
    rutas = []
    for raiz, _, archivos in os.walk(CARPETA_SCRIPT):
        if nombre in archivos:
            rutas.append(os.path.join(raiz, nombre))
    ruta_cwd = os.path.join(os.getcwd(), nombre)
    if os.path.isfile(ruta_cwd) and os.path.abspath(ruta_cwd) not in [os.path.abspath(r) for r in rutas]:
        rutas.append(ruta_cwd)
    if not rutas:
        return None
    infos = []
    print("Modelos encontrados:")
    for r in rutas:
        v, e, t = info_modelo(r)
        fecha = time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(r)))
        print(f"  - {r} | ventana {v} | {e} etapas | modificado {fecha}")
        infos.append((r, v))
    exactos = [x for x in infos if x[1] == tuple(muestra)]
    parecidos = [x for x in infos if x[1] and abs(x[1][0] / x[1][1] - muestra[0] / muestra[1]) <= 0.15]
    lista = exactos or parecidos or infos
    lista.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    return lista[0][0]


explicito = sys.argv[1] if len(sys.argv) > 1 else None
ruta_modelo = elegir_modelo(ARCHIVO_MODELO, MUESTRA_ENTRENADA, explicito)
if ruta_modelo is None:
    print(f"ERROR: no se encontró '{ARCHIVO_MODELO}'. Se buscó en: {CARPETA_SCRIPT}")
    raise SystemExit
print("Modelo cargado desde:", ruta_modelo)

clasificador = cargar_clasificador(ruta_modelo)
if clasificador.empty():
    print(f"ERROR: se encontró '{ruta_modelo}' pero OpenCV no pudo leerlo.")
    raise SystemExit

VENTANA, N_ETAPAS, TIPO_RASGOS = info_modelo(ruta_modelo)
if VENTANA is None:
    VENTANA = MUESTRA_ENTRENADA
    print("AVISO: no se pudo leer la ventana del modelo; se asume", VENTANA)
print(f"Modelo: ventana {VENTANA[0]}x{VENTANA[1]} | {N_ETAPAS} etapas | rasgos {TIPO_RASGOS}")

AVISO_MODELO = ""
if abs(VENTANA[0] / VENTANA[1] - MUESTRA_ENTRENADA[0] / MUESTRA_ENTRENADA[1]) > 0.15:
    AVISO_MODELO = f"OJO: modelo {VENTANA[0]}x{VENTANA[1]} (se esperaba {MUESTRA_ENTRENADA[0]}x{MUESTRA_ENTRENADA[1]})"
    print("*" * 70)
    print(AVISO_MODELO)
    print("Puede ser un cascade.xml viejo (de otro objeto) o se entrenó con otro tamaño de muestra.")
    print("Reentrena con 'Sample width' 48 y 'Sample height' 24, o cambia MUESTRA_ENTRENADA.")
    print("*" * 70)
if 0 < N_ETAPAS < 8:
    print(f"AVISO: el modelo tiene solo {N_ETAPAS} etapas; es poco y puede dar muchos falsos positivos o no detectar.")

# ---------------------------------------------------------------
# CÁMARA
# ---------------------------------------------------------------
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

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class LectorCamara:
    """Lee la cámara en un hilo aparte y guarda solo el cuadro más reciente."""

    def __init__(self, captura):
        self.captura = captura
        self.frame = None
        self.n = 0
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        while self.ok:
            ret, f = self.captura.read()
            if not ret:
                self.ok = False
                break
            with self.lock:
                self.frame = f
                self.n += 1

    def ultimo(self):
        with self.lock:
            return self.frame, self.n


lector = LectorCamara(cap)

# Estado compartido entre el hilo principal y el de detección
estado = {
    "salir": False,
    "giro": 0,
    "min_neighbors": MIN_NEIGHBORS,
    "diag": False,
    "ecualizar": ECUALIZAR_INICIAL,
    "ancho_idx": 0,
    "resultado": (0, []),     # (giro con el que se calculó, [(x, y, w, h, peso)])
    "ms": 0.0,                # lo que tarda cada detección (prueba de que el hilo está vivo)
    "error": "",
}


def detectar(gris, min_neighbors, min_size):
    try:
        rects, _, pesos = clasificador.detectMultiScale3(
            gris,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=min_neighbors,
            minSize=min_size,
            outputRejectLevels=True,
        )
        pesos = np.array(pesos).flatten()
        if len(pesos) != len(rects):
            pesos = np.zeros(len(rects))
    except cv2.error:
        # Respaldo: si detectMultiScale3 no funciona con este modelo, se usa el método simple
        rects = clasificador.detectMultiScale(
            gris,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        pesos = np.zeros(len(rects))
    return [(int(x), int(y), int(w), int(h), float(p))
            for (x, y, w, h), p in zip(rects, pesos)]


def trabajador_deteccion():
    ultimo_proc = -1
    while not estado["salir"]:
        frame, n = lector.ultimo()
        if frame is None or n == ultimo_proc:
            time.sleep(0.005)
            continue
        ultimo_proc = n

        try:
            t0 = time.time()
            giro_id = estado["giro"]
            diag = estado["diag"]
            min_neighbors = 1 if diag else estado["min_neighbors"]
            maximo = 10 if diag else MAX_CUADROS

            codigo = GIROS[giro_id][1]
            rot = cv2.rotate(frame, codigo) if codigo is not None else frame
            h, w = rot.shape[:2]
            ancho_proc = ANCHOS_PROCESO[estado["ancho_idx"]]
            escala = 1.0 if ancho_proc is None else min(1.0, ancho_proc / w)   # solo se reduce
            if escala < 1.0:
                pequeno = cv2.resize(rot, (int(w * escala), int(h * escala)),
                                     interpolation=cv2.INTER_AREA)
            else:
                pequeno = rot
            gris = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
            if estado["ecualizar"]:
                gris = cv2.equalizeHist(gris)

            # El mínimo nunca puede ser menor que la ventana con que se entrenó el modelo
            if MIN_SIZE_REAL is None:
                min_size = (VENTANA[0], VENTANA[1])
            else:
                min_size = (max(int(MIN_SIZE_REAL[0] * escala), VENTANA[0]),
                            max(int(MIN_SIZE_REAL[1] * escala), VENTANA[1]))

            finales = []
            for (x, y, dw, dh, p) in detectar(gris, min_neighbors, min_size):
                if p >= PESO_MIN:
                    finales.append((int(x / escala), int(y / escala),
                                    int(dw / escala), int(dh / escala), p))
            finales.sort(key=lambda d: d[4], reverse=True)
            estado["resultado"] = (giro_id, finales[:maximo])
            estado["ms"] = (time.time() - t0) * 1000
            estado["error"] = ""
        except Exception as e:
            estado["error"] = f"{type(e).__name__}: {e}"
            print("ERROR en la detección:", estado["error"])
            time.sleep(0.5)


threading.Thread(target=trabajador_deteccion, daemon=True).start()

# ---------------------------------------------------------------
# VENTANA (sin barras)
# ---------------------------------------------------------------
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
if PANTALLA_COMPLETA:
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

giro_actual = 0
completa = PANTALLA_COMPLETA
ultimo_n = -1
lienzo = None
dims_previas = None
fps = 0.0
t_anterior = time.time()

while True:
    frame, n = lector.ultimo()

    if frame is not None and n != ultimo_n:
        ultimo_n = n
        estado["giro"] = giro_actual

        # Girar y reducir la imagen al tamaño de pantalla ANTES de dibujar
        nombre_giro, codigo = GIROS[giro_actual]
        rot = cv2.rotate(frame, codigo) if codigo is not None else frame
        h, w = rot.shape[:2]
        esc = min(SW / w, SH / h) if completa else min(1.0, (SW * 0.9) / w, (SH * 0.8) / h)
        vista = cv2.resize(rot, (max(int(w * esc), 1), max(int(h * esc), 1)))
        vh, vw = vista.shape[:2]

        # Dibujar las detecciones calculadas por el otro hilo
        diag = estado["diag"]
        color = (0, 255, 255) if diag else (0, 255, 0)
        giro_res, lista = estado["resultado"]
        if giro_res == giro_actual:
            for (x, y, dw, dh, p) in lista:
                x, y, dw, dh = int(x * esc), int(y * esc), int(dw * esc), int(dh * esc)
                mx, my = int(dw * MARGEN), int(dh * MARGEN)
                x1, y1 = max(x - mx, 0), max(y - my, 0)
                x2, y2 = min(x + dw + mx, vw - 1), min(y + dh + my, vh - 1)
                cv2.rectangle(vista, (x1, y1), (x2, y2), color, 3)
                cv2.putText(vista, f"{ETIQUETA} {p:.1f}", (x1, max(y1 - 12, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            n_cuadros = len(lista)
        else:
            n_cuadros = 0

        ahora = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(ahora - t_anterior, 1e-6))
        t_anterior = ahora
        ancho_proc = ANCHOS_PROCESO[estado["ancho_idx"]]
        modo = "  [DIAGNOSTICO]" if diag else ""
        cv2.putText(vista, f"FPS: {fps:.0f}  Detecciones: {n_cuadros}  minNeighbors: "
                           f"{1 if diag else estado['min_neighbors']}{modo}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vista, f"Modelo {VENTANA[0]}x{VENTANA[1]} | {N_ETAPAS} etapas | {TIPO_RASGOS} | "
                           f"deteccion {estado['ms']:.0f} ms | proceso: {ancho_proc or 'completa'} px | "
                           f"ecualizar: {'SI' if estado['ecualizar'] else 'NO'}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        if AVISO_MODELO:
            cv2.putText(vista, AVISO_MODELO, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if estado["error"]:
            cv2.putText(vista, "ERROR: " + estado["error"][:80], (10, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vista, "R girar | F pantalla | +/- sensib. | W resolucion | D diagnostico | E ecualizar | ESC salir",
                    (10, vh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        if completa:
            # Lienzo reutilizado (no se crea uno nuevo en cada cuadro)
            if lienzo is None or lienzo.shape[:2] != (SH, SW):
                lienzo = np.zeros((SH, SW, 3), dtype=np.uint8)
            if dims_previas != (vh, vw):
                lienzo[:] = 0
                dims_previas = (vh, vw)
            y0, x0 = (SH - vh) // 2, (SW - vw) // 2
            lienzo[y0:y0 + vh, x0:x0 + vw] = vista
            cv2.imshow("frame", lienzo)
        else:
            cv2.imshow("frame", vista)
    else:
        if not lector.ok:
            print("Se perdió la conexión con la cámara.")
            break
        time.sleep(0.003)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k in (ord('r'), ord('R')):
        giro_actual = (giro_actual + 1) % len(GIROS)
    elif k in (ord('f'), ord('F')):
        completa = not completa
        dims_previas = None
        cv2.setWindowProperty(
            "frame", cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if completa else cv2.WINDOW_NORMAL)
    elif k in (ord('+'), ord('=')):
        estado["min_neighbors"] = min(estado["min_neighbors"] + 1, 100)
    elif k in (ord('-'), ord('_')):
        estado["min_neighbors"] = max(estado["min_neighbors"] - 1, 1)
    elif k in (ord('w'), ord('W')):
        estado["ancho_idx"] = (estado["ancho_idx"] + 1) % len(ANCHOS_PROCESO)
        print("Resolución de detección:", ANCHOS_PROCESO[estado["ancho_idx"]] or "completa")
    elif k in (ord('d'), ord('D')):
        estado["diag"] = not estado["diag"]
        print("Modo diagnóstico:", "ACTIVADO" if estado["diag"] else "desactivado")
    elif k in (ord('e'), ord('E')):
        estado["ecualizar"] = not estado["ecualizar"]
        print("Ecualizar histograma:", "SI" if estado["ecualizar"] else "NO")

estado["salir"] = True
lector.ok = False
time.sleep(0.1)
cap.release()
cv2.destroyAllWindows()
