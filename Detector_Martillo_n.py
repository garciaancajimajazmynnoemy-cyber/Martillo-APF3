import os
import time
import cv2


def guardar_jpg(ruta, img):
    """Guarda un JPG aunque la ruta tenga tildes, ñ o paréntesis (cv2.imwrite falla en Windows con eso)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False
    try:
        buf.tofile(ruta)
        return True
    except OSError as e:
        print("ERROR al guardar:", e)
        return False

# ---------------------------------------------------------------
# CAPTURA DE NEGATIVAS (imágenes SIN martillo) - con modo AUTOMÁTICO
# Guarda el cuadro COMPLETO (320 px de ancho). Así el entrenador puede sacar
# muchas ventanas de fondo de cada foto.
#
# Teclas (con la ventana de video seleccionada):
#     S = guardar una imagen
#     A = activar / desactivar captura AUTOMÁTICA (1 imagen cada 0.5 s)
#     ESC = salir
#
# Consejo: activa A y recorre el entorno apuntando a manos vacías, otras herramientas
# (destornillador, alicate, llave, cincel, serrucho, nivel), cables, cascos, cajas,
# mesas, paredes, pisos de obra, ladrillos, ropa...
# Lo importante: que NUNCA aparezca el martillo en la imagen.
# Las otras herramientas de mango largo y las manos son clave: así el modelo aprende a no
# confundir "un mango, un alicate o una mano" con "el martillo".
# ---------------------------------------------------------------
Datos = "n"
INTERVALO_AUTO = 0.5      # segundos entre imágenes en modo automático
ANCHO_GUARDADO = 320

IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), Datos)
os.makedirs(carpeta, exist_ok=True)
print("Las imágenes se guardarán en:", carpeta)

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

count = len([f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")])
auto = False
t_ultimo = 0.0


def guardar_imagen(img):
    """Reduce el cuadro completo y lo guarda. Devuelve True si se guardó."""
    global count
    alto, ancho = img.shape[:2]
    chica = cv2.resize(img, (ANCHO_GUARDADO, int(alto * ANCHO_GUARDADO / ancho)),
                       interpolation=cv2.INTER_AREA)
    ruta = os.path.join(carpeta, f"neg_{count}.jpg")
    if guardar_jpg(ruta, chica):
        count += 1
        return True
    print("ERROR: no se pudo guardar en", ruta)
    return False


print("Controles: [S] guardar | [A] automático | [ESC] salir")
print("Recuerda: que NO aparezca el martillo en la imagen.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("No se recibe imagen de la cámara. Fin.")
        break

    alto, ancho = frame.shape[:2]
    original = frame.copy()

    # Captura automática
    if auto and time.time() - t_ultimo >= INTERVALO_AUTO:
        if guardar_imagen(original):
            t_ultimo = time.time()

    color = (0, 0, 255) if auto else (0, 255, 255)
    cv2.rectangle(frame, (4, 4), (ancho - 5, alto - 5), color, 6)
    estado = "AUTOMATICO ACTIVADO (A para parar)" if auto else "Manual (A = automatico)"
    cv2.putText(frame, f"Carpeta {Datos}: {count} imagenes", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, estado, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "SIN martillo en la imagen | S guardar", (14, alto - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('s'), ord('S')):
        if guardar_imagen(original):
            print("Imagen guardada. Total:", count)
    elif k in (ord('a'), ord('A')):
        auto = not auto
        t_ultimo = 0.0
        print("Captura automática:", "ACTIVADA" if auto else "desactivada")
    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Total de negativas en la carpeta:", count)
