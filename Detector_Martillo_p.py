import os
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
# CAPTURA DE POSITIVAS (imágenes CON el martillo)
# Un martillo (cabeza + mango) es largo y delgado, así que el recuadro es HORIZONTAL (2 : 1)
# y cada positiva se guarda de 48 x 24 px (ancho x alto).
#
# IMPORTANTE: este tamaño (48 x 24) debe ser el MISMO que pongas como
# "Sample width / Sample height" en Cascade Trainer GUI.
#
# Controles (con la ventana de video seleccionada):
#   S o ESPACIO = guardar la muestra
#   A / D       = mover el recuadro a la izquierda / derecha
#   W / X       = mover el recuadro hacia arriba / abajo
#   + / -       = agrandar / achicar el recuadro
#   M           = activar / desactivar guardar también la copia en espejo
#   ESC         = salir
#
# Consejo: encuadra SOLO el martillo, completo (de la punta del mango a la cabeza), con poco
# fondo alrededor. Mantenlo siempre HORIZONTAL, con la cabeza a un lado (si lo vas a detectar
# vertical, en el detector V2 puedes girar la imagen con la tecla R).
# Varía: ángulo ligero, distancia, iluminación, fondo, y la posición: sobre una mesa, en el
# piso, en la mano, en una caja de herramientas. Si tienes martillos distintos (de uña, de
# bola, de goma), inclúyelos.
# ---------------------------------------------------------------
Datos = "p"

IP_CELULAR = "192.168.1.37"
PUERTO = 4747
URLS_PRUEBA = [
    f"http://{IP_CELULAR}:{PUERTO}/video",
    f"http://{IP_CELULAR}:{PUERTO}/mjpegfeed",
    f"http://{IP_CELULAR}:{PUERTO}/videofeed",
]

RELACION = 2.0            # ancho / alto del recuadro (2 : 1, como la muestra de 48 x 24)
ANCHO_INICIAL = 0.35      # fracción del ancho de la imagen que ocupa el recuadro al iniciar
TAM_MUESTRA = (48, 24)    # (ancho, alto) con el que se guarda cada positiva
GUARDAR_ESPEJO = False    # True = guarda también la copia volteada (útil: así hay martillos con la cabeza a la izquierda y a la derecha)

# ---------------------------------------------------------------
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
    print("Revisa: app DroidCam abierta, misma red WiFi e IP correcta.")
    raise SystemExit

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ok, frame = cap.read()
if not ok:
    print("No se recibe imagen de la cámara.")
    raise SystemExit
alto_f, ancho_f = frame.shape[:2]

bw = int(ancho_f * ANCHO_INICIAL)   # ancho del recuadro
cx = ancho_f // 2                   # centro horizontal del recuadro
cy = alto_f // 2                    # centro vertical del recuadro

count = len([f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")])
print("Controles: [S/ESPACIO] guardar | [A]/[D] izq-der | [W]/[X] arriba-abajo | [+]/[-] tamaño | [M] espejo | [ESC] salir")


def guardar(img):
    global count
    ruta = os.path.join(carpeta, f"objeto_{count}.jpg")
    if guardar_jpg(ruta, img):
        print("Imagen guardada:", ruta)
        count += 1
    else:
        print("ERROR: no se pudo guardar en", ruta)


while True:
    ret, frame = cap.read()
    if not ret:
        print("No se recibe imagen de la cámara. Fin.")
        break

    imAux = frame.copy()

    # Recuadro horizontal, sin salirse de la imagen
    bw = max(min(bw, ancho_f, int(alto_f * RELACION)), 50)
    bh = int(bw / RELACION)
    x1 = max(0, min(cx - bw // 2, ancho_f - bw))
    y1 = max(0, min(cy - bh // 2, alto_f - bh))
    x2, y2 = x1 + bw, y1 + bh

    # Oscurecer todo lo que queda FUERA del recuadro: lo claro es lo que se guarda
    sombra = (frame * 0.35).astype("uint8")
    sombra[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame = sombra

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
    cv2.putText(frame, f"Carpeta {Datos}: {count} imagenes", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Martillo DENTRO del recuadro | S guardar | M espejo: "
                + ("SI" if GUARDAR_ESPEJO else "NO"), (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Lo que se guardará (48 x 24), ampliado solo para verlo
    objeto = cv2.resize(imAux[y1:y2, x1:x2], TAM_MUESTRA, interpolation=cv2.INTER_AREA)
    vista = cv2.resize(objeto, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("frame", frame)
    cv2.imshow("objeto", vista)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('s'), ord('S'), 32):
        guardar(objeto)
        if GUARDAR_ESPEJO:
            guardar(cv2.flip(objeto, 1))
    elif k in (ord('a'), ord('A')):
        cx -= 10
    elif k in (ord('d'), ord('D')):
        cx += 10
    elif k in (ord('w'), ord('W')):
        cy -= 10
    elif k in (ord('x'), ord('X')):
        cy += 10
    elif k in (ord('+'), ord('=')):
        bw += 20
    elif k == ord('-'):
        bw -= 20
    elif k in (ord('m'), ord('M')):
        GUARDAR_ESPEJO = not GUARDAR_ESPEJO
        print("Guardar copia en espejo:", "SI" if GUARDAR_ESPEJO else "NO")
    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Total de positivas en la carpeta:", count)
