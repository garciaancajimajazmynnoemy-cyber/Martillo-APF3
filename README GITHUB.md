<div align="center">

# Detector de Martillo con Haar Cascade

**Visión Computacional aplicada a Ingeniería Civil · APF3**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Haar%20Cascade-5C3EE8?logo=opencv&logoColor=white)
![Modelo](https://img.shields.io/badge/Modelo-15%20etapas%20·%2048×24-1F4E9C)
![Estado](https://img.shields.io/badge/Estado-Demo%20académica-orange)

</div>

---

## ¿De qué trata?

Programa que detecta un **martillo en tiempo real** desde la cámara del celular (DroidCam) o la webcam. Usa un clasificador **Haar Cascade** que entrené con **Cascade Trainer GUI** sobre un dataset propio.

> **Idea de aplicación civil:** apoyar el inventario de herramientas en el almacén de obra y alertar si una queda en una zona de trabajo en altura (Ley N.° 29783 y Norma G.050).

## Datos del trabajo

| | |
|---|---|
| **Alumna** | Jazmyn Noemy Garcia Ancajima |
| **Curso** | Herramientas de desarrollo profesional Civil (UTP) |
| **Docente** | Hilder Javier Jimenez Pesantes |
| **Ciclo / Sección** | VI Ciclo · 49744 |

## El modelo en números

| Dato | Valor |
|---|---|
| Positivas (`p/`) | 135 imágenes de 48 × 24 px |
| Negativas (`n/`) | 254 imágenes de 320 × 180 px |
| Clasificador | Gentle AdaBoost · Haar BASIC |
| Etapas | 15 |
| Ventana | 48 × 24 px |
| Velocidad en vivo | 34 – 39 FPS |

## Estructura del repositorio

```text
Martillo-APF3/
├── cascade.xml               ← modelo entrenado
├── APF3-JazmynGarcia.py      ← detector
├── p/                        ← positivas (con martillo)
├── n/                        ← negativas (sin martillo)
├── pos.lst · neg.lst         ← listas para el entrenador
├── pos_samples.vec           ← muestras generadas
└── Informe_Tecnico.pdf
```

## Cómo ejecutarlo

```bash
pip install opencv-python numpy
```

| Comando | Qué hace |
|---|---|
| `python APF3-JazmynGarcia.py` | Usa DroidCam (`192.168.1.37:4747`) |
| `python APF3-JazmynGarcia.py 0` | Usa la webcam de la PC |
| `python APF3-JazmynGarcia.py foto.jpg` | Detecta en una imagen o video |

Deja `cascade.xml` en la misma carpeta que el script.

## Teclas

| Tecla | Acción |
|:---:|---|
| `ESC` | Salir |
| `C` | Marcar detección **correcta** |
| `F` | Marcar **falso positivo** |
| `N` | Marcar objeto **no detectado** |
| `G` | Guardar captura |

Al salir, el programa imprime el conteo de C, F y N.

## Limitaciones

- Casi todo el dataset es **un solo martillo** sobre una toalla beige.
- Las negativas no incluyen manos ni otras herramientas.
- La caja de detección sale más grande que el objeto.
- Los resultados medidos usan las mismas imágenes del entrenamiento, así que no estiman el desempeño con fotos nuevas.

## Video demostrativo

 [Ver el video en Google Drive](https://drive.google.com/file/d/1IF8PA_T7iG4Y-ZW1EextfJroWTCeR_Km/view?usp=sharing)

---

<div align="center"><sub>UTP · Piura · 2026</sub></div>
