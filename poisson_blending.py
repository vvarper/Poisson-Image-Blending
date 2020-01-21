# -*- coding: utf-8 -*-
"""
Proyecto Final: Poisson Image Blending
Nombre Estudiantes: Iván Garzón Segura
                    Víctor Alejandro Vargas Pérez
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse import linalg
from os import listdir
from os.path import isfile, join

# Función para cargar una imagen con valores de tipo float

def cargarImagen(filename, flagColor):
    imagen = cv2.imread(filename, flagColor)

    return imagen.astype(float)

# Función para normalizar una imagen 0-255, uint8

def normalizar(im):
    # Buscar máximo y mínimo en cada canal (si está en grises, hay un solo canal)
    maximo = np.amax(im, axis=(0, 1))
    minimo = np.amin(im, axis=(0, 1))

    if (maximo != minimo).any():
        # Usar estos valores para normalizar la imagen entre 0 y 255
        imagen_normalizada = np.uint8(255*((im-minimo)/(maximo-minimo)))
    else:
        imagen_normalizada = np.uint8(im)

    #imagen_normalizada = np.array(im * 255, dtype=np.uint8)

    return imagen_normalizada

# Función que comprueba si una imagen está en grises

def esGris(im):
    # Las imágenes en grises tienen 2 dimensiones
    if len(im.shape) == 2:
        return True
    else:
        return False

# Función que muestra una imagen ya cargada

def mostrarImagen(im, dibujar=True, titulo=None, norm=True):
    # Normalizar imagen
    if norm:
        imagen_norm = normalizar(im)
    else:
        imagen_norm = im.copy()

    # En primer lugar, se eliminan las dimensiones de los bordes de la imagen
    # mostradas por defecto con pyplot
    plt.xticks([])
    plt.yticks([])

    if esGris(im):  # Si está en grises, se muestra como tal
        plt.imshow(imagen_norm, cmap='gray')
    else:   # Si está en color, se hace una conversión de colores (BGR->RGB)
            # para que los colores con pyplot sean correctos
        imagen_plt_norm = cv2.cvtColor(imagen_norm, cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_plt_norm)

    if titulo != None:
        plt.title(titulo)

    # El parámetro dibujar determina si se quiere hacer un plt.show() o no
    if dibujar:
        plt.show()

# Función que muestra varias imágenes ya cargadas

def mostrarVariasImagenes(vim, titulos=[], reparto=(0, 0), dimensiones=(9, 9),
                          norm=True):

    num_imagenes = len(vim)

    # El número de imágenes por borde (reparto) puede especificarse.
    # En caso contrario, se intenta igualar el número de filas y columnas.
    filas, columnas = reparto
    if reparto == (0, 0):
        filas = int(np.sqrt(num_imagenes))
        columnas = np.ceil(num_imagenes / filas)

    ventana = plt.figure(figsize=dimensiones)
    for i in range(num_imagenes):
        sub_imagen = ventana.add_subplot(filas, columnas, i+1)
        if len(titulos) > 0:
            sub_imagen.set_title(titulos[i])  # Indica el título de la imagen
        mostrarImagen(vim[i], dibujar=False, norm=norm)
    plt.show()

# Función para cargar todos los sets de imágenes fuente/destino/máscaras

def cargaConjuntoImagenes():
    mascaras = []
    fuentes = []
    destinos = []

    for img in sorted(listdir("imagenes/masks")):
        mascara = cargarImagen("imagenes/masks/" + img, 0)
        mascara = np.array(mascara, dtype=np.uint8)
        x, mascara = cv2.threshold(mascara, 0, 255, cv2.THRESH_OTSU)
        mascaras.append(mascara)

    for img in sorted(listdir("imagenes/source")):
        fuentes.append(cargarImagen("imagenes/source/" + img, 1))

    for img in sorted(listdir("imagenes/target")):
        destinos.append(cargarImagen("imagenes/target/" + img, 1))

    return fuentes, mascaras, destinos

################################################################################

# Devuelve omega: los píxeles (índices) dentro de la máscara

def getPosicionesMascara(mascara):
    pos_y, pos_x = np.nonzero(mascara)

    posiciones = []
    for i in range(len(pos_y)):
        posiciones.append((pos_y[i], pos_x[i]))
    return posiciones

# Devuelve las posiciones adyacentes (vecinos) de una dada

def getVecindario(pos):
    i, j = pos

    vecindario = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    return vecindario

# Calcula la matriz de Poisson, es decir, la matriz de coeficientes A para
# un omega dado

def matriz_poisson(omega):
    # Calculamos el número de píxeles de omega (máscara)
    N = len(omega)
    # Matriz dispersa de NxN
    A = sparse.lil_matrix((N, N))

    for i, posicion in enumerate(omega):
        progress_bar(i, N-1)  # OJO
        A[i, i] = 4

        for vecino in getVecindario(posicion):
            if vecino in omega:
                j = omega.index(vecino)
                A[i, j] = -1

    return A

# Comprueba si una posición es parte de la frontera de omega

def esBorde(omega, posicion):
    if posicion in omega:
        for vecino in getVecindario(posicion):
            if vecino not in omega:
                return True

    return False

# Calcula el valor del borde en la imagen destino, necesario para el vector b

def calcularValorBorde(omega, mascara, posicion, destino):
    valorBorde = 0
    if (esBorde(omega, posicion)):
        for vecino in getVecindario(posicion):
            if not vecino in omega:
                if dentroImagen(destino, vecino):
                    valorBorde += destino[vecino]
                else:
                    valorBorde += destino[posicion]

    return valorBorde

# Comprueba que una posición esté dentro de la imagen

def dentroImagen(imagen, posicion):
    if posicion[0] >= 0 and posicion[0] < imagen.shape[0] and \
            posicion[1] >= 0 and posicion[1] < imagen.shape[1]:
        return True
    else:
        return False

# Calcula laplaciana de un píxel

def getLaplaciana(fuente, posicion):
    laplaciana = 0
    for vecino in getVecindario(posicion):
        if dentroImagen(fuente, vecino):
            laplaciana += fuente[posicion] - fuente[vecino]

    return laplaciana

# Calcula la laplaciana de un píxel acorde al criterio de MixingGradientes: en
# cada dirección, se usa el gradiente de mayor valor entre la imagen fuente y destino

def getLaplacianaMix(fuente, destino, posicion):
    laplaciana = 0
    for vecino in getVecindario(posicion):
        if dentroImagen(fuente, vecino):
            grad_fuente = fuente[posicion] - fuente[vecino]
            grad_destino = destino[posicion] - destino[vecino]
            if abs(grad_fuente) > abs(grad_destino):
                laplaciana += grad_fuente
            else:
                laplaciana += grad_destino

    return laplaciana

# Calcula la diverguencia del campo guía (valores vector b). Se calcula esta
# diverguencia para dos posibles v (el de importing gradientes y el de mixing
# gradients)

def calcularDivGuia(omega, mascara, fuente, destino):
    N = len(omega)
    b_import = np.zeros(N)
    b_mix = np.zeros(N)

    for i in range(N):
        progress_bar(i, len(omega)-1)  # OJO
        valorBorde = calcularValorBorde(omega, mascara, omega[i], destino)
        b_import[i] = getLaplaciana(fuente, omega[i]) + valorBorde
        b_mix[i] = getLaplacianaMix(fuente, destino, omega[i]) + valorBorde

    return b_import, b_mix

################################################################################


def progress_bar(n, N):
    '''
    print current progress
    '''

    step = 2
    percent = float(n) / float(N) * 100

    # convert percent to bar
    current = "#" * int(percent//step)
    remain = " " * int(100/step-int(percent//step))
    bar = "|{}{}|".format(current, remain)
    print("\r{}:{:3.0f}[%]".format(bar, percent), end="", flush=True)


# Devuelve tres imágenes (copia/pega en imágenes según criterios distintos):
#   -> Pegado directo, Poisson Blending con Importing Gradients y
#       Poisson Blending con Mixing Gradients
# mascara, fuente y destino son 3 imágenes del mismo tamaño (fil*col).
# mascara es en blanco y negro (0/255)
# fuente y destino son en color (3 canales)

def poisson_blending(mascara, fuente, destino):
    # Queremos obtener f tal que lapl(f) = div(v) = lapl(fuente) dentro de la máscara,
    # y f(borde) = destino(borde), donde borde es el borde de la máscara (está dentro)
    # f fuera de la máscara tendrá los valores de destino

    # Cada solución (solapado, import_gradients y mixing_gradients)
    # se inicializa con los valores de la imagen destino
    solucion_cloning = np.copy(destino)
    solucion_import = np.copy(destino)
    solucion_mix = np.copy(destino)

    # Lista con los índices píxeles máscara
    omega = getPosicionesMascara(mascara)

    # 1. Calcular matriz de coeficientes para omega
    print("\nCalculando A")
    A = matriz_poisson(omega)

    # Para cada canal
    for canal in range(destino.shape[2]):

        print(f"\nCÁLCULOS CANAL {canal}")

        # 2.1 Calcular vector columna b para importing y mixing

        print("Calculando b para importing gradients y para mixing gradients")
        b_import, b_mix = calcularDivGuia(
            omega, mascara, fuente[:, :, canal], destino[:, :, canal])

        # 2.2 Resolver A*f_omega=b

        print("\nCalculando f_omega para importing gradients")
        f_omega_import, _ = linalg.cg(A, b_import)

        print("Calculando f_omega para mixing gradients")
        f_omega_mix, _ = linalg.cg(A, b_mix)

        # 2.3 Incorporar f_omega a f

        print(
            f"Añadiendo los valores calculados al canal {canal} correspondiente en cada solución")

        for i, posicion in enumerate(omega):
            solucion_cloning[posicion[0], posicion[1],
                             canal] = fuente[posicion[0], posicion[1], canal]
            solucion_import[posicion[0], posicion[1],
                            canal] = f_omega_import[i]
            solucion_mix[posicion[0], posicion[1], canal] = f_omega_mix[i]

    # Devolver todas las soluciones, con valores entre 0 y 1
    return solucion_cloning, np.clip(solucion_import, 0, 255), np.clip(solucion_mix, 0, 255)

# Función para aplicar un offset a la máscara, y así colocar en un lugar concreto
# de la imagen destino

def aplicarDesplazamiento(fuente, mascara, despl):
    resultado = np.zeros(fuente.shape)
    mask_resultado = np.zeros(mascara.shape)

    for i in range(fuente.shape[0]):
        for j in range(fuente.shape[1]):
            if i+despl[0] >= 0 and i+despl[0] < fuente.shape[0] and \
                    j+despl[1] >= 0 and j+despl[1] < fuente.shape[1]:
                resultado[i+despl[0], j+despl[1]] = fuente[i, j]
                mask_resultado[i+despl[0], j+despl[1]] = mascara[i, j]

    return resultado, mask_resultado

# Función principal

def main():

    # Carga de imágenes
    fuentes, mascaras, destinos = cargaConjuntoImagenes()

    #despl = (20, 0)
    #fuente, mascara = aplicarDesplazamiento(fuente, mascara, despl)

    res_cloning, res_import, res_mix = poisson_blending(
        mascaras[0], fuentes[0], destinos[0])

    mostrarVariasImagenes([res_cloning, res_import, res_mix])

################################################################################


main()
