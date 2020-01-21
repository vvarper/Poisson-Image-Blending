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

################################################################################


def getPosicionesMascara(mascara):
    pos_y, pos_x = np.nonzero(mascara)

    posiciones = []
    for i in range(len(pos_y)):
        posiciones.append((pos_y[i], pos_x[i]))
    return posiciones


def getVecindario(pos):
    i, j = pos

    vecindario = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    return vecindario


def matriz_poisson(omega):
    # Calculamos el número de píxeles de omega (máscara)
    N = len(omega)
    # Matriz dispersa de NxN
    A = sparse.lil_matrix((N, N))

    for i, posicion in enumerate(omega):
        A[i, i] = 4

        for vecino in getVecindario(posicion):
            if vecino in omega:
                j = omega.index(vecino)
                A[i, j] = -1

    return A


def esBorde(omega, posicion):
    if posicion in omega:
        for vecino in getVecindario(posicion):
            if vecino not in omega:
                return True

    return False


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


def dentroImagen(imagen, posicion):
    if posicion[0] >= 0 and posicion[0] < imagen.shape[0] and \
            posicion[1] >= 0 and posicion[1] < imagen.shape[1]:
        return True
    else:
        return False


def getLaplaciana(fuente, posicion):
    laplaciana = 0
    for vecino in getVecindario(posicion):
        if dentroImagen(fuente, vecino):
            laplaciana += fuente[posicion] - fuente[vecino]

    return laplaciana


def calcularDivGuia(omega, mascara, fuente, destino):
    b = np.zeros(len(omega))
    for i in range(len(omega)):
        b[i] = getLaplaciana(fuente, omega[i]) + \
            calcularValorBorde(omega, mascara, omega[i], destino)

    return b


def getLaplacianaMix(fuente, destino, posicion):
    laplaciana = 0
    for vecino in getVecindario(posicion):
        if dentroImagen(fuente, vecino):
            # i,j = posicion
            # i_vecino, j_vecino = vecino
            grad_fuente = fuente[posicion] - fuente[vecino]
            grad_destino = destino[posicion] - destino[vecino]
            if abs(grad_fuente) > abs(grad_destino):
                laplaciana += grad_fuente
            else:
                laplaciana += grad_destino

    return laplaciana


def calcularDivGuiaMix(omega, mascara, fuente, destino):
    b = np.zeros(len(omega))

    for i in range(len(omega)):
        b[i] = getLaplacianaMix(fuente, destino, omega[i]) + \
            calcularValorBorde(omega, mascara, omega[i], destino)

    return b

################################################################################

# mascara, fuente y destino son 3 imágenes del mismo tamaño (fil*col).
# mascara es en blanco y negro (0/255)
# fuente y destino son en color (3 canales)


def poisson_blending(mascara, fuente, destino, mixingGradients):
    # Queremos obtener f tal que lapl(f) = div(v) = lapl(fuente) dentro de la máscara,
    # y f(borde) = destino(borde), donde borde es el borde de la máscara (está dentro)
    # f fuera de la máscara tendrá los valores de destino

    print(destino.shape)

    # Lista con coordenadas píxeles máscara
    omega = getPosicionesMascara(mascara)

    print("Calculando A")
    A = matriz_poisson(omega)

    u = []

    for canal in range(3):
        print("Calculando b")
        # Calcular b en canal
        if (mixingGradients):
            b = calcularDivGuiaMix(
                omega, mascara, fuente[:, :, canal], destino[:, :, canal])
        else:
            b = calcularDivGuia(
                omega, mascara, fuente[:, :, canal], destino[:, :, canal])

        print("Calculando u")
        # Resolver Au=b en canal y guardar u
        u_canal, x = linalg.cg(A, b)
        u.append(u_canal)
        # print(np.max(b))
        # print(np.min(b))
        # print(np.max(u_canal))
        # print(np.min(u_canal))

    solucion = np.copy(destino)

    print("Calculando solución final")
    for i, posicion in enumerate(omega):
        for canal in range(3):
            pos_y, pos_x = posicion
            solucion[pos_y, pos_x, canal] = u[canal][i]

    return np.clip(solucion, 0, 1)

    ############################################################################
    # Resolver Au = b
    #   A: Matriz poisson
    #   b: (g) (parte de la imagen fuente en omega/mascara)
    #   u: solución (valores finales de los píxeles)

    # Sacar máscara bordes (frontera) a partir de la máscara dada
    # Sacar región omega

    # Crear matriz A a partir de omega y mascara: coefficient_matrix

    # Crear la matriz (vector) de gradientes de tamaño num_filas de omega
    #   Mixing_gradients a partir de fuente, destino, omega y frontera
    #       -> Hacer esto para cada canal

    # Resolver, para cada canal, Au = b (sp.linalg.cg) -> devuelve u

    # Devolver el resultado como el destino sobreescribiendo las poisiciones
    # de omega con las de u
    ############################################################################

    # 1. Obtener laplaciana(fuente) dentro de la máscara -> (B)

    # 2. Obtener matriz A operador laplaciano con las dimensiones correspondientes a la máscara
    #
    # 3. Resolver Ax = B (función scipy)

    # 4. Construir f como la imagen destino con los valores de x dentro de la máscara


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


def superponer(mascara, fuente, destino):
    posiciones = getPosicionesMascara(mascara)
    resultado = np.copy(destino)

    for pos in posiciones:
        resultado[pos] = fuente[pos]

    return resultado


def main():
    mascara = cargarImagen('imagenes/mask.jpg', 0)
    mascara = np.array(mascara, dtype=np.uint8)
    x, mascara = cv2.threshold(mascara, 0, 255, cv2.THRESH_OTSU)
    fuente = cargarImagen('imagenes/luna.jpg', 1)
    destino = cargarImagen('imagenes/playa.jpg', 1)

    despl = (20, 0)
    #fuente, mascara = aplicarDesplazamiento(fuente, mascara, despl)
    mostrarImagen(mascara)
    resultado = poisson_blending(
        mascara/255.0, fuente/255.0, destino/255.0, True)
    # print(np.max(resultado))
    # print(np.min(resultado))
    mostrarImagen(resultado)


main()
