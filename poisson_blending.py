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

    return imagen_normalizada

# Función que comprueba si una imagen está en grises


def esGris(im):
    # Las imágenes en grises tienen 2 dimensiones
    if len(im.shape) == 2:
        return True
    else:
        return False

# Función que muestra una imagen ya cargada


def mostrarImagen(im, dibujar=True, titulo=None, norm=False):
    # Normalizar imagen
    if norm:
        imagen_norm = normalizar(im)
    else:
        imagen_norm = im.copy()
        imagen_norm = np.uint8(imagen_norm)

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

# Función para obtener las posiciones en la imagen fuente del objeto a pegar
# Se debe especificar una ruta con la imagen de la máscara (blanco/negro)


def getObjeto(nombre_mascara):
    ruta_mascara = "imagenes/masks/"
    mascara = np.array(cargarImagen(
        ruta_mascara + nombre_mascara, 0), dtype=np.uint8)
    _, mascara = cv2.threshold(mascara, 0, 255, cv2.THRESH_OTSU)

    posiciones_objeto = getPosicionesMascara(mascara)
    return posiciones_objeto

# Devuelve una lista con los píxeles (índices) dentro de la máscara


def getPosicionesMascara(mascara):
    pos_y, pos_x = np.nonzero(mascara)

    posiciones = []
    for i in range(len(pos_y)):
        posiciones.append((pos_y[i], pos_x[i]))
    return posiciones

# Función para calcular el desplazamiento necesario en una máscara para que
# el centro de esta quede en una posición concreta de la imagen destino
# Se devuelve también un booleano que indica si con ese desplazamiento el objeto
# se sale de la imagen destino


def calcularDesplazamiento(pos_dest, objeto, destino):

    # Si se pasan reales en pos_dest, se interpreta como un porcentaje
    # respecto al tamaño de la imagen destino. En caso contrario, son coordenadas
    # directas.
    if not type(pos_dest[0]) is int:
        pos_dest[0] = int(destino.shape[0]*pos_dest[0])
    if not type(pos_dest[1]) is int:
        pos_dest[1] = int(destino.shape[1]*pos_dest[1])
    pos_dest = np.array(pos_dest, dtype=np.uint32)

    # Obtener los extremos de la máscara y calcular su centro
    posiciones = np.array(objeto)
    max_x = np.max(posiciones[:, 1])
    min_x = np.min(posiciones[:, 1])
    max_y = np.max(posiciones[:, 0])
    min_y = np.min(posiciones[:, 0])

    x_centro = int((max_x + min_x) / 2)
    y_centro = int((max_y + min_y) / 2)

    # Obtener el desplazamiento como la diferencia del objetivo con el centro
    # de la máscara
    despl = pos_dest - (y_centro, x_centro)

    # Comprobar si es un desplazamiento válido
    despl_valido = False
    if dentroImagen(destino, [max_y, max_x] + despl) and dentroImagen(destino, [min_y, min_x] + despl):
        despl_valido = True

    return despl, despl_valido

# Pegar un objeto de forma directa con un desplazamiento dado


def pegarObjeto(omega, fuente, destino, despl):
    solucion = np.copy(destino)
    for posicion in omega:
        pos_dest = tuple(posicion+despl)
        solucion[pos_dest] = fuente[posicion]

    return solucion

# Devuelve las posiciones adyacentes (vecinos) de una dada


def getVecindario(pos):
    i, j = pos

    vecindario = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    return vecindario

# Calcula la matriz de Poisson, es decir, la matriz de coeficientes A para
# un omega dado


def matrizPoisson(omega):
    # Calculamos el número de píxeles de omega
    N = len(omega)
    # Matriz dispersa de NxN
    A = sparse.lil_matrix((N, N))

    # Para cada píxel/posición
    for i, posicion in enumerate(omega):
        progress_bar(i, N-1)  # OJO
        A[i, i] = 4  # 4 en la diagonal (píxel actual)

        # -1 en las columnas de los vecinos en omega
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


def calcularValorBorde(omega, posicion, destino, despl):
    valorBorde = 0
    if (esBorde(omega, posicion)):
        for vecino in getVecindario(posicion):
            if not vecino in omega:
                if dentroImagen(destino, vecino + despl):
                    valorBorde += destino[tuple(vecino + despl)]
                else:
                    valorBorde += destino[tuple(posicion + despl)]

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


def getLaplacianaMix(fuente, destino, posicion, despl):
    laplaciana = 0
    for vecino in getVecindario(posicion):
        # Caso general: calcular y comparar gradientes en las dos imágenes
        if dentroImagen(destino, vecino+despl) and dentroImagen(fuente, vecino):
            grad_fuente = fuente[posicion] - fuente[vecino]
            grad_destino = destino[tuple(
                posicion+despl)] - destino[tuple(vecino+despl)]
            if abs(grad_fuente) > abs(grad_destino):
                laplaciana += grad_fuente
            else:
                laplaciana += grad_destino
        # Si una de las imágenes no contiene al correspondiente vecino:
        # -> política de bordes de copia: gradiente intensidad 0. Se usa directamente
        # el gradiente de la otra
        elif dentroImagen(destino, vecino+despl):
            grad_destino = destino[tuple(
                posicion+despl)] - destino[tuple(vecino+despl)]
            laplaciana += grad_destino
        elif dentroImagen(fuente, vecino):
            grad_fuente = fuente[posicion] - fuente[vecino]
            laplaciana += grad_fuente

    return laplaciana

# Calcula la diverguencia del campo guía (valores vector b). Se calcula esta
# diverguencia para dos posibles v (el de importing gradientes y el de mixing
# gradients)


def calcularDivGuia(omega, fuente, destino, despl):
    N = len(omega)
    b_import = np.zeros(N)
    b_mix = np.zeros(N)

    # Para cada píxel en omega
    for i in range(N):
        progress_bar(i, len(omega)-1)  # OJO
        # Condición contorno (común a ambos métodos)
        valorBorde = calcularValorBorde(omega, omega[i], destino, despl)
        # Diverguencia de v + valorBorde
        b_import[i] = getLaplaciana(fuente, omega[i]) + valorBorde
        b_mix[i] = getLaplacianaMix(
            fuente, destino, omega[i], despl) + valorBorde

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


# Devuelve dos imágenes (copia/pega en imágenes según criterios distintos):
#   -> Poisson Blending con Importing Gradients y con Mixing Gradients
# objeto es una lista de índices de la imagen fuente
# fuente y destino son imágenes color (3 canales)
# despl es una pareja de enteros que indica el desplazamiento de cada
# píxel del objeto en la fuente respecto a su posición final en el destino

def poissonBlending(objeto, fuente, destino, despl):

    # Cada solución (import_gradients y mixing_gradients)
    # se inicializa con los valores de la imagen destino
    solucion_import = np.copy(destino)
    solucion_mix = np.copy(destino)

    print(f"Número de píxeles a modificar: {len(objeto)}")

    # 1. Calcular matriz de coeficientes para omega
    print("\nCalculando A")
    A = matrizPoisson(objeto)

    # Para cada canal
    for canal in range(destino.shape[2]):

        print(f"\nCÁLCULOS CANAL {canal}")

        # 2.1 Calcular vector columna b para importing y mixing

        print("Calculando b para importing gradients y para mixing gradients")
        b_import, b_mix = calcularDivGuia(
            objeto, fuente[:, :, canal], destino[:, :, canal], despl)

        # 2.2 Resolver A*f_omega=b

        print("\nCalculando f_omega para importing gradients")
        f_omega_import, _ = linalg.cg(A, b_import)

        print("Calculando f_omega para mixing gradients")
        f_omega_mix, _ = linalg.cg(A, b_mix)

        # 2.3 Incorporar f_omega a f

        print(
            f"Añadiendo los valores calculados al canal {canal} correspondiente en cada solución")

        for i, posicion in enumerate(objeto):
            solucion_import[posicion[0]+despl[0], posicion[1]+despl[1],
                            canal] = f_omega_import[i]
            solucion_mix[posicion[0]+despl[0], posicion[1] +
                         despl[1], canal] = f_omega_mix[i]

    # Devolver todas las soluciones, con valores entre 0 y 255
    return np.clip(solucion_import, 0, 255), np.clip(solucion_mix, 0, 255)

################################################################################

################################################################################


def pegarAvionEnMontaña():
    objeto = getObjeto("mask_avion.jpg")
    fuente = cargarImagen("imagenes/sources/avion.jpg", 1)
    destino = cargarImagen("imagenes/targets/montaña.jpg", 1)

    pos_dest = [0.5, 0.5]
    despl, despl_valido = calcularDesplazamiento(pos_dest, objeto, destino)

    if (despl_valido):
        res_paste = pegarObjeto(objeto, fuente, destino, despl)
        mostrarImagen(res_paste)

        res_import, res_mixing = poissonBlending(
            objeto, fuente, destino, despl)

        mostrarVariasImagenes([res_paste, res_import, res_mixing])
        cv2.imwrite('salidas/avion_montaña_paste.png', res_paste)
        cv2.imwrite('salidas/avion_montaña_import.png', res_import)
        cv2.imwrite('salidas/avion_montaña_mixing.png', res_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")

def pegarGradienteMuro():
    objeto = getObjeto("mask_grad.png")
    fuente = cargarImagen("imagenes/sources/grad.png", 1)
    destino = cargarImagen("imagenes/targets/muro.png", 1)

    pos_dest = [0.5, 0.5]
    despl, despl_valido = calcularDesplazamiento(pos_dest, objeto, destino)

    if (despl_valido):
        res_paste = pegarObjeto(objeto, fuente, destino, despl)
        mostrarImagen(res_paste)

        res_import, res_mixing = poissonBlending(
            objeto, fuente, destino, despl)

        mostrarVariasImagenes([res_paste, res_import, res_mixing])
        cv2.imwrite('salidas/grado_muro_paste.png', res_paste)
        cv2.imwrite('salidas/grado_muro_import.png', res_import)
        cv2.imwrite('salidas/grado_muro_mixing.png', res_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")

def pegarGrafitiPared():
    objeto = getObjeto("mask_grafiti.jpg")
    fuente = cargarImagen("imagenes/sources/grafiti.jpg", 1)
    destino = cargarImagen("imagenes/targets/pared.jpg", 1)

    pos_dest = [0.3, 0.5]
    despl, despl_valido = calcularDesplazamiento(pos_dest, objeto, destino)

    if (despl_valido):
        res_paste = pegarObjeto(objeto, fuente, destino, despl)
        mostrarImagen(res_paste)

        res_import, res_mixing = poissonBlending(
            objeto, fuente, destino, despl)

        mostrarVariasImagenes([res_paste, res_import, res_mixing])
        cv2.imwrite('salidas/grafiti_pared_paste.png', res_paste)
        cv2.imwrite('salidas/grafiti_pared_import.png', res_import)
        cv2.imwrite('salidas/grafiti_pared_mixing.png', res_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")

def pegarPinguinoParque():
    objeto = getObjeto("mask_pinguino.jpg")
    fuente = cargarImagen("imagenes/sources/pinguino.jpg", 1)
    destino = cargarImagen("imagenes/targets/parque.jpg", 1)

    pos_dest = [0.8, 0.15]
    despl, despl_valido = calcularDesplazamiento(pos_dest, objeto, destino)

    if (despl_valido):
        res_paste = pegarObjeto(objeto, fuente, destino, despl)
        mostrarImagen(res_paste)

        res_import, res_mixing = poissonBlending(
            objeto, fuente, destino, despl)

        mostrarVariasImagenes([res_paste, res_import, res_mixing])
        cv2.imwrite('salidas/parque_pinguino_paste.png', res_paste)
        cv2.imwrite('salidas/parque_pinguino_import.png', res_import)
        cv2.imwrite('salidas/parque_pinguino_mixing.png', res_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")

def pegarPinguinoPlaya():
    objeto = getObjeto("mask_pinguino.jpg")
    fuente = cargarImagen("imagenes/sources/pinguino.jpg", 1)
    destino = cargarImagen("imagenes/targets/playa_atardecer.jpg", 1)

    pos_dest = [0.8, 0.7]
    despl, despl_valido = calcularDesplazamiento(pos_dest, objeto, destino)

    if (despl_valido):
        res_paste = pegarObjeto(objeto, fuente, destino, despl)
        mostrarImagen(res_paste)

        res_import, res_mixing = poissonBlending(
            objeto, fuente, destino, despl)

        mostrarVariasImagenes([res_paste, res_import, res_mixing])
        cv2.imwrite('salidas/playa_pinguino_paste.png', res_paste)
        cv2.imwrite('salidas/playa_pinguino_import.png', res_import)
        cv2.imwrite('salidas/playa_pinguino_mixing.png', res_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")

def pegarTazaCocina():
    objeto = getObjeto("mask_taza.jpg")
    fuente = cargarImagen("imagenes/sources/taza.jpg", 1)
    destino = cargarImagen("imagenes/targets/cocina.jpg", 1)

    pos_dest = [282, 256]
    despl, despl_valido = calcularDesplazamiento(pos_dest, objeto, destino)

    if (despl_valido):
        res_paste = pegarObjeto(objeto, fuente, destino, despl)
        mostrarImagen(res_paste)

        res_import, res_mixing = poissonBlending(
            objeto, fuente, destino, despl)

        mostrarVariasImagenes([res_paste, res_import, res_mixing])
        cv2.imwrite('salidas/taza_cocina_paste.png', res_paste)
        cv2.imwrite('salidas/taza_cocina_import.png', res_import)
        cv2.imwrite('salidas/taza_cocina_mixing.png', res_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen") 


def pegarOsoNiñosPlaya():
    mask_niño = getObjeto("mask_niño.jpg")
    mask_oso = getObjeto("mask_oso.jpg")
    mask_niña = getObjeto("mask_niña.jpg")
    niños = cargarImagen("imagenes/sources/niños.png", 1)
    oso = cargarImagen("imagenes/sources/oso.jpg", 1)
    destino = cargarImagen("imagenes/targets/perez_water.jpg", 1)

    pos_dest_oso = [0.25, 0.5]
    despl_oso, despl_valido1 = calcularDesplazamiento(
        pos_dest_oso, mask_oso, destino)

    pos_niño = [0.75, 0.65]
    despl_niño, despl_valido2 = calcularDesplazamiento(
        pos_niño, mask_niño, destino)

    pos_niña = [0.75, 0.45]
    despl_niña, despl_valido3 = calcularDesplazamiento(
        pos_niña, mask_niña, destino)

    if despl_valido1 and despl_valido2 and despl_valido3:
        res_paste = pegarObjeto(mask_oso, oso, destino, despl_oso)
        res_paste = pegarObjeto(mask_niño, niños, res_paste, despl_niño)
        res_paste = pegarObjeto(mask_niña, niños, res_paste, despl_niña)
        mostrarImagen(res_paste)

        res1_import, res1_mixing = poissonBlending(
            mask_oso, oso, destino, despl_oso)
        res2_import, _ = poissonBlending(
            mask_niño, niños, res1_import, despl_niño)
        _, res2_mixing = poissonBlending(
            mask_niño, niños, res1_mixing, despl_niño)
        _, res3_mixing = poissonBlending(
            mask_niña, niños, res2_mixing, despl_niña)
        res3_import, _ = poissonBlending(
            mask_niña, niños, res2_import, despl_niña)

        mostrarVariasImagenes(
            [res_paste, res3_import, res3_mixing], norm=False)
        cv2.imwrite('salidas/oso_niños_paste.png', res_paste)
        cv2.imwrite('salidas/oso_niños_import.png', res3_import)
        cv2.imwrite('salidas/oso_niños_mixing.png', res3_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")

def pegarLunaPlaya():
    mask_luna = getObjeto("mask_luna.jpg")
    mask_brillo = getObjeto("mask_luna_brillo.jpg")
    source = cargarImagen("imagenes/sources/luna.jpg", 1)
    destino = cargarImagen("imagenes/targets/playa.jpg", 1)

    pos_luna = [0.15,0.75]
    despl_luna, despl_valido1 = calcularDesplazamiento(pos_luna, mask_luna, destino)

    pos_brillo = [0.75, 0.75]
    despl_brillo, despl_valido2 = calcularDesplazamiento(pos_brillo, mask_brillo, destino)

    if despl_valido1 and despl_valido2:
        res_paste = pegarObjeto(mask_luna, source, destino, despl_luna)
        res_paste = pegarObjeto(mask_brillo, source, res_paste, despl_brillo)
        mostrarImagen(res_paste)

        res1_import, res1_mixing = poissonBlending(mask_brillo, source, destino, despl_brillo)
        res2_import, _ = poissonBlending(mask_luna, source, res1_import, despl_luna)
        _, res2_mixing = poissonBlending(mask_luna, source, res1_mixing, despl_luna)

        mostrarVariasImagenes([res_paste, res2_import, res2_mixing], norm=False)
        cv2.imwrite('salidas/playa_luna_paste.jpg', res_paste)
        cv2.imwrite('salidas/playa_luna_import.jpg', res2_import)
        cv2.imwrite('salidas/playa_luna_mixing.jpg', res2_mixing)
    else:
        print("Posición destino no válida: el objeto se sale de la imagen")



# pegarGrafitiPared()
# pegarGradienteMuro()
# pegarAvionEnMontaña()
# pegarOsoNiñosPlaya()
# pegarLunaPlaya()
# pegarPinguinoParque()
# pegarTazaCocina()
pegarPinguinoParque()
