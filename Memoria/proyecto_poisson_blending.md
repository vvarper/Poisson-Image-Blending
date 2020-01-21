---
title: |
 	| Proyecto Final. Poisson Blending.
subtitle: |
	| Visión por Computador
	| Iván Garzón Segura
	| Víctor Alejandro Vargas Pérez
titlepage: true
toc: true # Añadir índice
toc-own-page: true
lang: es-ES
listings-no-page-break: true
listings-disable-line-numbers: true
logo: img/logoUGR.jpg
logo-width: 175
numbersections: true
---

# Introducción {#intro}

El objetivo de este proyecto es implementar una herramienta que permita recortar objetos de una imagen y pegarlos en otra imagen sin que aparentemente se note en la imagen final que los objetos han sido pegados. Para ello, existen diferentes métodos, como el uso de pirámides Laplacianas. Sin embargo, hemos optado por utilizar el método Poisson Blending, que obtiene mejores resultados. Dicho método se describe en detalle en el paper [Poisson Image Editing](https://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf) (2003), e indicaremos a continuación lo más relevante del mismo.

Esta técnica tiene como elemento principal la ecuación diferencial de Poisson con 
la condición de frontera de Dirichlet, que especifica la Laplaciana de una función (imagen)
desconocida en un dominio de interés, junto con sus valores para la frontera del dominio.

Resolver la ecuación de Poisson se puede ver como un problema de minimización en el que se calcula la función cuyo gradiente es el más cercano a un campo vectorial guía establecido bajo unas condiciones de contorno dadas. De esta forma, la función (imagen) reconstruida interpola las condiciones de contorno hacia dentro, mientras sigue las variaciones espaciales del campo vectorial guía de la forma más fiel posible. 

# Fundamentos Teóricos

Para presentar esta técnica, consideremos los siguientes elementos:

- S: dominio de la imagen (subconjunto cerrado de $\mathbb{R}^{2}$)

- $\Omega$: subconjunto cerrado de S con frontera $\partial \Omega$ (zona de pegado en la imagen destino). Para nuestro caso concreto (imágenes) podemos definir $\partial \Omega$ como todos los píxeles de S que tienen algún vecino (uno de los cuatro píxeles adyacentes) en $\Omega$ (sin estar contenido en este). 

- f*: función escalar definida en S menos el interior de $\Omega$, que representa los valores 
      de la imagen destino fuera del área de pegado.

- f: función desconocida dentro $\Omega$, es decir, los píxeles finales de dicha zona tras pegar el objeto.

- g: función escalar dentro de $\Omega$ cuyos valores corresponden a los píxeles del objeto en la imagen fuente.

- v: campo vectorial guía definido sobre $\Omega$. La elección de este campo guía determinará el efecto de la técnica, y en el presente proyecto se contemplarán las dos opciones presentadas en el paper relacionadas con la inserción de imágenes(importing gradientes y mixing gradients).

\pagebreak

La interpolación de f de f* sobre $\Omega$, con la restricción añadida de un campo vectorial v, corresponde al siguiente problema:

$$
\min _{f} \iint_{\Omega}|\nabla f-\mathbf{v}|^{2} \text { con }\left.f\right|_{\partial \Omega}=f^{*} |_{\partial \Omega}
$$

La solución de este problema corresponde a la siguiente ecuación diferencial de Poisson con condiciones de frontera de Dirichlet:

$$
\Delta f=\text { divv over } \Omega, \text { con }\left.f\right|_{\partial \Omega}=f^{*} |_{\partial \Omega}
$$

Esta ecuación es fundamental, pues su solución para f contendrá los valores finales de los píxeles en la zona de pegado. Por consiguiente, la implementación del algoritmo consistirá en calcular la divergencia de v, y posteriormente obtener f como la solución de Af=b, donde:

- b es un vector columna con tantos valores n como píxeles en $\Omega$ (junto a $\partial \Omega$).

- f es otro vector columna del mismo tamaño (píxeles de la solución).

- A es el operador laplaciano, es decir, una matriz dispersa nxn con 4's (en la diagonal) y -1's, tal que al multiplicarla por un vector columna, calcula otro vector columna con la laplaciana de cada valor del primero: dado un píxel, su laplaciana es la suma de sus gradientes, o dicho de otra forma, la suma de las diferencias con sus cuatro vecinos (dado $x_{i,j}$, su laplaciana sería $4*x_{i,j} - x_{i-1,j} - x_{i+1,j} - x_{i,j-1} - x_{i,j+1}$)


Esta implementación se verá en detalle en la sección 3.

## Importing Gradients 

Para pegar un objeto de una imagen en otra, la elección básica para el campo v es el gradiente de la parte correspondiente al objeto en la imagen fuente (g). De esta forma, $\mathbf{v}=\nabla g$, y por lo tanto la ecuación de Poisson pasaría a ser la siguiente:

$$
\Delta f=\Delta g \text { over } \Omega, \text { con }\left.f\right|_{\partial \Omega}=f^{*} |_{\partial \Omega}
$$

\pagebreak

## Mixing Gradients

Si bien la elección anterior para v funciona correctamente cuando se tratan objetos opacos, los resultados no serían satisfactorios para objetos con transparencias o agujeros, pues v no contiene información del fondo de la imagen destino. Por consiguiente, se propone usar para v variaciones tanto de g como de f*, escogiendo en cada caso aquella de mayor valor:

$$
\text { para todo } \mathbf{x} \in \Omega, \mathbf{v}(\mathbf{x})=\left\{\begin{array}{ll}
{\nabla f^{*}(\mathbf{x})} & {\text { si }\left|\nabla f^{*}(\mathbf{x})\right|>|\nabla g(\mathbf{x})|} \\
{\nabla g(\mathbf{x})} & {\text { en otro caso }}
\end{array}\right.
$$

# Implementación 

# Resultados 

# Referencias