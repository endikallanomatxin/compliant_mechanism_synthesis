# Refactor del pipeline de entrenamiento

## Idea central

Hemos sido demasiado optimistas con el hecho de tener una simulación
diferenciable.

La diferenciabilidad **sí ayuda mucho a optimizar un caso concreto**:
dado un mecanismo, permite obtener gradientes útiles para refinarlo.

Pero eso **no implica** que sea fácil entrenar un **modelo general** que aprenda
a optimizar muchos casos distintos de forma estable.

En otras palabras:
- optimizar un diseño concreto con gradientes de la simulación es una cosa,
- entrenar un optimizador amortizado que generalice a muchos diseños es otra
  mucho más difícil.


## Problema actual: RL demasiado temprano.

El pipeline actual mete RL demasiado pronto y además lo hace sobre una base
todavía débil.

Problemas principales:
- mezcla demasiado pronto supervisado y RL,
- los ejemplos de partida no son lo bastante buenos,
- se le pide al modelo explorar antes de saber refinar.

Resultado:
- aprende algo al principio,
- parece prometedor,
- pero luego se vuelve inestable,
- y deja de mejorar de forma fiable.


## Nueva filosofía

El nuevo orden será:

1. generar buenos ejemplos con un optimizador explícito de casos,
2. entrenar supervisado sobre esos ejemplos,
3. y solo después meter RL / online refinement / self-improvement.

Primero aprender a hacer mejoras marginales útiles.  
Después aprender a explorar estrategias más globales.

Hasta que no se demuestre lo contrario, la arquitectura actual vale. El
cuello de botella es el pipeline de entrenamiento.


## Generación de ejemplos

Como los vamos a usar para entrenamiento supervisado, es importante que estos
sean ya resultados buenos. Cuanto mejores sean, más útil será el entrenamiento
supervisado, y más fácil será luego introducir RL.

Afortunadamente, la simulación diferenciable sí es una herramienta potente para
generar buenos ejemplos, porque permite construir un optimizador explícito que
use directamente los gradientes de la simulación para refinar casos concretos.

Su papel será: partir de buenos starting points, usar gradientes de la
simulación para refinar mecanismos concretos y producir ejemplos de bastante
calidad.

### Idea: Generación de starting points

Primero se hace un grafo `scaffold`, con pocos nodos y conexiones binarias.

Luego toman tramos de ese grafo y se usan como puntos de beziers para la
generación de primitivas de tipo aleatorio.

### Idea: Optimización en dos fases.

Primero optimizar a nivel scaffold y luego optimizar a nivel de nodos libres.

### Idea: Aumentar diversidad de la consigna de rigidez.

Como se diseñan en batch, se puede hacer que si un componente de rigidez está
por encima o por debajo de la media, lo lleve un poco más lejos usando una loss
adicional para esto.

Así la optimización no solo optimiza, sino que también diversifica el dataset.


## Entrenamiento supervisado

### Condicionamiento de la generación supervisada

No tiene sentido supervisar una única solución final:

$$
(K^\*, s_0) \mapsto s_T
$$

porque puede haber muchas soluciones válidas para un mismo target.

Cuando las mejoras son marginales, esto no tiene por qué ser un problema.
El problema aparece al intentar partir de cero, porque hay demasiadas maneras
posibles de llegar a una solución válida.

Condicionar la generación supervisada es darle al modelo más información sobre
el modo de solución que se espera, sin darle la solución completa.

$$
(K^\*, s_0, style) \mapsto s_T
$$

#### Idea: Token de estilo

Durante entrenamiento, un modelo auxiliar puede ver el resultado final o
información del futuro y comprimirlo en un token pequeño que ayude al modelo
principal a orientarse.

La idea no es darle la respuesta, sino una pista comprimida sobre el modo o el
estilo de solución.

Es importante que el token sea pequeño, para que no sea una fuga de información
detallada de la solución. También podemos cubrirnos de sobrefijación en este
token usando ruido y dropout (estilo VAE).

Esto no es lo primero que hay que hacer, pero sí una extensión muy razonable
una vez exista una fase supervisada seria.

> [!NOTE]
>
> Esta idea está inspirada en el paper de Aloha, en el VAE que entrena
> para obtener un embedding de estilo.

Más adelante podría intentarse predecir ese token sin ver el futuro, solo en
base al input ya con ruido,  para reducir la brecha entre entrenamiento e
inferencia. Si no, dejarlo a 0 también es una opción válida.


### Curriculum learning

Para facilitar el aprendizaje, es mejor empezar por mejoras marginales y ir
progresando hacia casos más ruido.


# Etapas del refactor

## Etapa 0. Reset conceptual y limpieza

Replantear la codebase alrededor del nuevo pipeline.

Separar mejor:

- simulación estructural diferenciable,
- dataset:
  - generación de casos,
  - optimización de casos,
- entrenamiento:
  - supervisado,
  - entrenamiento online / RL,
- evaluación,
- visualizaciones.


## Etapa 1. Dataset offline bueno

- Pasarnos ya a 3D:
  - FEM
  - Funciones de visualización

- Generar unos buenos starting points a partir de primitivas como:
  - láminas celosía rectas, curvadas y en helix,
  - vigas rectas y curvadas,
  - truss rígidos siguiendo un camino,
  - puntos sueltos para que el optimizador aproveche.

- Definir bien las funciones de loss para el optimizador explícito de casos,
que aproveche los gradientes de la simulación para refinar casos concretos.

  Es importante que el optimizador explícito de casos produzca ejemplos con las
  características deseables en el modelo final, porque el modelo aprenderá a
  imitar esos ejemplos.

- Programar el optimizador y refinar los ejemplos.


## Etapa 2. Supervisado fuerte

Entrenar el modelo principal sobre ese dataset offline.

El objetivo de esta fase es que el modelo ya sepa refinar casos con ruido sin
necesidad de RL.

Curriculum learning: La dificultad tiene que ir aumentando progresivamente,
empezando por casos ya razonables y luego yendo hacia casos más ruidosos y
ambiguos.

Para las primeras pruebas igual no hace falta el token de estilo, pero para
partir de cero seguramente sea necesario. Evaluar su necesidad cuando llegue el
momento.


## Etapa 3. RL sobre un modelo que ya entiende el problema

Cuando el modelo ya sepa refinar razonablemente bien en supervisado, entonces sí
introducir RL / online refinement.


## Etapa 4. Unión -> Mecanismo

### `mobile` -> `input` y `output`

Hasta ahora solo habíamos planteado como un solido `mobile` se mueve en relación
a un sólido `fixed` bajo la influencia de unas fuerzas, optimizando los nodos
`free` y sus conexiones.

Ahora el objetivo sería pasar a especificar la relación entre un solido `input`
y un sólido `output`, y poder definir relaciones más ricas como trayectorias.

Cada uno de los sólidos tiene un conjunto de nodos con movimiento solidario.


### Lineal -> No lineal

Si queremos modelar mecanismos de verdad, habrá que salir de la lógica
puramente lineal alrededor de un estado, y hacer un análisis no lineal del
comportamiento a lo largo del recorrido.


### Especificar trayectorias

Ahora el target es una matriz de rigidez global, para poder definir un
mecanismo, deberíamos definir un mapa entre trayectorias de entrada y de
salida, y de rigideces a lo largo de esa trayectoria.

Definimos:
- Posición de reposo del mecanismo.
- `t_deformación` a lo largo de la trayectoria física del mecanismo.
  - Va de -1 a 1. En 0, el mecanismo puede no estar en su posición de reposo,
  llegará a la posición del instante 0 linealmente.
  - 2 gdl? podríamos abrir la puerta a más?
- Puntos de control:
  - En cada instante de `t_deformación`, el input y el output tienen
  posiciones (6D) determinadas.
  - En esas posiciones, el mecanismo tiene una matriz de rigidez global
  determinada. (Igual deberíamos considerar que la matriz de rigidez global
  se exprese en las coordenadas locales del sistema de referencia de input).

Aun así, la idea es seguir usando la misma arquitectura, con el mismo número de
tokens. Los tokens toman un carácter multi-instante.

I/O:
- La consigna de los t_d de input y output puede ir en los tokens que
pertenecen a los nodos de los sólidos input y output. O igual es mejor que sea
algo más global?
- La rigidez en cada punto de la trayectoria puede ir en un embedding global
que se suma a todos los tokens.
- Las magnitudes pueden representarse con dimensiones crudas o codificadas
sinusoidalmente/Fourier, o una combinación de ambas. Pensarlo bien.

### Ponderación de prioridades del recorrido

La capacidad de definir la importancia de cada requisito se vuelve importante.
Querríamos poder definir indiferencia en ciertos requisitos.

Igual un peso por cada instante y componente de posición y rigidez.

