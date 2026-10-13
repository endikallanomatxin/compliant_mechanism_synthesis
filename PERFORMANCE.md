# Performance

## Recorrido natural de menor rigidez

El análisis principal usa ahora `soft_modes.py`: relajación de nodos libres con el sólido móvil en una pose prescrita, condensación de rigidez y continuación del modo más blando. La búsqueda diferencia el recorrido completo, incluida la dirección modal. Los recorridos retenidos permiten saltarse Newton al repetir el gradiente; cambiar la consigna invalida la caché porque cambia el sentido y la longitud de avance. Cambiar los pesos del coste conserva los recorridos.

Los espectros efectivos de 3×3 o 6×6 se calculan en el dispositivo. En Intel XPU se usa Jacobi simétrico, ya que `torch.linalg.eigh` hace fallback a CPU en este entorno. El backward del modo mínimo evita los denominadores entre modos superiores repetidos. Las comprobaciones incluyen comparación CPU/XPU de espectros, recorridos y derivadas, condensación analítica y diferencias finitas de recorridos 2D/3D.

Medición CPU con población 8, lotes de 4 y 12 incrementos, un hilo: evaluación inicial 2,52 s y pico RSS 713 MB; paso de gradiente 2,02 s y pico RSS acumulado 802 MB. Esta medición se tomó antes de añadir la cuadratura de longitud de la consigna y no es una garantía de tiempo para otras estructuras o poblaciones. La caché guarda estados separados del grafo y se limita a la población retenida. Las gráficas numéricas se actualizan al recibir un snapshot, no al girar la cámara.

Las mediciones siguientes corresponden a verificaciones previas del FEM controlado por carga. No describen el coste del nuevo recorrido natural.

# Mediciones de tiempo y memoria

Medido localmente el 13 de octubre de 2026, con PyTorch 2.14.1+xpu e Intel Arc B390. Se distinguen RAM residente del proceso, memoria de tensores XPU y heap del navegador. **No son medidas intercambiables.**

## Qué explica la memoria

Los siete servidores de prueba que habían quedado abiertos sumaban aproximadamente 15 GB de RAM. Ya se cerraron, conservando la instancia del usuario. Eso explica una parte importante de la presión del sistema y la swap llena.

Las medidas adicionales muestran que los datos del diseño no explican por sí solos el tamaño de un proceso XPU:

| Etapa en procesos aislados | RAM residente aproximada |
| --- | ---: |
| Python antes de importar PyTorch | 13 MiB |
| Importar PyTorch | 476 MiB |
| Primer tensor XPU | 566 MiB |
| Primer solve XPU | 752 MiB |
| Primer backward de solve XPU | 833 MiB |
| Primera Hessiana de vigas XPU | 2.52 GiB |
| Síntesis XPU, tras calentamiento | 3.4–3.5 GiB |
| Síntesis CPU, tras calentamiento | 0.70–0.71 GiB |

En la prueba XPU, los tensores persistentes ocupaban **48.128 bytes**, el allocator reservaba 2–6 MiB y el pico de tensores durante gradiente era aproximadamente 4 MiB. Al borrar la búsqueda, ejecutar GC y `torch.xpu.empty_cache()`, la memoria asignada/reservada XPU volvió a cero, pero la RAM residente se mantuvo alrededor de 3.4 GiB. Un `malloc_trim` diagnóstico liberó apenas unos 8 MiB. Por tanto, vaciar el allocator de tensores no resuelve esa retención.

Esto apunta a memoria nativa asociada a inicialización/ejecución del backend XPU, fuera del allocator de tensores que expone PyTorch. Estas medidas **no identifican todavía la biblioteca o allocation nativa exacta**: atribuirlo específicamente al driver, compilador o una caché concreta requeriría trazado de allocations nativas. No se han añadido llamadas repetidas a GC, `empty_cache` o `malloc_trim` como supuesto remedio.

En seis generaciones CPU, la RAM pasó de unos 717 a 719 MiB y se estabilizó; tras gradiente llegó a 725 MiB. La caché de respuestas mantuvo una entrada. En cuatro generaciones XPU la RAM también se estabilizó tras el calentamiento. Estas pruebas no muestran crecimiento continuo, aunque no descartan problemas en sesiones largas, diseños mayores o condiciones diferentes.

`auto` ejecuta una prueba en ambos dispositivos y por ello inicializa XPU incluso si elige CPU. El dispositivo inicial es ahora **CPU**, para evitar ese coste nativo por defecto. XPU y Auto siguen seleccionables en el configurador y CLI. El visor WebGL sigue usando el dispositivo gráfico del navegador independientemente de esta elección.

## Tiempo del backend antes de las correcciones estructurales

Con 16 individuos, el diseño de 16 nodos y 118 conexiones candidatas, y cuatro incrementos nominales:

- La evaluación CPU instrumentada dedicó aproximadamente el 73 % del tiempo de un paso de gradiente a transformaciones de Hessianas; seis generaciones tardaron entre 0.8 y 2.6 s bajo cProfile. Un paso de gradiente tardó unos 11 s. cProfile añade coste, por lo que estos valores no son un benchmark de velocidad absoluta.
- En la prueba XPU sincronizada, las generaciones tardaron aproximadamente 1.3–2 s. En el paso de gradiente, 4.6 de 7.8 s se dedicaron a Hessianas. Las secuencias CPU/XPU tienen mutaciones distintas; no permiten comparar exactamente una misma generación.
- Una medición CPU sin cProfile y con dos incrementos dedicó 2.6 de 3.7 s de gradiente a Hessianas.
- Consultar una respuesta ya cacheada tardó aproximadamente 0.06–0.13 ms en Python, sin repetir FEM.

En esa implementación, el coste dominante era diferenciar repetidamente la energía respecto a **todos los DOFs globales** para obtener la tangente de Newton, más que almacenar geometría. Además, el backtracking de diseño evalúa varios recorridos completos. Dividir lotes y filas de Hessiana limita el pico de memoria, pero no elimina ese trabajo.

Se contrastó forward-over-reverse frente a reverse-over-reverse por bloques para una Hessiana 2D CPU: 9.5 frente a 10.8 ms, con resultados iguales. La mejora aislada fue pequeña; no se cambió toda la formulación basándose en esa prueba.

Se implementaron después tangentes por elemento y su ensamblaje, conservando la derivación implícita del equilibrio. La equivalencia con la Hessiana global de la energía se comprueba en 2D y 3D, CPU y XPU, con deformaciones y rotaciones no nulas.

### Reducción del trabajo de Newton

Newton elimina ahora de su lote temporal las conexiones cuya presencia es exactamente cero en todos los individuos. Conserva una lista común de conexiones para seguir vectorizando. Una conexión presente en cualquier individuo permanece; no se aplica un umbral a conexiones débiles. El grafo completo se utiliza para las derivadas implícitas del diseño, porque la derivada respecto a una presencia cero puede ser distinta de cero. Cuando no se necesitan derivadas del diseño, se reutiliza además el último residuo de Newton en la corrección final, evitando otra evaluación de la energía diferenciada.

Comparación CPU con un hilo, misma población inicial de ocho individuos, cuatro incrementos, calentamiento y cinco repeticiones alternando la versión anterior y la nueva, sin ejecutar tests simultáneamente: mediana de evaluación completa **496 ms → 418 ms**, aproximadamente **16 % menos tiempo**, con el mismo objetivo (0.18441602930945195). El lote pasó de 118 a 60 conexiones. Una medición aislada de Hessianas dio 13.74 → 12.07 ms. El ahorro depende de la unión de conexiones presentes en cada lote; si todas están presentes, no se espera ese beneficio. Estas cifras no son un benchmark de XPU ni una reducción medida del RSS.

Las pruebas comprueban energía, residuo y tangente equivalentes antes/después de eliminar conexiones en 2D y 3D, y comparan las derivadas implícitas respecto a todas las presencias, incluidas las nulas, con recorridos individuales sin esa eliminación. La suite completa pasa 38 pruebas, incluidas las comprobaciones CPU/XPU existentes.

## Correcciones del planteamiento del cálculo

Antes de las correcciones se instrumentó qué miembros del lote necesitaban una Hessiana. En CPU, con población 16, cuatro incrementos y tres generaciones desde la semilla por defecto, hubo 2.116 evaluaciones individuales de Hessiana dentro de 301 llamadas batched. **1.088 correspondían a miembros ya convergidos o deshabilitados** al empezar la llamada: aproximadamente 51 %. La evaluación inicial tuvo cero casos de este tipo. El contador no incluía todos los abandonos por fallo de line search, por lo que podía subestimar el trabajo innecesario. El porcentaje de cálculos evitables no implica el mismo porcentaje de ahorro de tiempo, especialmente en XPU.

Ahora:

- **Lotes activos:** Newton reúne solo los diseños activos para calcular residuos, tangentes y direcciones. El line search también reúne únicamente los candidatos pendientes. Los resultados se devuelven a sus índices originales; convergencia y fallos siguen siendo independientes.
- **Tangentes locales:** cada viga se diferencia en seis coordenadas de extremos en 2D y doce en 3D. Se ensambla la tangente global con el Jacobiano de la configuración y la segunda derivada de la transformación del sólido móvil, necesaria con rotaciones finitas. La misma energía por elemento se usa para calcular el objetivo y las tangentes. El solve global sigue siendo denso.
- **Estabilización en el dispositivo:** se prueba una dirección de Newton regularizada. Si el solve falla o la dirección no es finita y descendente, se utiliza un desplazamiento diagonal con una cota de Gershgorin que hace la matriz definida positiva. El line search comprueba el potencial real. Los autovalores quedan como diagnóstico una vez por intento de equilibrio, fuera del bucle Newton; siguen calculándose en CPU porque XPU no dispone de ese eigensolver. Se comprobó que Cholesky también hacía fallback a CPU en este entorno y se evitó introducirlo como sustituto.
- **Recorridos reutilizados:** se guardan coordenadas generalizadas y diagnósticos de los estados aceptados, separados del grafo y en CPU. El visor reconstruye configuración, energía y reacción sin resolver otra vez. El gradiente adjunta derivadas implícitas a los equilibrios almacenados y tampoco repite Newton. En las verificaciones controladas por carga, cambiar solo el objetivo reutilizaba el recorrido mecánico; en el análisis natural cambiar la consigna lo invalida. Las claves incluyen geometría, radios y presencia; tras la selección se conservan únicamente los recorridos de la población actual.
- **Backtracking adaptativo:** cada padre acepta su primera propuesta finita que no empeore su puntuación. Solo los pendientes prueban escalas menores. El máximo de intentos sigue siendo configurable. Es un cambio explícito de política: ya no se exploran todas las escalas para escoger entre ellas.

También se corrigieron los diagnósticos de continuación: un miembro terminado conserva el residuo y los diagnósticos de su último estado aceptado en lugar de recibir valores de un intento posterior deshabilitado.

### Comparación de extremo a extremo

Dos procesos finitos, CPU con un hilo, semilla 7, población 16, lotes de ocho y cuatro incrementos. Cada proceso evaluó el diseño inicial, tres generaciones y un paso de gradiente. No había tests ejecutándose simultáneamente. Son mediciones de una ejecución, con calentamiento incluido en la evaluación inicial; no son garantías de velocidad ni un benchmark estadístico:

| Operación | Antes | Después |
| --- | ---: | ---: |
| Evaluación inicial y visualización | 3.11 s | 1.75 s |
| Primera generación | 1.22 s | 0.77 s |
| Segunda generación | 3.53 s | 2.33 s |
| Tercera generación | 5.41 s | 2.07 s |
| Paso de gradiente | 7.92 s | 0.60 s |

En el paso de gradiente se evaluaron **16 propuestas en vez de 64**: todos los padres aceptaron el primer intento. El mejor objetivo fue 0.06736863421979 en ambas ejecuciones, aunque la población superviviente cambia debido a la política de backtracking. Las tres generaciones anteriores también seleccionaron los mismos mejores objetivos dentro del error numérico. La estabilización puede cambiar el número de incrementos adaptativos, por lo que no se exige identidad de todos los pasos de Newton.

En el diagnóstico CPU adicional (dos generaciones, cuatro incrementos), la RAM residente pasó de 710 a 713 MiB. La caché mantuvo 16 recorridos con **24.960 bytes de coordenadas**, además de metadatos. El gradiente tardó 0.41 s. En XPU, el diagnóstico sincronizado dio 1.64 s para el gradiente, un pico de tensores de 5.65 MiB y RAM alrededor de 3.56 GiB tras calentamiento. Las secuencias CPU y XPU no son idénticas; esos tiempos no son una comparación directa entre dispositivos. La retención de RAM nativa XPU persiste y estas correcciones no se presentan como una solución a ella.

Validación: 51 pruebas, incluyendo tangentes locales frente a Hessianas globales con deformación en 2D/3D y CPU/XPU, derivadas implícitas de recorridos guardados frente a solves nuevos, diferencias finitas existentes, exclusión real de miembros terminados, estabilización de matrices indefinidas y singulares en CPU/XPU, backtracking independiente, visor sin continuación adicional, reutilización al cambiar el objetivo y caché acotada sin grafos retenidos.

Artefactos locales: `runs/diagnostics/refactor-before.json`, `refactor-after.json`, `refactor-cpu/cpu-profile.json` y `refactor-xpu/xpu-profile.json`.

## Navegador

Se midieron los datos y el heap con Chrome DevTools Protocol en Chromium headless. El contexto WebGL de esta prueba usó SwiftShader: **no mide el rendimiento real de la GPU del navegador del usuario**.

- La respuesta completa de la UI tenía aproximadamente 64 KB de JSON.
- Tras 100, 500 y 1.000 renders, el heap recogido se mantuvo alrededor de 3–3.2 MiB. Los nodos DOM y listeners permanecieron constantes.
- Había churn: antes de GC se observaban decenas de MiB temporales. Las tarjetas de población se regeneraban innecesariamente en cada fotograma.
- Se conservan ahora las tarjetas hasta que cambia la respuesta y se reutiliza el buffer de subida WebGL. En la misma prueba, 1.000 renders pasaron de 5.36 a 1.92 ms por render. La identidad de las tarjetas se conservó y no hubo errores WebGL.

Estas medidas no apoyan una fuga creciente de JavaScript como causa principal. No incluyen todo el RSS de Chromium ni allocations del driver gráfico del navegador.

## Repetir la medición

```sh
uv run python -m compliant_mechanism_synthesis.diagnostics --device cpu
uv run python -m compliant_mechanism_synthesis.diagnostics --device xpu
```

El diagnóstico ejecuta una búsqueda temporal, mide fases de energía/residuo/Hessiana, RSS/PSS, memoria XPU asignada/reservada y su pico, y escribe JSON en `runs/diagnostics/`. Se cierra al terminar: no abre servidores. Para 3D, añadir `--dimension 3`; esas pruebas pueden ser considerablemente más lentas.

Las capturas de esta investigación están en `runs/diagnostics/`: `cpu-lifecycle.json`, `xpu-runtime.json`, `xpu-lifecycle.json`, `xpu-isolation.json`, `xpu-cleanup.json`, `browser.json`, `browser-after.json`, `gradient.txt` y `evolution_5.txt`. Son artefactos locales excluidos de Git.

## Población, controles y recorrido con reposo interior

Los nuevos valores por defecto son 32 estructuras y lotes de hasta 16. Se pueden ajustar por separado en la especificación y durante la síntesis, junto con ciclos de gradiente/evolución, tasas de paso por variable, pesos y escalas de posición/orientación y componentes de similitud. La comparación de diversidad ocurre fuera del FEM y no retiene grafos. Tras cada selección, la caché sigue limitada a la población retenida. Cambiar pesos del objetivo reutiliza los equilibrios; cambiar los parámetros de continuación o el progreso del reposo invalida el recorrido mecánico.

La similitud usa productos matriciales, con memoria O(N² + N·características), evitando arrays de todos los pares por todas las vigas. Compara radios solo en vigas comunes; los parámetros de vigas ausentes no aportan novedad ni invalidan la caché física. La suite final pasa 62 pruebas, incluidas estas propiedades, selección de variantes distintas frente a clones y ajuste de población/pesos durante la búsqueda.

Un reposo interior añade el coste físico de preparar el punto inicial mediante precarga desde el estado sin carga. Los recorridos guardados, incluida su carga con signo, se reutilizan para sus derivadas y su visualización. Se compararon los recorridos negativos→reposo→positivos y sus gradientes contra solves individuales en 2D/3D y CPU/XPU. El punto de reposo se evalúa explícitamente aunque no coincida con el paso nominal.

Validación adicional en Chromium headless con SwiftShader: crear una consigna con reposo central, ampliar una población de cuatro a ocho sin reiniciar, limitarla a lotes de dos, cambiar pesos y ejecutar dos pasos de gradiente seguidos de una evolución. No hubo errores JavaScript ni WebGL. El servidor temporal se cerró al terminar. Captura local en `runs/diagnostics/search-controls-ui.png`. Esto comprueba el flujo web, no el rendimiento de la GPU del navegador del usuario.
