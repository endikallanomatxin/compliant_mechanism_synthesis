# Performance

Las comparaciones de condensación del ejemplo canónico se realizaron con la distribución regular de 30 nodos y 88 vigas, antes de sustituirlo por la distribución aleatoria reproducible. La aplicación actual usa únicamente 3D y conexiones iniciales por vecinos próximos. Los comandos actuales usan el nuevo ejemplo; sus tiempos y recorridos pueden diferir aunque se conserve el número de nodos y vigas.

## Paso de gradiente: ruta predeterminada 3D

Los registros de una sesión real mostraron 82,3 s por paso de gradiente con 32 alternativas. El perfil de cuatro alternativas localizó reconstrucciones de rigidez en la ruta de una sola división por viga, que aún utilizaba ensamblaje y derivadas de 12 coordenadas. Esa ruta ahora comparte el kernel de nueve coordenadas y ensamblaje local de las vigas subdivididas. La tangente usada como inversa implícita se construye con datos separados del grafo de gradiente; una corrección basta para la primera derivada espacial del equilibrio. La rigidez final sigue siendo diferenciable si la pérdida de soporte la necesita. Si todos los estados de un recorrido almacenado superan el mínimo de soporte, se omite su grafo de derivadas, pues la penalización es localmente cero. La geometría de colisiones y longitudes se interpola solo para la unión de vigas presentes, en lugar de las 519 conexiones admisibles.

Comparación puntual sin profiler, CPU float64, un hilo, cuatro alternativas idénticas del ejemplo, configuración predeterminada de recorrido y objetivo, un intento de paso, caché inicial completa: 23,56 → 10,79 s para calcular gradientes y evaluar el paso propuesto (54,2 % menos). Posiciones, radios parametrizados y puntuaciones resultantes coinciden con rtol 1e-5 y atol 1e-7. Había servidor y otras verificaciones activos; no es una mediana ni una medida de una población de 32 o de XPU. No se redujeron puntos del recorrido, subdivisiones, población ni componentes de la pérdida.

## Mutaciones acotadas y lotes de topologías similares

La mutación anterior aplicaba probabilidades por cada una de las 519 conexiones posibles. En el ejemplo de 85 vigas, sus expectativas eran 17,9 adiciones frente a 2,85 eliminaciones. El nuevo presupuesto aplica hasta dos ediciones distintas, con probabilidad 0,6 por edición, y elige añadir/eliminar antes de seleccionar una conexión ponderada por distancia. Con ambas operaciones disponibles y sesgo 0,5, sus expectativas son 0,6 adiciones y 0,6 eliminaciones. Las variantes locales reducen la probabilidad de aplicar cada edición. La recombinación exploratoria sigue siendo capaz de introducir cambios mayores que ese presupuesto de mutación.

Los lotes agrupan grafos similares y limitan la relación entre vigas de la unión y vigas presentes medias (por defecto 2). La población inicial de 16 alternativas pasa de unas 100 vigas medias y una unión de 279 a 84,75 y 92 con el nuevo muestreo. También separan recorridos almacenados con diferentes mallas de continuación, y candidatos sin caché. Así no se pierde la reproducción sin Newton por mezclar estados incompatibles. El solver subdividido conserva dos correcciones implícitas al pedir gradientes; hace una sola limpieza numérica al evaluar sin derivadas.

Medición puntual CPU float64, un hilo, con servidor y otras pruebas activos: 16 estructuras distintas, entrada de 0,05 mm, dos incrementos, dos segmentos por viga, máximo Newton 20 e incremento mínimo 0,01; objetivo completo con colisiones y longitudes. Se calentó con un diseño antes de medir y se vació la caché.

| Evaluación | Tiempo | Convergentes |
|---|---:|---:|
| Versión anterior, población anterior | 29,52 s | 16/16 |
| Versión nueva, exactamente la misma población | 14,60 s | 16/16 |
| Versión nueva, nueva población inicial | 6,29 s | 16/16 |

En el mismo problema, el ahorro medido es 50,5 % y la diferencia máxima de puntuación 5,8e-15. Los nuevos lotes calculan 3.106 slots de vigas frente a 4.464 (279×16), para 1.580 vigas realmente presentes. La nueva población usa 1.472 slots para 1.356 vigas presentes (92,1 % útiles). La tercera fila contiene diseños diferentes y no es una aceleración del mismo cálculo. Es una evaluación puntual, no una mediana ni una medida de generaciones completas o XPU.

## Rigidez de segmentos espaciales: menos derivadas redundantes

El perfil de un recorrido 3D mostró que el principal coste era construir las fuerzas y Hessianas locales durante Newton. El kernel utilizaba 12 coordenadas por segmento, aunque la energía depende solo de 9: desplazamiento relativo y dos rotaciones. Ahora deriva en esas 9 coordenadas y reconstruye exactamente los bloques de los extremos. La misma evaluación entrega energía, fuerzas y rigidez; elimina dos evaluaciones separadas. No cambia la energía física, la discretización ni el número de muestras de continuación.

Medición CPU float64, un hilo, medianas tras calentamiento, con un servidor de usuario activo:

| Trabajo idéntico antes/después | Antes | Después | Reducción |
|---|---:|---:|---:|
| Kernel: 8 diseños, 72 vigas, 4 segmentos (7 repeticiones) | 205 ms | 148 ms | 28 % |
| Recorrido: 2 diseños, 33 nodos, 85 vigas, 2 segmentos, entrada 0,6 mm, 2 incrementos (3 repeticiones) | 799 ms | 689 ms | 14 % |

Ambos recorridos completos convergen y sus poses finales coinciden (rtol 1e-7, atol 1e-9). Las energías, fuerzas y rigideces del kernel coinciden con la versión anterior (rtol 1e-9, atol 1e-7). Son mediciones locales, no tiempos de una generación completa ni de XPU. La nueva geometría inicial tiene tres uniones no colineales por sólido, que añaden vigas; la comparación del recorrido usa esa misma geometría en ambas versiones.

## Predictor de continuación entrada–salida

Cada estado estable conserva temporalmente la sensibilidad `dq_unknown / dq_active = -H_uu⁻¹ H_ua` para predecir el siguiente equilibrio. La predicción y las decisiones de subdivisión están separadas del grafo de gradiente; las derivadas del estado corregido siguen siendo implícitas. La corrección de pose pasiva se compara con tolerancias configurables de posición (m) y orientación (rad, distancia entre rotaciones en 3D). Se subdivide también cuando Newton converge pero esa corrección es grande. Es un indicador local, no una cota certificada de interpolación ni de unicidad de rama. El incremento mínimo sigue limitando las subdivisiones. El predictor no se ejecuta en la reproducción de estados almacenados; la precarga desde reposo conserva la continuación previa.

Medición calentada CPU, un hilo, diseño 2D canónico, entrada 3 mm y 12 incrementos: sin predictor 0,481 s y 31 iteraciones Newton aceptadas; con predictor 0,362 s y 12 iteraciones, ambos con 13 estados. Es una medición local de un recorrido, no una garantía para la población o XPU. Las pruebas comparan equilibrio final y derivadas en 2D/3D, y verifican que tolerancias más estrictas refinan aun con convergencia de Newton.

Beam presence is now strictly binary. Evolution exclusively controls topology; gradient descent optimizes positions and radii. The input-output continuation prunes the union of present beams once per path, rather than repeating `nonzero`/indexing for each increment. Cached implicit equilibrium corrections no longer prepare a detached Newton batch. These changes reduce setup and dynamic-size queries without changing the physical equations; regression tests count pruning calls and reject Newton preparation during cached replay. Earlier measurements below describe their recorded configurations, including continuous beam weights where applicable.

## Lotes fijos con máscaras en Newton

`ContinuationConfig.newton_batch_mode` permite `auto`, `compact` y `masked`, también en la pestaña Análisis de la configuración de síntesis. Auto compacta en CPU y conserva las dimensiones del lote en XPU. Los miembros convergidos quedan congelados; sus sistemas se sustituyen por identidades y sus residuos por cero antes del solve. Los fallos individuales no invalidan los demás miembros. El equilibrio final, la estabilidad y las derivadas implícitas mantienen las mismas comprobaciones.

Comparación calentada de tres repeticiones de equilibrio 2D, 30 nodos, entradas entre 0 y 0,6 mm, en segundos:

| Dispositivo | Diseños | Compactación | Máscaras |
|---|---:|---:|---:|
| CPU | 1 | 0,0576 | 0,0524 |
| CPU | 4 | 0,0734 | 0,0814 |
| CPU | 16 | 0,1460 | 0,1478 |
| XPU | 1 | 0,1960 | 0,1886 |
| XPU | 4 | 0,3592 | 0,3109 |
| XPU | 16 | 0,3737 | 0,3550 |

Había un servidor activo: son medidas locales del equilibrio, no de una generación completa ni pruebas aisladas de hardware. Las máscaras evitan `nonzero` y selección dinámica dentro de Newton, a cambio de evaluar también miembros terminados. Siguen existiendo decisiones escalares para terminar Newton y backtracking, y compactación en la continuación. Se verifican equilibrio, conteos y derivadas en 2D/3D, CPU/XPU, además de aislamiento de fallos.

La fusión también se probó por separado con `torch.compile` del kernel analítico de vigas 2D: 3,57 → 0,64 ms por evaluación de 88 vigas tras calentamiento, con unos 15 s de compilación inicial. El entorno carecía de las cabeceras Level Zero y del ejecutable `ocloc`; la prueba usó paquetes extraídos en un directorio temporal, sin modificar el sistema. Las derivadas respecto a radios del kernel compilado coinciden con las del kernel eager. Esto demuestra margen en el kernel aislado, pero no mide el recorrido completo ni sus variantes de forma. La compilación no está activada en la aplicación: todavía requiere integrar y medir los lotes, autograd, cambios de topología y costes de recompilación.

## Revisión de trabajo innecesario dentro del cálculo vectorizado

Se identificaron y corrigieron tres duplicaciones en el núcleo:

- Newton ensamblaba una tangente antes de comprobar si el residuo ya convergía. Ahora comprueba el residuo primero y calcula la dirección solo para miembros pendientes. La estabilidad y las derivadas siguen usando la tangente final.
- La Hessiana local mediante `jacrev(grad(...))` ya calcula el gradiente elemental, pero se calculaba una segunda vez para la curvatura del sólido rígido. `has_aux=True` permite reutilizarlo en 3D.
- En 2D, el gradiente y Hessiana de la energía corrotacional tienen expresiones analíticas. Se usan para residuos y tangentes, incluyendo fuerza axial, curvatura de la dirección de la viga, acoplamiento giro-flexión y rotación de los cuerpos. Todas las expresiones continúan diferenciables respecto al diseño; se verifican contra derivadas de la energía y diferencias finitas del recorrido.

Comparación calentada de tres repeticiones del mismo equilibrio 2D, un diseño de 30 nodos con entrada de 0,3 mm: CPU 0,165 → 0,071 s; XPU 1,215 → 0,673 s. Los resultados coinciden con tolerancia relativa 1e-7 y absoluta 1e-10. La máquina tenía un servidor activo y otras verificaciones: son mediciones de este equilibrio, no un speedup garantizado para una generación completa o para 3D.

Las trazas de PyTorch CPU/XPU de ese equilibrio registraron 7.601 → 5.245 kernels. El tiempo acumulado de kernels fue 307 → 185 ms. Los tiempos del runtime y las anotaciones del profiler dependen fuertemente de la instrumentación y la concurrencia; no equivalen al tiempo matemático ni deben sumarse a las categorías anidadas. Continúan 28 llamadas `nonzero` y 30 extracciones escalares: la continuación/compactación adaptativa conserva decisiones que sincronizan el dispositivo. No se ha fusionado todo el cálculo ni eliminado toda la fragmentación.

Para generar una traza reproducible (con calentamiento previo):

```sh
uv run python -m compliant_mechanism_synthesis.trace_diagnostics --device xpu --dimension 2 --batch-size 1
```

Escribe timeline compatible con Chrome/Perfetto, tabla por operación y resumen de operaciones de control en `runs/diagnostics/`. Las trazas de esta revisión están en `xpu-equilibrium-before-analytic-trace.json` y `xpu-equilibrium-after-analytic-trace.json`; el resumen comparativo está en `xpu-equilibrium-review.json`.

## Revisión de lentitud con Intel XPU

En la máquina actual, una medición instrumentada 2D de cuatro estructuras y dos incrementos dio CPU 2,30 s para evaluación inicial y 1,70 s para gradiente; Intel XPU 7,00 s y 8,05 s. En XPU, residuos y tangentes sumaron 4,34 s inicialmente y 4,43 s en gradiente; estabilidad y espectros pequeños, 0,65 s y 0,82 s. RSS fue aproximadamente 0,73–0,81 GB en CPU y 3,64–3,73 GB en XPU. La instrumentación sincroniza cada fase y había un servidor del usuario activo: son cifras de diagnóstico, no un benchmark aislado ni una garantía de rendimiento.

Se corrigió la selección automática: la medición anterior construía coordenadas para un único sólido y ya no correspondía al modelo activo/pasivo. Ahora mide continuación no lineal real con los dos cuerpos, calentamiento previo y lotes acotados. CPU continúa siendo el valor por defecto; Auto elige por tiempo medido. Seleccionar XPU explícitamente sigue usando XPU, aunque resulte más lento.

El Jacobi XPU del espectro efectivo pequeño rota parejas disjuntas en paralelo, reduciendo los grupos secuenciales de 150 a 40 para matrices 6×6. En una comparación de cinco repeticiones calentadas de lotes de cuatro matrices 6×6, el componente espectral pasó de 31,5 ms a 16,8 ms. Se comprueban espectros, autovectores y derivadas contra CPU. También se elimina una tangente intermedia innecesaria de las correcciones primales; se conserva cuando es necesaria para las derivadas implícitas. Estas mejoras no convierten esta carga de matrices pequeñas y autograd en una tarea necesariamente favorable a GPU: las operaciones cortas y sus lanzamientos/sincronizaciones siguen dominando. Una repetición 2D concurrente con pruebas no mostró una mejora global concluyente, por lo que no se atribuye un speedup total a estos cambios.

## Continuación no lineal entrada–salida

El análisis principal usa `prescribed_motion.py`: se prescribe el sólido activo y se equilibran simultáneamente el sólido pasivo y los nodos libres mediante energía corrotacional, tangentes actualizadas y continuación adaptativa. El gradiente usa derivadas implícitas sobre los estados almacenados, evitando repetir Newton. Cambiar la consigna pasiva reutiliza el recorrido; cambiar la entrada activa lo invalida. Las dos transformaciones rígidas incluyen su curvatura en la tangente.

La estabilidad se comprueba con pivotes LDL positivos en el dispositivo, evitando un eigensolve global o Cholesky con fallback a CPU en Intel XPU. Solo la rigidez efectiva pequeña del pasivo (3×3 o 6×6) necesita un espectro; XPU usa Jacobi. Newton y line search compactan diseños pendientes en CPU; en XPU conservan el lote con máscaras para evitar consultas de tamaño dinámico. Las cachés quedan limitadas a la población retenida y contienen tensores separados del grafo de gradiente.

Medición del análisis entrada–salida actual en CPU, float64, un hilo, población 8, lotes de 4 y 12 incrementos: evaluación inicial 2,95 s, dos pasos de gradiente 3,47 s y 3,53 s. Pico RSS acumulado: 714,5 MB, 845,4 MB y 857,8 MB. Son mediciones de esta configuración, no garantías para cualquier estructura ni del consumo nativo de XPU.

El ejemplo ampliado tiene 30 nodos, 88 vigas iniciales y 432 conexiones candidatas. En una prueba 3D con cuatro individuos y seis incrementos, excluir las conexiones ausentes del grafo diferenciable redujo el pico RSS de aproximadamente 3,4 GB a 1,7 GB. Se conservan las derivadas de las conexiones presentes y los logits de las ausentes tienen derivada cero por su máscara discreta. Una comprobación dedicada compara objetivo y gradientes con el cálculo sin esa exclusión. Los lotes de gradiente tienen un límite independiente, por defecto 4, para no construir 16 grafos grandes simultáneamente.

Los descendientes evolutivos se refinan mediante gradiente antes de la selección. Durante varios pasos de refinamiento se podan las cachés intermedias, conservando solo padres y descendientes actuales. Este trabajo adicional es configurable y su tiempo se incluye en los diagnósticos de evolución.

Las mediciones siguientes corresponden a verificaciones mecánicas y formulaciones previas, no al coste del análisis entrada–salida actual. Para perfilarlo ejecuta `uv run python -m compliant_mechanism_synthesis.diagnostics`.

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

## Experimento de subdivisión y condensación 2D

`subdivided.py` evalúa segmentos de todas las vigas en un lote y ensambla sistemas internos privados. Condensa residuo y tangente por complemento de Schur, resuelve las uniones y recupera los movimientos internos en cada iteración global. No ejecuta un Newton independiente por viga. Ese experimento inicial precedió a la integración entrada–salida descrita a continuación. La disposición de componentes sigue intercalada para aislar la comparación física del experimento de almacenamiento.

El experimento reproducible `uv run python -m compliant_mechanism_synthesis.subdivision_diagnostics --device cpu --batch-size 16 --segments 4` usa dos vigas conectadas por una unión libre y extremos prescritos. Con cuatro segmentos conserva 9 grados nodales globales y elimina 18 privados por diseño. Tres repeticiones calentadas dieron CPU 10,48/10,86/10,50 ms; XPU 180,89/183,23/185,98 ms. Los 16 casos convergieron. Son tiempos de un ejemplo pequeño, no de la población canónica ni una comparación condensado/sin condensar de rendimiento. La equivalencia se comprueba contra el sistema completo para pasos de Newton y solución no lineal; las derivadas, contra diferencias finitas. También se comprueba convergencia al refinar de 2 a 4 y 8 segmentos.

## Condensación integrada en síntesis 2D — 14 de octubre de 2026

`ContinuationConfig.beam_segments` controla la discretización privada de cada viga; uno conserva el núcleo de un segmento. Con más segmentos, `condensed_mechanics.py` conserva empotramientos y sólidos activo/pasivo exactos, curvatura de sus transformaciones, Newton global, recuperación de interiores, predictor y derivadas implícitas. No hay Newton independiente por viga. La caché CPU guarda únicamente interiores de vigas presentes, sin grafos; reconstruye extremos al reproducir el recorrido. Cambiar segmentos invalida la caché. El visor y los diagnósticos usan los segmentos calculados; la evolución sigue operando sobre el grafo de diseño. La subdivisión 3D se rechaza explícitamente; el análisis 3D de un segmento sigue disponible.

Se eliminó un intermedio de tamaño lote × vigas × incógnitas² del ensamblaje. CPU transforma contribuciones locales 6×6 y las suma al sistema global. XPU usa contracciones matriciales sobre el eje aplanado de extremos: el scatter con atómicos float64 fue más lento en esta Intel. Tampoco se prepara el Jacobiano de ligaduras cuando solo se necesita energía para backtracking. La disposición de componentes sigue intercalada: estos resultados no incluyen un cambio a SoA.

### Recorrido completo, misma discretización

Diseño canónico de 30 nodos y 88 vigas, entrada de 0,6 mm, float64, CPU con un hilo, calentamiento y procesos separados. La referencia deja cada nodo interno como nodo libre del FEM, con las mismas energías y ligaduras. Se comprobaron poses a lo largo del recorrido, malla final y factores de continuación; coinciden dentro de tolerancia. Son tiempos del recorrido primal, no del backward ni de una generación completa.

| Dispositivo | Diseños en lote | Segmentos/viga | Condensado | Malla completa | Aceleración | Incógnitas globales/diseño |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 4 | 0,133 s | 15,614 s | 117,1× | 75 frente a 867 |
| CPU | 4 | 2 | 0,161 s | 8,439 s | 52,4× | 75 frente a 339 |
| Intel XPU | 1 | 2 | 0,669 s | 4,762 s | 7,1× | 75 frente a 339 |
| Intel XPU | 4 | 2 | 2,349 s | No terminó | Sin comparación | 75 frente a 339 |

Condensado: tres repeticiones; referencias completas: dos. CPU con cuatro segmentos aceptó seis estados y 15 iteraciones Newton. Los casos de dos segmentos aceptaron cinco estados y 13 iteraciones por diseño. La referencia XPU de cuatro diseños agotó 90 s durante el calentamiento; una ejecución anterior también quedó detenida varios minutos. No se ha aislado la causa de ese bloqueo nativo y no se infiere una aceleración para ese caso. El benchmark limita los procesos y registra el timeout, sin dejar trabajadores ejecutándose.

RSS máximo CPU, un diseño/cuatro segmentos: **498 frente a 838 MiB**. XPU, un diseño/dos segmentos: tensores **1,80 frente a 19,36 MiB**, RSS **2,77 frente a 2,94 GiB**. Para cuatro diseños XPU, retirar el intermedio por viga redujo el pico de tensores de 22,59 a **6,83 MiB**. La retención nativa del runtime XPU persiste; no equivale a memoria de tensores ni se da por resuelta.

El modelo de un segmento tardó 0,117 s en CPU (cuatro incrementos) y 0,265 s en XPU (dos incrementos). Es una aproximación más gruesa y acepta menos muestras. **117× compara dos formas de resolver la misma malla fina; no significa acelerar 117× el programa anterior.** En estas cargas completas la CPU sigue siendo más rápida.

Artefactos locales: `condensation-cpu-final-engine.json`, `condensation-cpu-final.json`, `condensation-cpu-batch4-sparse.json`, `condensation-cpu-batch4.json`, `condensation-xpu-final-engine.json`, `condensation-xpu-final.json` y `condensation-xpu-batch4-final-engine.json`, bajo `runs/diagnostics/`. Los tiempos de referencia proceden de ejecuciones aisladas anteriores del mismo solver sin condensar; las salidas de la implementación final se comprobaron contra esos artefactos. Cada método escribe también TensorBoard.

### Escalado de las operaciones intraviga

`beam_scaling.py` mide únicamente residuos/tangentes de segmentos y condensación local, sin equilibrio global ni continuación. Float64, cuatro segmentos por viga, 88 vigas por diseño, CPU con un hilo y tres repeticiones sincronizadas por tamaño. Comprueba equivalencia numérica CPU/XPU. Con lotes pequeños ganó CPU; con **256 diseños, 90.112 segmentos**, cálculo y condensación tardaron de media **0,131 s en CPU frente a 0,089 s en XPU**, aproximadamente 1,48× de rendimiento. Hubo variabilidad y un valor extremo XPU en un lote menor; no debe extrapolarse a generaciones completas.

Esto confirma que aumentar trabajo independiente puede amortizar lanzamientos y aprovechar la GPU. Aumentar el tamaño del sistema global denso no ofrece la misma garantía. La condensación permite refinar vigas sin aumentar las incógnitas del solve global, pero los lotes de síntesis actuales siguen siendo pequeños para esta GPU.

```sh
uv run python -m compliant_mechanism_synthesis.subdivision_benchmark --device cpu --segments 4 --steps 4 --repeats 3
uv run python -m compliant_mechanism_synthesis.subdivision_benchmark --device xpu --segments 2 --steps 2 --batch-size 4 --timeout-seconds 90
uv run python -m compliant_mechanism_synthesis.beam_scaling
```

`--mode condensed|full|single` mide un solo método; el modo por defecto compara procesos aislados. `beam_scaling` guarda sus resultados en `runs/diagnostics/beam-scaling.json`. Las pruebas de integración comparan recorrido y gradientes de posiciones/radios con mallas sin condensar en CPU/XPU, caché sin Newton, topologías distintas en un lote y precarga a ambos lados del reposo. La comprobación web incluye cambio de resolución, gradiente, evolución y paso a 3D.
