# Equilibrio reutilizable para síntesis no lineal 3D

Estado: aprobada por el usuario e implementada; resultados y límites en PERFORMANCE.md.

## Objetivo y alcance

Reducir el tiempo por mejora validada del mecanismo. Conservar grafos de vigas biempotradas, sólidos fijo/activo/pasivo independientes, entrada prescrita, equilibrio no lineal, trayectoria completa, float64 y gradientes en PyTorch sobre CPU e Intel XPU. Mantener evolución binaria y descenso directo; ningún modelo aprendido.

Primera entrega: solver normalizado, factorizaciones nativas por dispositivo, backward adjunto y reutilización conservadora entre diseños de la misma topología. La distribución adaptativa de esfuerzo y la resolución variable quedan para una entrega posterior: cambiar ambas a la vez impediría atribuir las diferencias de calidad y tiempo.

## Alternativas y decisión

1. **Solver reutilizable con adjunto explícito (elegida).** Mantiene el modelo físico, reduce trabajo repetido y permite comparar exactamente el mismo problema. Requiere gestionar memoria y verificar sensibilidad y rama de equilibrio.
2. **Newton sin ensamblar matrices, mediante productos Hessiana-vector.** Puede ayudar al crecer las estructuras, pero con unos 150 grados de libertad y mal condicionamiento no está demostrado que supere factorizaciones directas. Posponer hasta medir el solver elegido.
3. **Modelos reducidos de vigas o mecanismos.** Pueden reducir más el coste, pero introducen nuevos límites de precisión en grandes giros, torsión, pandeo y colisiones. Posponer hasta disponer de una referencia mecánica rápida y validada.

## Separación del cálculo

- El kernel elemental produce energía, residuo y tangente; conserva la condensación exacta de interiores y los términos de curvatura de las transformaciones rígidas.
- El solver primal obtiene equilibrio, verifica estabilidad y entrega datos separados del grafo de autograd.
- Un contexto de sensibilidad conserva la factorización final válida y los estados privados necesarios. El contexto de autograd vive durante una evaluación de gradiente. Sus factorizaciones separadas del grafo se pueden conservar para recorridos exactos de alternativas vivas, con presupuesto configurable de 256 MiB; no se almacenan todas las factorizaciones de toda la historia evolutiva.
- El backward usa el contexto para resolver el adjunto y obtiene los productos de derivadas respecto al diseño mediante PyTorch.
- La continuación controla rama, precisión y recuperación de fallos; la búsqueda proporciona parentesco y aproximaciones iniciales. La UI recibe resultados y diagnósticos, sin dependencia del núcleo numérico.

Aplicación bajo `src/compliant_mechanism_synthesis/`. Reemplazar interfaces internas directamente; no introducir adaptadores de compatibilidad.

## Coordenadas normalizadas

Resolver en coordenadas z con q = S z. S escala traslaciones mediante una longitud característica positiva de configuración; giros permanecen adimensionales. Aplicar Rz = Sᵀ Rq y Hz = Sᵀ Hq S. La misma transformación se usa en Newton, line search, sensibilidad, estabilidad y recuperación de interiores.

Definir tolerancias sobre residuos normalizados y reportar además fuerzas y momentos en unidades físicas. No cambiar el criterio silenciosamente: verificar que las soluciones satisfacen los límites físicos de referencia. El escalado cambia las coordenadas numéricas, nunca la energía ni las condiciones impuestas.

## Factorización y estabilidad

CPU usa Cholesky nativo con información de fallo para matrices simétricas definidas positivas; reutilizar esa factorización en los solves correspondientes. XPU usa una operación realmente soportada en dispositivo, con diagnóstico de fallo y sin fallback oculto a CPU. Verificar capacidades fuera del hot path. Mantener una alternativa en dispositivo cuando la operación nativa no exista.

Una factorización fallida no convierte un diseño inestable en válido. Las direcciones regularizadas de Newton pueden servir para buscar equilibrio; aceptación y sensibilidad exigen estabilidad de la tangente física final. No añadir regularización artificial a la puntuación mecánica.

## Backward adjunto

Para R(q,p,a) = 0, entrada a prescrita y pérdida L:

- resolver Hᵀ λ = ∂L/∂q en las incógnitas libres/pasivas;
- devolver dL/dp = ∂L/∂p − λᵀ ∂R/∂p;
- incluir la contribución a la entrada prescrita y las transformaciones rígidas exactas;
- recuperar sensitividades de interiores mediante los mismos bloques de condensación.

Las pérdidas de colisión y longitud dependen de interiores y estados interpolados: su adjunto debe incorporar esas contribuciones, no solamente la pose pasiva. La pérdida de soporte depende de la tangente y puede requerir derivadas de mayor orden; conservar su derivada cuando esté activa y comprobarla independientemente. El alcance requerido es primera derivada de la pérdida respecto al diseño; derivadas de segundo orden del optimizador quedan fuera.

Mantener el kernel y los solves en el mismo dispositivo. Usar autograd para productos de derivadas del residuo, evitando materializar un Jacobiano completo respecto a todas las variables de diseño.

## Continuación entre diseños próximos

Separar dos conceptos: caché exacta (mismo diseño, reutilización certificada del resultado) y aproximación inicial (diseño cercano, exige resolver y verificar nuevamente).

Primera entrega: aproximaciones entre diseños con la misma topología, roles, discretización e input. La búsqueda pasa explícitamente el padre; no buscar padres por comparación indiscriminada de toda la caché. Cambios de topología utilizan continuación desde reposo hasta que una extensión posterior esté validada.

Para cada paso de entrada:

1. Mantener el equilibrio del hijo en el paso anterior como referencia de rama.
2. Usar la solución del padre al progreso correspondiente como propuesta; adaptar desplazamientos relativos a las nuevas posiciones de referencia. Mapear interiores por viga y discretización, conservando extremos ligados exactamente.
3. Corregir con las ecuaciones y tangente del hijo. Una solución del padre nunca se acepta solo por proximidad.
4. Comprobar residuo, estabilidad y diferencia frente al predictor continuo del hijo. Si falla, subdividir o resolver desde su estado anterior.
5. Cuando haya ambigüedad, ejecutar el recorrido del hijo desde reposo y usar ese resultado. Conservar también la preparación desde reposo cuando el recorrido comienza predeformado.

Para cambios mayores de geometría/radio usar continuación en el parámetro de diseño con incrementos adaptativos, o rechazar el atajo. Ninguna comprobación local certifica igualdad global de rama: casos multestables exigen comparación contra el recorrido desde reposo. Los gradientes se definen sobre la rama convergente seleccionada; las decisiones discretas de aceptación no se diferencian.

## Batching, memoria y diagnósticos

Conservar agrupación por topología y malla de continuación compatible. La primera entrega mantiene la malla común por grupo para evitar cambiar simultáneamente la integración de la pérdida. Guardar estados exactos separados de aproximaciones de parentesco; invalidar por cambios de input, discretización o roles.

Liberar contextos de sensibilidad después del backward. Limitar aproximaciones a padres/hijos vivos y podarlas tras selección. Registrar en TensorBoard: tiempo primal/backward, montajes de tangente, factorizaciones, solves adjuntos, estados, iteraciones Newton, intentos de reutilización, fallbacks, ratio de vigas útiles y memoria máxima. Los contadores pertenecen a la evaluación completa y no se borran al generar el snapshot.

## Validación y aceptación

- Fuerzas/tangentes y gradientes frente a energía independiente y diferencias finitas en CPU/XPU.
- Equilibrio, exactitud de clamps/rigid ties, posición y orientación de todo el recorrido con y sin aproximaciones del padre, a progresos comunes.
- Interiores condensados frente a discretización explícita; colisiones interpoladas, longitudes y soporte activo/inactivo incluidos.
- Inicio predeformado, input con reposo intermedio, radios/geometría próximos y lejanos, fallback, estructuras inestables y casos con varias ramas.
- Pruebas de vida del contexto: ausencia de grafos retenidos y memoria estable tras varias iteraciones.
- Medir poblaciones idénticas guardadas y secuencias reproducibles, separando CPU/XPU y topología homogénea/heterogénea. Calentamiento y medianas de al menos tres repeticiones. Comparar evaluación, paso de gradiente y ciclo completo con idéntica precisión física.
- Reportar calidad por tiempo y tasa de convergencia, además de tiempo bruto. No afirmar aceleración de generaciones usando únicamente un kernel.

Adoptar cada componente cuando conserva tolerancias verificadas y mejora tiempo o memoria del trabajo representativo. Si aproximaciones de parentesco no reducen el coste total incluyendo fallbacks, conservarlas desactivadas. Las cifras previas de 38× para estabilidad y mejora del condicionamiento son diagnósticos aislados, no promesas de aceleración global.

## Límites de esta entrega

La rama estable cuasiestática y el modelo de vigas existente siguen delimitando la física. No incluir snap-through dinámico, fuerzas de contacto, plasticidad, modelos aprendidos ni cambio a topología continua. La precisión final del recorrido permanece central; ningún atajo puede presentar una trayectoria no convergente como solución válida.
