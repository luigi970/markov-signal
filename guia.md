# 📘 Guía de Interpretación: Regímenes de Mercado (Markov)

Este documento explica de forma sencilla qué son los "Regímenes" que detecta nuestro script y cómo usarlos para tomar mejores decisiones de inversión.

---

## 🧐 1. ¿Qué es un "Régimen"?
En el mundo de las finanzas, un **Régimen** es el "estado de ánimo" o el "clima" del mercado en un momento dado. 

El mercado no se comporta siempre igual. A veces está tranquilo y sube despacio, otras veces entra en pánico y cae bruscamente. El **Modelo de Markov** es una inteligencia matemática que agrupa los días de la historia que se parecen entre sí para decirnos en qué "estación climática" estamos hoy.

---

## 🚦 2. Los 3 Estados de Ánimo del Mercado

El modelo suele clasificar tus acciones en tres categorías principales:

### 🟢 Régimen de "Baja Volatilidad" (El Puerto Seguro)
* **Qué es:** El mercado está en calma. Los inversores tienen confianza.
* **Cómo se ve:** El precio sube o baja de forma constante, sin saltos bruscos. Las velas del gráfico son pequeñas.
* **Riesgo:** **Bajo**. Es el mejor momento para inversores conservadores.
* **Estrategia:** Mantener o comprar. El camino es despejado.

### 🟡 Régimen de "Transición / Lateral" (La Indecisión)
* **Qué es:** El mercado está "esperando" algo. No hay una dirección clara.
* **Cómo se ve:** El precio sube un poco un día y baja lo mismo al siguiente. El volumen de transacciones suele ser inestable.
* **Riesgo:** **Moderado**. Es un estado de "ver y esperar".
* **Estrategia:** No hacer movimientos grandes. Esperar a que el modelo detecte un salto a un régimen más definido.

### 🔴 Régimen de "Alta Volatilidad / Nerviosismo" (La Tormenta)
* **Qué es:** Los inversores están asustados o hay mucha especulación.
* **Cómo se ve:** El precio da "latigazos" (sube y baja un 2% o 3% en minutos). Hay mucha incertidumbre.
* **Riesgo:** **Alto**. Es donde la mayoría de los traders pierden dinero por movimientos imprevistos.
* **Estrategia:** Proteger el capital. Ajustar los *Stop Loss* o reducir el tamaño de las inversiones.

---

## 📊 3. ¿Cómo leer los resultados del Script? (Diccionario de Regímenes)

El modelo de Markov clasifica el clima financiero basándose en el **Retorno Medio** (dirección) y la **Volatilidad** (nerviosismo/riesgo):

| Si el resultado muestra... | El Régimen es... | Interpretación Real (Acción sugerida) |
| :--- | :--- | :--- |
| **Retorno (+) / Volatilidad Baja** | **Alcista Saludable** | **Camino despejado:** El precio sube de forma sólida y constante. Es el mejor momento para mantener o comprar. |
| **Retorno (+) / Volatilidad Alta** | **Euforia Arriesgada** | **Subida con Miedo:** El precio sube pero con "latigazos" bruscos. Muy peligroso para entrar tarde; mejor ajustar *Stop Loss*. |
| **Retorno (~0) / Volatilidad Baja** | **Acumulación** | **Mercado Dormido:** No hay dirección clara. Los inversores esperan noticias. Ideal para observar y esperar la ruptura. |
| **Retorno (-) / Volatilidad Baja** | **Desangrado Lento** | **La Trampa:** El precio cae poco a poco sin que parezca una crisis. Es peligroso porque no genera pánico inmediato, pero erosiona tu capital. |
| **Retorno (-) / Volatilidad Alta** | **Pánico / Crash** | **La Tormenta:** Caídas violentas y miedo extremo. Momento de proteger el capital o buscar activos refugio. |

---

### 💡 Tips para una lectura profesional:

1. **La Confianza (100%):** No es una garantía de futuro, es una **certeza de diagnóstico**. El modelo está 100% seguro de que los síntomas de HOY pertenecen a ese régimen específico.
2. **La Persistencia:** Si un régimen suele durar **1.3 días** (como vimos en KO), no tomes decisiones de muy largo plazo basadas en él, porque es un estado fugaz.
3. **El Salto de Mañana:** Si estás en un régimen "Alcista Saludable" pero la matriz dice que hay un 40% de probabilidad de saltar a "Pánico", el modelo te está avisando que el clima está por cambiar.

---

## ❓ 4. Preguntas Frecuentes

### ¿Por qué el modelo dice "Confianza 100%"?
No significa que el futuro sea 100% seguro. Significa que los síntomas de hoy (volumen, riesgo y retorno) **encajan perfectamente** con la descripción matemática de ese régimen. Es como decir: "Estoy 100% seguro de que hoy es lunes", eso no te dice qué pasará el martes, pero sí define dónde estás parado.

### ¿Qué es la "Persistencia del Estado"?
Es cuánto tiempo suele durar ese clima. 
* Si la persistencia es de **1.3 días**, el estado es un susto pasajero. 
* Si la persistencia es de **5 días o más**, estamos ante una tendencia sólida que durará toda la semana.

### ¿Qué es la "Matriz de Probabilidades"?
Es la previsión del tiempo. Te dice qué tan probable es que mañana el mercado cambie de humor. Si la probabilidad de saltar de un régimen bueno a uno malo es alta, es tu señal de alerta temprana.

---

> **⚠️ NOTA IMPORTANTE:** Este modelo es una herramienta de probabilidad. Las noticias del mundo real (guerras, decisiones de bancos centrales) pueden cambiar el régimen de un segundo a otro. Úsalo como una brújula, no como un mapa infalible.