# Markov Signal

Dashboard en Streamlit para analizar **regímenes de mercado** con HMM.

## Para qué sirve

Ayuda a responder 3 preguntas:
1. **En qué régimen está hoy** el activo.
2. **Qué régimen es más probable mañana**.
3. **Qué tan confiable es esa señal** en datos no vistos.

No es una recomendación automática de inversión. Es una herramienta de apoyo.

## Qué mirar primero (orden recomendado)

1. **Señal Operativa**
- `Señal base (R)`: dirección principal (comprar / reducir / neutral).
- `Filtro de robustez`: calidad de esa señal con validación fuera de muestra.
- `Conclusión final`: acción sugerida combinando ambos.

2. **Validación (datos no vistos)**
- `Calidad promedio`.
- `Confianza media`.
- `Modo rápido` o `modo robusto (walk-forward)`.

3. **Probabilidades para mañana**
- Distribución de probabilidades por régimen.
- Método activo: `hard` o `soft`.

## Conceptos clave en la app

- `R0, R1, ...`: regímenes del **modelo operativo actual**.
- `WF0, WF1, ...`: regímenes del **bloque walk-forward** (validación robusta).
- No hay equivalencia 1:1 garantizada entre `R*` y `WF*`.

## Modos importantes

### Selección de estados
- **Manual**: elegís 2–5 estados.
- **Auto (AIC/BIC)**: prueba 2–5 y elige automáticamente.
  - `AIC/BIC`: menor valor = mejor equilibrio ajuste/simplicidad.
  - `BIC` penaliza más complejidad (suele elegir modelos más simples).

### Validación
- **Rápida (train/test)**: una sola partición temporal.
- **Robusta (walk-forward)**: varias ventanas en el tiempo (más exigente para uso real).

### Predicción de mañana
- **Hard**: usa solo el estado más probable de hoy.
- **Soft**: usa mezcla de estados de hoy y suele ser más estable.

## Requisitos

- Python 3.11+
- Dependencias en `requirements.txt`

## Ejecutar local

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Estructura mínima

- `app.py`: dashboard principal.
- `guia.md`: guía de interpretación.
- `requirements.txt`: dependencias.

## Nota final

Si vas a usar la señal para decisiones reales:
- preferí validación **Robusta (walk-forward)**,
- y evitá basarte solo en una métrica aislada.
