# Markov Signal Dashboard

Dashboard de análisis de **regímenes de mercado** con **Modelos Ocultos de Markov (HMM)**, construido en Streamlit.

Permite detectar estados de mercado (alcista, bajista, lateral, alta volatilidad), estimar su persistencia y visualizar probabilidades de transición.

## Qué hace esta app

- Descarga datos históricos con `yfinance`.
- Construye variables de mercado:
  - `Returns` (retorno logarítmico diario)
  - `Range` (volatilidad diaria relativa)
  - `Vol_Change` (cambio porcentual de volumen)
- Entrena un modelo `GaussianHMM`.
- Clasifica cada día en un régimen.
- Muestra:
  - Estado actual
  - Régimen más probable para mañana
  - Confianza del modelo
  - Persistencia estimada del estado
  - Matriz de transición
  - Diagnóstico detallado por régimen

## Stack

- Python 3.11
- Streamlit
- yfinance
- pandas
- numpy
- hmmlearn
- scikit-learn
- plotly
- matplotlib (opcional para estilos de tabla)

## Para qué se usa cada librería

- `streamlit`: interfaz web interactiva del dashboard (sidebar, tabs, métricas, tablas).
- `yfinance`: descarga de datos históricos de mercado desde Yahoo Finance.
- `pandas`: manipulación y limpieza de datos en `DataFrame`.
- `numpy`: cálculos numéricos (retornos logarítmicos y operaciones vectorizadas).
- `time`: esperas entre reintentos cuando hay límite de consultas en Yahoo.
- `hmmlearn (GaussianHMM)`: detección de regímenes con modelo oculto de Markov.
- `scikit-learn (StandardScaler)`: normalización de variables antes del entrenamiento.
- `plotly.graph_objects`: gráficos interactivos.
- `plotly.subplots (make_subplots)`: composición de gráficos en múltiples paneles.
- `matplotlib`: soporte opcional para gradientes de estilo en tablas.

## Estructura del proyecto

- `app.py`: dashboard principal.
- `test_markov.py`: script de consola para ejecución/diagnóstico manual.
- `guia.md`: guía de interpretación financiera de regímenes.
- `historial_markov.txt`: notas/historial del proyecto.
- `requirements.txt`: dependencias para local/cloud.
- `runtime.txt`: versión de Python para deploy.

## Instalación local

1. Crear y activar entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Ejecutar la app:

```powershell
streamlit run app.py
```

## Uso rápido

1. Elegí `Ticker` en el sidebar (ej: `KO`, `AAPL`, `BTC-USD`).
2. Elegí período (`1y`, `2y`, `5y`, `10y`, `max`).
3. Ajustá cantidad de estados (`2` a `5`).
4. Revisá:
   - `Lectura Rápida`
   - `Señal Operativa (beta)`
   - Tabs de `Diagnóstico`, `Probabilidades`, `Matriz`, `Modo Script`.

## Deploy en Streamlit Cloud

1. Subí el repo a GitHub.
2. En Streamlit Cloud, crear app desde ese repo.
3. Archivo principal: `app.py`.
4. Verificá que existan:
   - `requirements.txt`
   - `runtime.txt`

La app ya está preparada para:
- Fijar `auto_adjust=False` en `yfinance` (evita warning de cambio de default).
- Reintentos ante `YFRateLimitError` (límite temporal de Yahoo Finance).

## Problemas comunes

- `ModuleNotFoundError` en cloud:
  - Falta `requirements.txt` o no se instaló correctamente.
- `Too Many Requests` / `YFRateLimitError`:
  - Esperar 1-2 minutos y reintentar.
  - En Streamlit Cloud puede pasar por IP compartida.
- Datos vacíos para un ticker:
  - Verificar símbolo correcto y mercado (`BTC-USD`, `KO`, etc.).

## Notas de interpretación

La app es una herramienta probabilística de apoyo, no una recomendación financiera automática.

Para interpretación extendida de los regímenes:
- Ver [`guia.md`](guia.md)

## Roadmap sugerido

- Exportar resultados a CSV/JSON.
- Agregar backtesting básico de la señal.
- Integrar fuente de datos alternativa como fallback a Yahoo.
- Separar lógica de modelo en módulo dedicado (`model.py`) para tests unitarios.
