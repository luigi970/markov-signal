import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- COMPROBACIÓN DE DEPENDENCIAS ---
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Markov Signal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
<style>
    :root {
        --bg-main: #f6f9fc;
        --bg-panel: #ffffff;
        --bg-accent: #ecf4ff;
        --text-main: #14213d;
        --text-muted: #52606d;
        --brand: #006d77;
        --brand-soft: #83c5be;
        --warn: #e76f51;
        --border: #d7e3f0;
        --shadow: 0 10px 28px rgba(20, 33, 61, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 16%, rgba(131, 197, 190, 0.20) 0, rgba(131, 197, 190, 0) 38%),
            radial-gradient(circle at 88% 8%, rgba(0, 109, 119, 0.12) 0, rgba(0, 109, 119, 0) 34%),
            linear-gradient(180deg, #f8fbff 0%, var(--bg-main) 52%, #eef3f8 100%);
        color: var(--text-main) !important;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif;
        font-size: 16px;
    }

    .block-container {
        padding-top: 1.4rem !important;
    }

    .hero-banner {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        border: 1px solid rgba(0, 109, 119, 0.18);
        background: linear-gradient(135deg, #ffffff 0%, #f1f8ff 42%, #e4f1ef 100%);
        box-shadow: var(--shadow);
        padding: 20px 22px;
        margin-bottom: 1rem;
        animation: introFade 0.7s ease-out;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.3rem;
        color: var(--text-main);
        letter-spacing: -0.01em;
        font-weight: 760;
        line-height: 1.05;
    }
    .hero-banner p {
        margin: 0.35rem 0 0;
        color: var(--text-muted);
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] h1 {
        color: var(--brand) !important;
        font-size: 1.3rem !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stSlider div[data-baseweb="slider"] {
        border-radius: 12px !important;
    }

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-main) !important;
        box-shadow: 0 2px 8px rgba(20, 33, 61, 0.05);
    }

    div[data-testid="metric-container"] {
        border: 1px solid var(--border) !important;
        background: linear-gradient(165deg, #ffffff 0%, var(--bg-accent) 100%) !important;
        padding: 18px 20px !important;
        border-radius: 14px !important;
        box-shadow: var(--shadow);
    }
    [data-testid="stMetricValue"] {
        color: var(--brand) !important;
        font-weight: 760 !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    h2 {
        font-size: 1.35rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
    }

    .stPlotlyChart, .stDataFrame, .stMarkdown > div {
        animation: introFade 0.8s ease-out;
    }

    @keyframes introFade {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 768px) {
        .hero-banner h1 {
            font-size: 1.1rem;
        }
        .hero-banner p {
            font-size: 0.92rem;
        }
        .block-container {
            padding-top: 0.9rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- LÓGICA DEL MODELO ---
def _download_with_retry(ticker, period, interval="1d", retries=3, base_delay=2):
    last_error = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df, None
        except Exception as e:
            last_error = str(e)
            if "YFRateLimitError" in last_error or "Too Many Requests" in last_error:
                time.sleep(base_delay * (attempt + 1))
                continue
            return None, last_error

        time.sleep(base_delay * (attempt + 1))

    return None, last_error or "No se pudieron descargar datos."


@st.cache_data(ttl=3600)
def get_data(ticker, period="5y"):
    try:
        df, download_error = _download_with_retry(ticker, period=period, interval="1d")
        if df.empty:
            return None, download_error or "Sin datos para el ticker solicitado."

        # yfinance puede devolver columnas MultiIndex; construimos un DataFrame limpio.
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

        clean_df = pd.DataFrame(index=df.index)
        clean_df['Close'] = pd.to_numeric(close, errors='coerce')
        clean_df['High'] = pd.to_numeric(high, errors='coerce')
        clean_df['Low'] = pd.to_numeric(low, errors='coerce')
        clean_df['Volume'] = pd.to_numeric(volume, errors='coerce')

        # Ingeniería de Variables
        clean_df['Returns'] = np.log(clean_df['Close'] / clean_df['Close'].shift(1))
        clean_df['Range'] = (clean_df['High'] - clean_df['Low']) / clean_df['Close']
        clean_df['Vol_Change'] = clean_df['Volume'].pct_change()
        clean_df.dropna(subset=['Close', 'High', 'Low', 'Volume', 'Returns', 'Range', 'Vol_Change'], inplace=True)
        if clean_df.empty:
            return None, "Datos insuficientes luego de limpieza."
        return clean_df, None
    except Exception as e:
        return None, str(e)

def train_markov(df, n_states=3):
    features = ['Returns', 'Range', 'Vol_Change']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    model = GaussianHMM(
        n_components=n_states, 
        covariance_type="full", 
        n_iter=1000, 
        random_state=42
    )
    model.fit(X)
    df['Estado'] = model.predict(X)
    
    # Perfil de estados
    perfiles = {}
    for i in range(n_states):
        subset = df[df['Estado'] == i]
        ret = subset['Returns'].mean()
        vol = subset['Range'].mean()
        
        # Etiquetado inteligente con colores de alta legibilidad
        if ret > 0.0005 and vol < df['Range'].mean() * 0.9:
            tag = "Alcista Saludable"
            color = "#4ade80" # Green 400
        elif ret < -0.0005 and vol > df['Range'].mean() * 1.1:
            tag = "Pánico / Crash"
            color = "#f87171" # Red 400
        elif vol > df['Range'].mean() * 1.2:
            tag = "Extrema Volatilidad"
            color = "#fbbf24" # Amber 400
        elif ret < 0:
            tag = "Bajista / Desangrado"
            color = "#fb923c" # Orange 400
        else:
            tag = "Lateral / Acumulación"
            color = "#60a5fa" # Blue 400
            
        perfiles[i] = {"tag": f"R{i}: {tag}", "color": color, "ret": ret, "vol": vol}
        
    return model, scaler, perfiles

def build_state_diagnostics(df, perfiles, n_states):
    rows = []
    mean_range = df['Range'].mean()
    for i in range(n_states):
        subset = df[df['Estado'] == i]
        ret = subset['Returns'].mean()
        vol = subset['Range'].mean()
        freq = len(subset) / len(df) if len(df) else 0
        sesgo = "Alcista" if ret > 0.0001 else "Bajista" if ret < -0.0001 else "Lateral"
        riesgo = "Alto" if vol > mean_range else "Bajo"
        rows.append(
            {
                "Régimen": i,
                "Etiqueta": perfiles[i]["tag"],
                "Retorno Prom (%)": ret * 100,
                "Volatilidad Prom (%)": vol * 100,
                "Frecuencia (%)": freq * 100,
                "Sesgo": sesgo,
                "Riesgo": riesgo,
            }
        )
    return pd.DataFrame(rows).sort_values("Régimen")

# --- SIDEBAR ---
st.sidebar.title("🔍 Configuración")
ticker = st.sidebar.text_input("Ticker (Ej: AAPL, BTC-USD, KO)", value="KO").upper()
periodo = st.sidebar.selectbox("Periodo de datos", ["1y", "2y", "5y", "10y", "max"], index=2)
n_estados = st.sidebar.slider("Número de Estados (Regímenes)", 2, 5, 3)

st.sidebar.markdown("---")
st.sidebar.info("Este dashboard utiliza Modelos Ocultos de Markov (HMM) para detectar cambios en la estructura del mercado.")

# --- MAIN CONTENT ---
st.markdown(
    f"""
    <div class="hero-banner">
        <h1>Activo · {ticker}</h1>
        <p>Análisis de regímenes de mercado con Modelos Ocultos de Markov (HMM).</p>
    </div>
    """,
    unsafe_allow_html=True
)

df, data_error = get_data(ticker, periodo)

if df is not None:
    model, scaler, perfiles = train_markov(df, n_estados)
    diagnostico_df = build_state_diagnostics(df, perfiles, n_estados)
    
    # Predicción para mañana
    features = ['Returns', 'Range', 'Vol_Change']
    ultima_fila = scaler.transform(df[features].iloc[-1].values.reshape(1, -1))
    probs_manana = model.predict_proba(ultima_fila)[0]
    estado_hoy = df['Estado'].iloc[-1]
    estado_predicho = np.argmax(probs_manana)
    confianza = probs_manana[estado_predicho]
    
    # Probabilidades de transición
    trans_matrix = model.transmat_
    p_permanecer = trans_matrix[estado_hoy][estado_hoy]
    duracion_media = 1 / (1 - p_permanecer) if p_permanecer < 1 else 99
    top_transiciones = (
        pd.DataFrame(
            {
                "Régimen destino": [perfiles[i]["tag"] for i in range(n_estados)],
                "Probabilidad": trans_matrix[estado_hoy],
            }
        )
        .sort_values("Probabilidad", ascending=False)
        .reset_index(drop=True)
    )

    estado_hoy_row = diagnostico_df[diagnostico_df["Régimen"] == estado_hoy].iloc[0]
    estado_pred_row = diagnostico_df[diagnostico_df["Régimen"] == estado_predicho].iloc[0]

    # METRICAS SUPERIORES
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estado Actual", perfiles[estado_hoy]['tag'])
    with col2:
        st.metric("Predicción Mañana", perfiles[estado_predicho]['tag'], delta=f"{confianza:.2%}")
    with col3:
        st.metric("Persistencia", f"{duracion_media:.2f} días")
    with col4:
        st.metric("Confianza Modelo", f"{confianza:.2%}")

    st.write("### Lectura Rápida")
    st.markdown(
        f"""
        - **Hoy** el mercado está en **{perfiles[estado_hoy]['tag']}** (sesgo: **{estado_hoy_row['Sesgo']}**, riesgo: **{estado_hoy_row['Riesgo']}**).
        - **Mañana** el régimen más probable es **{perfiles[estado_predicho]['tag']}** con **{confianza:.2%}** de confianza.
        - Si el régimen actual se mantiene, su duración media estimada es de **{duracion_media:.2f} días**.
        """
    )

    st.write("### Señal Operativa")
    if estado_pred_row["Sesgo"] == "Alcista" and estado_pred_row["Riesgo"] == "Bajo" and confianza >= 0.55:
        st.success(
            f"Sesgo operativo: **Comprar / Aumentar exposición**. "
            f"Régimen probable {perfiles[estado_predicho]['tag']} con confianza {confianza:.2%}."
        )
    elif estado_pred_row["Sesgo"] == "Bajista" and confianza >= 0.55:
        st.warning(
            f"Sesgo operativo: **Reducir exposición / Cobertura**. "
            f"Régimen probable {perfiles[estado_predicho]['tag']} con confianza {confianza:.2%}."
        )
    else:
        st.info(
            f"Sesgo operativo: **Esperar / Neutral**. "
            f"No hay señal fuerte (confianza {confianza:.2%} o régimen mixto)."
        )

    # GRAFICO PRINCIPAL
    df_plot = df.copy()
    df_plot['Close_plot'] = pd.to_numeric(df_plot['Close'], errors='coerce').round(2)
    df_plot['Range_plot'] = pd.to_numeric(df_plot['Range'], errors='coerce').round(2)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'Precio de {ticker} por Régimen', 'Volatilidad (Range)'), 
                        row_width=[0.3, 0.7])

    # Colores por estado
    for i in range(n_estados):
        state_df = df_plot[df_plot['Estado'] == i].dropna(subset=['Close_plot'])
        fig.add_trace(go.Scatter(
            x=state_df.index, y=state_df['Close_plot'],
            mode='markers',
            name=perfiles[i]['tag'],
            marker=dict(
                color=perfiles[i]['color'],
                size=6,
                opacity=0.85,
                line=dict(width=0.4, color="rgba(20,33,61,0.35)")
            ),
            hovertemplate="%{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=1, col=1)

    # Línea de precio base
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot['Close_plot'],
            mode='lines',
            line=dict(color='rgba(20,33,61,0.55)', width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>",
            showlegend=False
        ),
        row=1,
        col=1
    )
    
    # Volatilidad en el segundo eje
    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot['Range_plot'],
            marker_color='rgba(0,109,119,0.45)',
            opacity=0.75,
            name="Volatilidad",
            hovertemplate="%{x|%Y-%m-%d}<br>Range: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        height=600, 
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#14213d'),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            title="Regímenes",
            bgcolor="rgba(255,255,255,0.65)",
            bordercolor="rgba(20,33,61,0.15)",
            borderwidth=1
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(20,33,61,0.08)")
    fig.update_yaxes(
        row=1,
        col=1,
        tickformat=",.2f",
        showgrid=True,
        gridcolor="rgba(20,33,61,0.08)"
    )
    fig.update_yaxes(
        row=2,
        col=1,
        tickformat=",.2f",
        showgrid=True,
        gridcolor="rgba(20,33,61,0.08)"
    )
    st.plotly_chart(fig, width="stretch")

    tabs = st.tabs(["Diagnóstico", "Probabilidades", "Matriz", "Modo Script"])

    with tabs[0]:
        st.write("### Diagnóstico por Régimen")
        view_diag = diagnostico_df.copy()
        for col in ["Retorno Prom (%)", "Volatilidad Prom (%)", "Frecuencia (%)"]:
            view_diag[col] = view_diag[col].map(lambda x: f"{x:.2f}%")
        st.dataframe(view_diag, width="stretch", hide_index=True)

        st.write("### Estado Actual vs Estado Probable")
        comp_df = pd.DataFrame(
            [
                {
                    "Tipo": "Actual",
                    "Régimen": perfiles[estado_hoy]["tag"],
                    "Sesgo": estado_hoy_row["Sesgo"],
                    "Riesgo": estado_hoy_row["Riesgo"],
                    "Retorno Prom": f"{estado_hoy_row['Retorno Prom (%)']:.2f}%",
                    "Volatilidad Prom": f"{estado_hoy_row['Volatilidad Prom (%)']:.2f}%",
                },
                {
                    "Tipo": "Probable (mañana)",
                    "Régimen": perfiles[estado_predicho]["tag"],
                    "Sesgo": estado_pred_row["Sesgo"],
                    "Riesgo": estado_pred_row["Riesgo"],
                    "Retorno Prom": f"{estado_pred_row['Retorno Prom (%)']:.2f}%",
                    "Volatilidad Prom": f"{estado_pred_row['Volatilidad Prom (%)']:.2f}%",
                },
            ]
        )
        st.dataframe(comp_df, width="stretch", hide_index=True)

    with tabs[1]:
        st.write("### 🔮 Probabilidades para Mañana")
        prob_df = pd.DataFrame(
            {
                "Régimen": [perfiles[i]["tag"] for i in range(n_estados)],
                "Probabilidad": probs_manana,
            }
        ).sort_values("Probabilidad", ascending=False)
        st.bar_chart(prob_df.set_index("Régimen"))

        st.write("### Desde el Estado de Hoy")
        top_view = top_transiciones.copy()
        top_view["Probabilidad"] = top_view["Probabilidad"].map(lambda x: f"{x:.2%}")
        st.dataframe(top_view, width="stretch", hide_index=True)

    with tabs[2]:
        st.write("### 🔄 Matriz de Transición")
        matrix_df = pd.DataFrame(
            trans_matrix,
            index=[perfiles[i]['tag'] for i in range(n_estados)],
            columns=[perfiles[i]['tag'] for i in range(n_estados)]
        )
        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix_df.values,
                x=matrix_df.columns,
                y=matrix_df.index,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                text=[[f"{v:.2%}" for v in row] for row in matrix_df.values],
                texttemplate="%{text}",
            )
        )
        heatmap.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Estado mañana",
            yaxis_title="Estado hoy",
            height=430,
        )
        st.plotly_chart(heatmap, width="stretch")

        styled_matrix = matrix_df.style.format("{:.2%}")
        if HAS_MATPLOTLIB:
            try:
                styled_matrix = styled_matrix.background_gradient(cmap='Blues')
            except Exception:
                pass
        st.dataframe(styled_matrix, width="stretch")

    with tabs[3]:
        st.write("### Resumen Textual (estilo script)")
        resumen = [
            f"ANÁLISIS DE MARKOV PARA: {ticker}",
            f"Estado actual: {perfiles[estado_hoy]['tag']}",
            f"Predicción para mañana: {perfiles[estado_predicho]['tag']}",
            f"Confianza del modelo: {confianza:.2%}",
            f"Persistencia estimada del estado actual: {duracion_media:.2f} días",
            "",
            "DIAGNÓSTICO DE RÉGIMENES:",
        ]
        for _, row in diagnostico_df.iterrows():
            resumen.append(
                f"- {row['Etiqueta']}: retorno {row['Retorno Prom (%)']:.2f}%, "
                f"volatilidad {row['Volatilidad Prom (%)']:.2f}%, frecuencia {row['Frecuencia (%)']:.2f}% "
                f"(sesgo {row['Sesgo']}, riesgo {row['Riesgo']})"
            )
        st.code("\n".join(resumen), language="text")

else:
    if data_error and ("YFRateLimitError" in data_error or "Too Many Requests" in data_error):
        st.warning(
            "Yahoo Finance limitó temporalmente las consultas (rate limit). "
            "Esperá 1-2 minutos y volvé a intentar."
        )
    else:
        st.error(f"No se pudieron obtener datos para el ticker {ticker}. Verifica que sea correcto.")
