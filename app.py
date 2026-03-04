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
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }
    .sidebar-brand {
        color: var(--brand);
        font-size: 1.25rem;
        font-weight: 800;
        margin: 0.2rem 0 0.3rem;
    }
    .sidebar-config-title {
        color: #000000;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0.1rem 0 0.65rem;
    }
    .sidebar-field-label {
        color: var(--text-muted);
        font-weight: 600;
        margin: 0.35rem 0 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-help-dot {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1rem;
        height: 1rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        color: var(--brand);
        font-size: 0.72rem;
        font-weight: 700;
        cursor: help;
        background: #ffffff;
    }
    .help-dot-small {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 0.86rem;
        height: 0.86rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        color: var(--brand);
        font-size: 0.64rem;
        font-weight: 700;
        cursor: help;
        background: #ffffff;
        vertical-align: middle;
        margin-left: 0.25rem;
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

    /* Tabs: contraste alto en seleccionada y no seleccionadas */
    .stTabs [role="tablist"] {
        gap: 10px;
    }
    .stTabs [role="tab"] {
        color: #334155 !important;
        background: rgba(255,255,255,0.85) !important;
        border: 1px solid rgba(15,23,42,0.15) !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 8px 14px !important;
        font-weight: 600 !important;
    }
    .stTabs [role="tab"]:hover {
        color: #0f172a !important;
        background: rgba(255,255,255,0.98) !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #0f172a !important;
        background: #ffffff !important;
        border-bottom: 2px solid var(--brand) !important;
    }

    /* Tabla HTML clara (independiente del tema Streamlit) */
    .light-table-wrap {
        overflow-x: auto;
        background: #ffffff;
        border: 1px solid #d7e3f0;
        border-radius: 10px;
    }
    .light-table {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        color: #0f172a;
        font-size: 0.98rem;
    }
    .light-table thead th {
        background: #f8fafc;
        color: #0f172a;
        font-weight: 700;
        text-align: left;
        border: 1px solid #dbe3ee;
        padding: 10px 12px;
        white-space: nowrap;
    }
    .light-table tbody td {
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #e5e7eb;
        padding: 10px 12px;
        white-space: nowrap;
    }
    .light-code {
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #d7e3f0;
        border-radius: 10px;
        padding: 12px 14px;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.92rem;
        line-height: 1.45;
        white-space: pre-wrap;
    }

    /* Tablas en modo claro */
    div[data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid rgba(15,23,42,0.15) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stDataFrame"] table {
        background: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-testid="stDataFrame"] thead tr th {
        background: #f8fafc !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDataFrame"] tbody tr td {
        color: #0f172a !important;
    }

    /* Alertas legibles sobre fondos claros (warning/info/success/error) */
    div[data-testid="stAlert"] {
        color: var(--text-main) !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: var(--text-main) !important;
    }
    div[data-testid="stAlert"] svg {
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
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

def classify_regime(ret, vol, mean_range):
    if ret > 0.0005 and vol < mean_range * 0.9:
        return "Alcista Saludable", "#4ade80"
    if ret < -0.0005 and vol > mean_range * 1.1:
        return "Panico / Crash", "#f87171"
    if vol > mean_range * 1.2:
        return "Extrema Volatilidad", "#fbbf24"
    if ret < 0:
        return "Bajista / Desangrado", "#fb923c"
    return "Lateral / Acumulacion", "#60a5fa"

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
        tag, color = classify_regime(ret, vol, df['Range'].mean())

        perfiles[i] = {"tag": f"R{i}: {tag}", "color": color, "ret": ret, "vol": vol}
        
    return model, scaler, perfiles

def hmm_num_params(n_states, n_features):
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features
    covars = n_states * (n_features * (n_features + 1) // 2)
    return startprob + transmat + means + covars

def select_best_n_states(df, candidates=(2, 3, 4, 5), criterion="BIC"):
    features = ['Returns', 'Range', 'Vol_Change']
    X = StandardScaler().fit_transform(df[features])
    n_obs = len(X)
    n_feat = X.shape[1]
    rows = []

    for n in candidates:
        try:
            model = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=600,
                random_state=42
            )
            model.fit(X)
            ll = model.score(X)
            k = hmm_num_params(n, n_feat)
            aic = -2 * ll + 2 * k
            bic = -2 * ll + np.log(max(n_obs, 2)) * k
            rows.append({"Estados": n, "LogL": ll, "AIC": aic, "BIC": bic})
        except Exception:
            continue

    if not rows:
        return None, None

    df_sel = pd.DataFrame(rows).sort_values("Estados").reset_index(drop=True)
    metric = "BIC" if criterion.upper() == "BIC" else "AIC"
    best_n = int(df_sel.sort_values(metric).iloc[0]["Estados"])
    return best_n, df_sel

def evaluate_walk_forward(df, n_states=3, test_ratio=0.2, min_train_ratio=0.6):
    features = ['Returns', 'Range', 'Vol_Change']
    n_obs = len(df)
    block_size = max(20, int(n_obs * test_ratio / 3))
    min_train_size = max(n_states * 10, int(n_obs * min_train_ratio))

    if min_train_size + block_size >= n_obs:
        return None, "Datos insuficientes para validación robusta (walk-forward)."

    ll_values = []
    conf_values = []
    last_state_freq = None
    last_state_labels = None
    last_test_start = None
    last_test_end = None

    for train_end in range(min_train_size, n_obs - block_size + 1, block_size):
        test_end = min(train_end + block_size, n_obs)
        train_slice = df.iloc[:train_end]
        test_slice = df.iloc[train_end:test_end]
        if len(test_slice) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_slice[features])
        X_test = scaler.transform(test_slice[features])

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=300,
            random_state=42
        )
        model.fit(X_train)

        ll_values.append(model.score(X_test) / len(X_test))
        posterior_test = model.predict_proba(X_test)
        conf_values.append(posterior_test.max(axis=1).mean())

        states_test = model.predict(X_test)
        last_state_freq = (
            pd.Series(states_test)
            .value_counts(normalize=True)
            .reindex(range(n_states), fill_value=0.0)
        )
        mean_range_ref = train_slice['Range'].mean()
        labels = []
        for i in range(n_states):
            state_subset = test_slice.iloc[np.where(states_test == i)[0]]
            if state_subset.empty:
                s_ret = 0.0
                s_vol = mean_range_ref
            else:
                s_ret = state_subset['Returns'].mean()
                s_vol = state_subset['Range'].mean()
            s_tag, _ = classify_regime(s_ret, s_vol, mean_range_ref)
            labels.append(f"WF{i}: {s_tag}")
        last_state_labels = labels
        last_test_start = train_end
        last_test_end = test_end

    if not ll_values or last_state_freq is None or last_state_labels is None or last_test_start is None or last_test_end is None:
        return None, "No se pudieron generar ventanas de validación walk-forward."

    return {
        "windows": len(ll_values),
        "mean_ll": float(np.mean(ll_values)),
        "worst_ll": float(np.min(ll_values)),
        "mean_conf": float(np.mean(conf_values)),
        "last_test_start": int(last_test_start),
        "last_test_end": int(last_test_end),
        "state_freq_last": last_state_freq,
        "state_labels_last": last_state_labels,
    }, None

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


def show_light_dataframe(df, hide_index=True):
    html = df.to_html(index=not hide_index, classes="light-table", border=0, escape=False)
    st.markdown(f'<div class="light-table-wrap">{html}</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown('<div class="sidebar-brand">Markov Signal</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-config-title">Configuración</div>', unsafe_allow_html=True)
ticker = st.sidebar.text_input("Ticker (Ej: AAPL, BTC-USD, KO)", value="KO").upper()
periodo = st.sidebar.selectbox(
    "Periodo de datos",
    ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    index=6,
)
st.sidebar.markdown(
    "<div class=\"sidebar-field-label\">Seleccion de estados <span class=\"sidebar-help-dot\" title=\"Manual: vos elegis 2-5. Auto: prueba 2,3,4,5 y elige por AIC/BIC.\">i</span></div>",
    unsafe_allow_html=True,
)
state_selection_mode = st.sidebar.selectbox(
    "Seleccion de estados",
    ["Manual", "Auto (AIC/BIC)"],
    index=1,
    label_visibility="collapsed",
)
if state_selection_mode == "Manual":
    n_estados = st.sidebar.slider("Numero de estados (regimenes)", 2, 5, 3)
    auto_criterion = "BIC"
else:
    st.sidebar.markdown(
        "<div class=\"sidebar-field-label\">Criterio automatico (AIC/BIC) <span class=\"sidebar-help-dot\" title=\"AIC y BIC comparan modelos penalizando complejidad. Menor valor = mejor equilibrio entre ajuste y simplicidad. BIC penaliza mas y suele elegir modelos mas simples.\">i</span></div>",
        unsafe_allow_html=True,
    )
    auto_criterion = st.sidebar.selectbox("Criterio automatico", ["BIC", "AIC"], index=0)
    n_estados = 3
st.sidebar.markdown(
    '<div class="sidebar-field-label">Tipo de validación <span class="sidebar-help-dot" title="Rápida: una sola partición train/test.&#10;Robusta: varias pruebas en el tiempo (walk-forward), más exigente y realista.">i</span></div>',
    unsafe_allow_html=True,
)
validation_mode = st.sidebar.selectbox(
    "Tipo de validación",
    ["Rápida (train/test)", "Robusta (walk-forward)"],
    index=1,
    label_visibility="collapsed",
)
st.sidebar.markdown(
    '<div class="sidebar-field-label">Tamaño de prueba (OOS) <span class="sidebar-help-dot" title="Define cuánto de los datos se usa para probar el modelo en datos no vistos. Más alto = prueba más dura.">i</span></div>',
    unsafe_allow_html=True,
)
test_ratio = st.sidebar.slider("Porción de test (OOS)", 0.10, 0.40, 0.20, 0.05, label_visibility="collapsed")
st.sidebar.markdown(
    '<div class="sidebar-field-label">Predicción de mañana <span class="sidebar-help-dot" title="Hard: usa solo el estado más probable de hoy y toma su fila de transición.&#10;Soft: usa la distribución completa de estados de hoy y la proyecta con la matriz de transición.">i</span></div>',
    unsafe_allow_html=True,
)
modo_prediccion = st.sidebar.selectbox(
    "Predicción de mañana",
    ["Transición desde estado actual (hard)", "Transición desde distribución posterior (soft)"],
    index=1,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Este dashboard utiliza Modelos Ocultos de Markov (HMM) para detectar cambios en la estructura del mercado.\n\n"
    "Cuándo usar cada validación:\n"
    "- Rápida (train/test): para revisar ideas rápido.\n"
    "- Robusta (walk-forward): para uso real, porque prueba varias veces en el tiempo y es más exigente."
)

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
    features = ['Returns', 'Range', 'Vol_Change']
    estado_freq_test = pd.Series([0.0] * n_estados, index=range(n_estados))
    eval_freq_title = "Frecuencia de Estados en Test (OOS)"
    eval_state_labels = [f"R{i}" for i in range(n_estados)]
    eval_freq_help = (
        "Que mirar: esta tabla muestra cuanto aparece cada estado en la validacion. "
        "Importante: en walk-forward los IDs (WF0/WF1/...) son internos de ese bloque y "
        "no equivalen 1:1 con R0/R1/... del modelo actual."
    )
    model_selection_df = None

    if state_selection_mode == "Auto (AIC/BIC)":
        best_n, model_selection_df = select_best_n_states(
            df,
            candidates=(2, 3, 4, 5),
            criterion=auto_criterion,
        )
        if best_n is not None:
            n_estados = best_n
            estado_freq_test = pd.Series([0.0] * n_estados, index=range(n_estados))
            eval_state_labels = [f"R{i}" for i in range(n_estados)]
        else:
            st.warning("No se pudo seleccionar estados en automático. Se usa 3 por defecto.")

    if validation_mode == "Rápida (train/test)":
        split_idx = int(len(df) * (1 - test_ratio))
        split_idx = max(split_idx, n_estados * 8)
        split_idx = min(split_idx, len(df) - max(30, n_estados * 4))
        if split_idx <= n_estados * 6 or split_idx >= len(df):
            st.error("No hay suficientes datos para separar train/test de forma robusta. Probá un periodo mayor o menos estados.")
            st.stop()

        df_train_fit = df.iloc[:split_idx].copy()
        df_test_oos = df.iloc[split_idx:].copy()
        model, scaler, perfiles = train_markov(df_train_fit, n_estados)

        # Aplicamos el modelo de train sobre toda la serie para ver estados y señal actual.
        X_all = scaler.transform(df[features].values)
        df['Estado'] = model.predict(X_all)
        posterior_all = model.predict_proba(X_all)
        diagnostico_df = build_state_diagnostics(df, perfiles, n_estados)

        X_train = scaler.transform(df_train_fit[features].values)
        X_test = scaler.transform(df_test_oos[features].values)
        train_ll = model.score(X_train) / len(X_train)
        test_ll = model.score(X_test) / len(X_test)
        estado_freq_test = (
            df.iloc[split_idx:]['Estado']
            .value_counts(normalize=True)
            .reindex(range(n_estados), fill_value=0.0)
        )

        valid_count_label = "Datos Train/Test"
        valid_count_value = f"{len(df_train_fit)} / {len(df_test_oos)}"
        valid_ll_value = test_ll
        valid_ll_delta = f"train {train_ll:.4f}"
        valid_conf_value = posterior_all[split_idx:, :].max(axis=1).mean()
        valid_caption = (
            f"Modo rápido: train hasta {df.index[split_idx - 1].date()} | "
            f"test desde {df.index[split_idx].date()}."
        )
        eval_state_labels = [perfiles[i]["tag"] for i in range(n_estados)]
    else:
        wf_stats, wf_error = evaluate_walk_forward(df, n_states=n_estados, test_ratio=test_ratio)
        if wf_error:
            st.error(wf_error)
            st.stop()

        # Modelo operativo final entrenado con toda la historia disponible.
        model, scaler, perfiles = train_markov(df, n_estados)
        X_all = scaler.transform(df[features].values)
        posterior_all = model.predict_proba(X_all)
        diagnostico_df = build_state_diagnostics(df, perfiles, n_estados)

        estado_freq_test = wf_stats["state_freq_last"]
        eval_freq_title = "Frecuencia en el último bloque walk-forward"
        eval_state_labels = wf_stats["state_labels_last"]
        valid_count_label = "Bloques evaluados"
        valid_count_value = str(wf_stats["windows"])
        valid_ll_value = wf_stats["mean_ll"]
        valid_ll_delta = f"peor {wf_stats['worst_ll']:.4f}"
        valid_conf_value = wf_stats["mean_conf"]
        valid_caption = (
            f"Modo robusto: {wf_stats['windows']} bloques. "
            f"Último bloque validado: {df.index[wf_stats['last_test_start']].date()} "
            f"a {df.index[wf_stats['last_test_end'] - 1].date()}."
        )

    # Predicción para mañana
    posterior_hoy = posterior_all[-1]
    estado_hoy = int(df['Estado'].iloc[-1])
    trans_matrix = model.transmat_
    if modo_prediccion == "Transición desde estado actual (hard)":
        probs_manana = trans_matrix[estado_hoy]
        metodo_prediccion_label = "Hard (solo el estado más probable de hoy)"
    else:
        probs_manana = posterior_hoy @ trans_matrix
        metodo_prediccion_label = "Soft (mezcla de todos los estados de hoy)"
    probs_manana = np.asarray(probs_manana)
    estado_predicho = np.argmax(probs_manana)
    confianza = probs_manana[estado_predicho]

    # Probabilidades de transición
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

    st.write("### Validación (datos no vistos)")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.metric(valid_count_label, valid_count_value)
    with v2:
        st.metric("Calidad promedio", f"{valid_ll_value:.4f}", delta=valid_ll_delta)
    with v3:
        st.metric("Confianza media", f"{valid_conf_value:.2%}")
    st.caption(valid_caption)
    if state_selection_mode == "Auto (AIC/BIC)":
        st.caption(f"Estados elegidos en auto: {n_estados} (criterio {auto_criterion}).")
    else:
        st.caption(f"Estados elegidos en manual: {n_estados}.")

    st.write("### Lectura Rápida")
    st.markdown(
        f"""
        - **Hoy** el mercado está en **{perfiles[estado_hoy]['tag']}** (sesgo: **{estado_hoy_row['Sesgo']}**, riesgo: **{estado_hoy_row['Riesgo']}**).
        - **Mañana** el régimen más probable es **{perfiles[estado_predicho]['tag']}** con **{confianza:.2%}** de confianza.
        - **Método de predicción**: {metodo_prediccion_label}.
        - Si el régimen actual se mantiene, su duración media estimada es de **{duracion_media:.2f} días**.
        """
    )
    st.write("### Señal Operativa")
    if estado_pred_row["Sesgo"] == "Alcista" and estado_pred_row["Riesgo"] == "Bajo" and confianza >= 0.55:
        accion_base = "Comprar / Aumentar exposición"
        nivel_base = "fuerte"
    elif estado_pred_row["Sesgo"] == "Bajista" and confianza >= 0.55:
        accion_base = "Reducir exposición / Cobertura"
        nivel_base = "fuerte"
    else:
        accion_base = "Esperar / Neutral"
        nivel_base = "debil"

    if valid_conf_value >= 0.58:
        filtro_robustez = "alto"
    elif valid_conf_value >= 0.50:
        filtro_robustez = "medio"
    else:
        filtro_robustez = "bajo"

    if nivel_base == "debil":
        accion_final = "Esperar / Neutral"
        resumen_final = "La señal base no es suficientemente clara."
    elif filtro_robustez == "bajo":
        accion_final = "Cautela / Tama?o peque?o"
        resumen_final = "La validaci?n en datos no vistos est? floja."
    else:
        accion_final = accion_base
        resumen_final = "La señal base est? acompa?ada por validaci?n aceptable."

    st.markdown(
        f"- **Señal base (R)**: {accion_base} (confianza de mañana: **{confianza:.2%}**).\n"
        f"- **Filtro de robustez ({validation_mode})**: nivel **{filtro_robustez}** "
        f"(confianza media OOS: **{valid_conf_value:.2%}**).\n"
        f"- **Conclusión final**: **{accion_final}**. {resumen_final}"
    )

    if accion_final == "Comprar / Aumentar exposición":
        st.success(f"Conclusión final: **{accion_final}**")
    elif accion_final == "Reducir exposición / Cobertura":
        st.warning(f"Conclusión final: **{accion_final}**")
    elif accion_final == "Cautela / Tama?o peque?o":
        st.warning(f"Conclusión final: **{accion_final}**")
    else:
        st.info(f"Conclusión final: **{accion_final}**")

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
        paper_bgcolor='rgba(255,255,255,0.96)', 
        plot_bgcolor='rgba(255,255,255,0.98)',
        font=dict(color='#0f172a', size=15),
        title=dict(text=""),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            title="Regímenes",
            font=dict(color='#0f172a', size=16),
            title_font=dict(color='#0f172a', size=17),
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="rgba(15,23,42,0.35)",
            borderwidth=1.2
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.16)",
        tickfont=dict(color='#0f172a', size=13),
        title_font=dict(color='#0f172a', size=14)
    )
    fig.update_yaxes(
        row=1,
        col=1,
        tickformat=",.2f",
        showgrid=True,
        gridcolor="rgba(15,23,42,0.16)",
        tickfont=dict(color='#0f172a', size=13),
        title_font=dict(color='#0f172a', size=14)
    )
    fig.update_yaxes(
        row=2,
        col=1,
        tickformat=",.2f",
        showgrid=True,
        gridcolor="rgba(15,23,42,0.16)",
        tickfont=dict(color='#0f172a', size=13),
        title_font=dict(color='#0f172a', size=14)
    )
    st.plotly_chart(fig, width="stretch")

    tabs = st.tabs(["Diagnóstico", "Probabilidades", "Matriz", "Modo Script"])

    with tabs[0]:
        if model_selection_df is not None:
            st.write("### Seleccion Automatica de Estados")
            select_view = model_selection_df.copy()
            select_view["LogL"] = select_view["LogL"].map(lambda x: f"{x:.2f}")
            select_view["AIC"] = select_view["AIC"].map(lambda x: f"{x:.2f}")
            select_view["BIC"] = select_view["BIC"].map(lambda x: f"{x:.2f}")
            show_light_dataframe(select_view, hide_index=True)
        st.write("### Diagnostico por Regimen")
        view_diag = diagnostico_df.copy()
        for col in ["Retorno Prom (%)", "Volatilidad Prom (%)", "Frecuencia (%)"]:
            view_diag[col] = view_diag[col].map(lambda x: f"{x:.2f}%")
        show_light_dataframe(view_diag, hide_index=True)

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
        show_light_dataframe(comp_df, hide_index=True)

        st.markdown(
            f'### {eval_freq_title} <span class="help-dot-small" title="{eval_freq_help}">i</span>',
            unsafe_allow_html=True,
        )
        freq_test_df = pd.DataFrame(
            {
                "Estado": eval_state_labels,
                "Frecuencia OOS": estado_freq_test.values,
            }
        )
        freq_test_df["Frecuencia OOS"] = freq_test_df["Frecuencia OOS"].map(lambda x: f"{x:.2%}")
        show_light_dataframe(freq_test_df, hide_index=True)
        st.caption("R*: estados del modelo operativo actual. WF*: estados del bloque de validacion walk-forward.")

    with tabs[1]:
        st.write("### 🔮 Probabilidades para Mañana")
        st.caption(f"Método activo: {metodo_prediccion_label}")
        prob_df = pd.DataFrame(
            {
                "Régimen": [perfiles[i]["tag"] for i in range(n_estados)],
                "Probabilidad": probs_manana,
            }
        ).sort_values("Probabilidad", ascending=False)
        prob_fig = go.Figure(
            data=go.Bar(
                x=prob_df["Probabilidad"],
                y=prob_df["Régimen"],
                orientation="h",
                marker_color="rgba(0,109,119,0.75)",
                text=[f"{v:.2%}" for v in prob_df["Probabilidad"]],
                textposition="outside",
            )
        )
        prob_fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickformat=".2%", gridcolor="rgba(15,23,42,0.12)", range=[0, 1]),
            yaxis=dict(tickfont=dict(color="#0f172a")),
            font=dict(color="#0f172a"),
            height=360,
        )
        st.plotly_chart(prob_fig, width="stretch")

        st.write("### Desde el Estado de Hoy")
        top_view = top_transiciones.copy()
        top_view["Probabilidad"] = top_view["Probabilidad"].map(lambda x: f"{x:.2%}")
        show_light_dataframe(top_view, hide_index=True)

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
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(heatmap, width="stretch")

        matrix_view = matrix_df.copy()
        for c in matrix_view.columns:
            matrix_view[c] = matrix_view[c].map(lambda x: f"{x:.2%}")
        show_light_dataframe(matrix_view, hide_index=False)

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
        st.markdown(
            f'<div class="light-code">{"\n".join(resumen)}</div>',
            unsafe_allow_html=True
        )

else:
    if data_error and ("YFRateLimitError" in data_error or "Too Many Requests" in data_error):
        st.warning(
            "Yahoo Finance limitó temporalmente las consultas (rate limit). "
            "Esperá 1-2 minutos y volvé a intentar."
        )
    else:
        st.error(f"No se pudieron obtener datos para el ticker {ticker}. Verifica que sea correcto.")
