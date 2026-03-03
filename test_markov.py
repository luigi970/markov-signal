import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def ejecutar_modelo_completo(ticker):
    # 1. Obtención de datos
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    
    # 2. Ingeniería de Variables
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['Vol_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)

    # 3. Normalización
    features = ['Returns', 'Range', 'Vol_Change']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # 4. Entrenamiento del Modelo de Markov
    n_estados = 3
    model = GaussianHMM(n_components=n_estados, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    df['Estado_Actual'] = model.predict(X)

    # 5. PREDICCIÓN PARA MAÑANA
    ultima_fila_scaled = scaler.transform(df[features].iloc[-1].values.reshape(1, -1))
    probs_manana = model.predict_proba(ultima_fila_scaled)[0]

    print(f"\n" + "="*40)
    print(f"ANÁLISIS DE MARKOV PARA: {ticker}")
    print("="*40)

    # 6. PERFIL DE CADA RÉGIMEN (Lo nuevo)
    print("\nDIAGNÓSTICO DE LOS ESTADOS DETECTADOS:")
    for i in range(n_estados):
        datos_estado = df[df['Estado_Actual'] == i]
        ret_promedio = datos_estado['Returns'].mean() * 100 # En porcentaje
        vol_promedio = datos_estado['Range'].mean() * 100
        frecuencia = len(datos_estado) / len(df) * 100
        
        # Etiqueta automática simple
        tipo = "ALCISTA" if ret_promedio > 0.01 else "BAJISTA" if ret_promedio < -0.01 else "LATERAL"
        riesgo = "ALTO" if vol_promedio > df['Range'].mean()*100 else "BAJO"

        print(f"Régimen {i} ({tipo}):")
        print(f"   - Retorno diario prom: {ret_promedio:.4f}%")
        print(f"   - Volatilidad (Riesgo): {riesgo} ({vol_promedio:.2f}%)")
        print(f"   - Ocurre el {frecuencia:.1f}% del tiempo")

    print("\n" + "-"*40)
    # 7. RESULTADO FINAL
    estado_predicho = np.argmax(probs_manana)
    print(f"PREDICCIÓN ACTUAL: Régimen {estado_predicho}")
    print(f"CONFIANZA DEL MODELO: {probs_manana[estado_predicho]:.2%}")
    print("-"*40)

    # 8. MATRIZ DE TRANSICIÓN (Probabilidades Futuras)
    print("\n" + "="*40)
    print("MATRIZ DE PROBABILIDADES (¿Qué pasará mañana?)")
    print("="*40)
    
    trans_matrix = model.transmat_
    for i in range(n_estados):
        print(f"\nSi hoy estamos en Régimen {i}:")
        for j in range(n_estados):
            prob = trans_matrix[i][j] * 100
            print(f"   -> Probabilidad de ir al Régimen {j} mañana: {prob:.2f}%")

    # 9. ESTIMACIÓN DE DURACIÓN
    # La teoría de Markov permite saber cuánto tiempo suele durar un estado
    # Duración esperada = 1 / (1 - P(i,i))
    print("\n" + "-"*40)
    print("PERSISTENCIA DEL ESTADO:")
    for i in range(n_estados):
        p_permanecer = trans_matrix[i][i]
        duracion_esperada = 1 / (1 - p_permanecer) if p_permanecer < 1 else float('inf')
        print(f"   - El Régimen {i} suele durar: {duracion_esperada:.1f} días")

# Prueba con KO
ejecutar_modelo_completo("KO")