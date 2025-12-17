#Pipeline presentado para el Pyday La Paz 2025
#Autora: Lucia Martinez Zuzunaga
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de Tiempo (24 horas, cada 10 minutos -> 144 puntos)
timestamps = pd.date_range(start="2024-07-27 00:00:00", end="2024-07-27 23:50:00", freq="10min")
n = len(timestamps)

# 2. Generación de Patrones Normales (CORREGIDO)
# Variable t va de 0 a 2pi (un ciclo completo)
t = np.linspace(0, 2 * np.pi, n)

# Temperatura: Usamos -cos(t). Empieza baja, sube al mediodía, baja al final.
# Rango aprox: 18°C noche a 34°C día
temp_base = 26 - 8 * np.cos(t) + np.random.normal(0, 0.5, n)

# Humedad (RH): Sigue al cos(t). Empieza alta (noche), baja al mediodía.
# Rango aprox: 85% noche a 45% día
rh_base = 65 + 20 * np.cos(t) + np.random.normal(0, 1, n)
rh_base = np.clip(rh_base, 0, 100)

# Humedad Suelo: Decrecimiento lineal suave
soil_base = 20 - 0.5 * np.linspace(0, 1, n) + np.random.normal(0, 0.1, n)

# Flujo Savia: Solo positivo durante el día (pico al mediodía)
sap_curve = np.maximum(0, -15 * np.cos(t) - 5) 
sap_base = sap_curve * 0.5 + np.random.normal(0, 0.05, n)
sap_base = np.maximum(0, sap_base)

# Albedo: ~0.15 de día, 0 de noche
is_day = sap_base > 0.1
albedo_base = np.where(is_day, 0.15 + np.random.normal(0, 0.01, n), 0.0)

# 3. Crear DataFrame
df_sintetico = pd.DataFrame({
    'Timestamp': timestamps,
    'temperature_C': temp_base,
    'RH_percent': rh_base,
    'soil_moisture_percent': soil_base,
    'sap_flow_gph': sap_base,
    'albedo': albedo_base,
    'anomaly_labels': 1 # 1 = Normal
})

# 4. Inyectar Anomalías (AHORA SÍ COHERENTES)

# --- Anomalía 1: INCENDIO a las 14:00 (Índices 84-88) ---
# A las 14:00 la temperatura base ya es alta (~33°C).
# Sumamos +15°C -> Llegará a ~48°C (Drástico)
# Restamos -20% Humedad -> Bajará a ~25% (Muy seco)
indices_incendio = range(84, 88)
df_sintetico.loc[indices_incendio, 'temperature_C'] += 15 
df_sintetico.loc[indices_incendio, 'RH_percent'] -= 20
df_sintetico.loc[indices_incendio, 'albedo'] = 0.60 # Ceniza
df_sintetico.loc[indices_incendio, 'anomaly_labels'] = -1 

# --- Anomalía 2: FALLO SENSOR a las 04:00 (Índices 24-26) ---
indices_fallo = range(24, 26)
df_sintetico.loc[indices_fallo, 'soil_moisture_percent'] = 0.0
df_sintetico.loc[indices_fallo, 'temperature_C'] = -50.0
df_sintetico.loc[indices_fallo, 'anomaly_labels'] = -1

# 5. Guardar y Verificar
nombre_archivo = 'datos_hidroclimaticos_24h.csv'
df_sintetico.to_csv(nombre_archivo, index=False)

print("--- Verificación de Datos a las 14:00 (Incendio) ---")
# Mostramos índices alrededor de la anomalía para ver el salto
print(df_sintetico.iloc[82:87][['Timestamp', 'temperature_C', 'RH_percent', 'anomaly_labels']])

print("\n--- Verificación de Medianoche (Inicio) ---")
print(df_sintetico.iloc[0:3][['Timestamp', 'temperature_C', 'RH_percent']])

# ==========================================
# PASO 2: CARGA DE DATOS (Ya generado previamente)
# ==========================================
try:
    df = pd.read_csv('datos_hidroclimaticos_24h.csv')
except FileNotFoundError:
    print("¡Error! No se encuentra 'datos_hidroclimaticos_24h.csv'. Ejecuta el paso 1 primero.")
    # (Si necesitas el código de generación de nuevo, avísame)

# Definimos las columnas de los sensores (Features) y la etiqueta real (Ground Truth)
features = ['temperature_C', 'RH_percent', 'soil_moisture_percent', 'sap_flow_gph', 'albedo']
ground_truth = df['anomaly_labels'] # 1: Normal, -1: Anomalía

# ==========================================
# PASO 3: PREPROCESAMIENTO
# ==========================================
# Limpieza básica
X = df[features].dropna()
# Actualizamos ground_truth por si se borraron filas
y_true = df.loc[X.index, 'anomaly_labels']

# Normalización (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# PASO 4: APLICAR PCA
# ==========================================
# Reducimos a 2 componentes como en el paper
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Creamos un DataFrame para facilitar el manejo posterior
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

# ==========================================
# PASO 5: IMPLEMENTAR ISOLATION FOREST
# ==========================================
# Configuración basada en el paper: n_estimators=200, contamination ~0.2 (ajustable)
iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso_forest.fit(X_pca)
y_pred_iso = iso_forest.predict(X_pca)

# ==========================================
# PASO 6: EVALUACIÓN DEL MODELO
# ==========================================
def calcular_metricas(y_real, y_pred, nombre_modelo):
    # pos_label=-1 porque nos interesa detectar la clase Anomalía (-1)
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, pos_label=-1, zero_division=0)
    rec = recall_score(y_real, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_real, y_pred, pos_label=-1, zero_division=0)
    
    return {
        "Modelo": nombre_modelo,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "Detection Rate": round(rec, 2) # En este contexto es igual al Recall
    }

resultados = []
resultados.append(calcular_metricas(y_true, y_pred_iso, "Isolation Forest"))

# ==========================================
# PASO 7: VISUALIZACIÓN DE DATOS
# ==========================================
plt.figure(figsize=(10, 6))
# Puntos normales predichos
plt.scatter(df_pca.loc[y_pred_iso == 1, 'PC1'], df_pca.loc[y_pred_iso == 1, 'PC2'], 
            c='blue', label='Normal', alpha=0.6, s=20)
# Anomalías predichas
plt.scatter(df_pca.loc[y_pred_iso == -1, 'PC1'], df_pca.loc[y_pred_iso == -1, 'PC2'], 
            c='red', label='Anomalía Detectada', edgecolors='k', s=50)

plt.title('Detección de Anomalías con Isolation Forest (PCA 2D)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ==========================================
# PASO 8: IMPLEMENTACIÓN Y EVALUACIÓN DE OTROS MODELOS
# ==========================================

# --- A. One-Class SVM ---
oc_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
y_pred_svm = oc_svm.fit_predict(X_pca)
resultados.append(calcular_metricas(y_true, y_pred_svm, "One-Class SVM"))

# --- B. Local Outlier Factor (LOF) ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_lof = lof.fit_predict(X_pca)
resultados.append(calcular_metricas(y_true, y_pred_lof, "Local Outlier Factor"))

# --- C. KMeans Distance ---
# Lógica: Los puntos más lejanos al centro de su cluster son anomalías
kmeans = KMeans(n_clusters=1, random_state=42, n_init=10) # Asumimos 1 cluster "normal" principal
kmeans.fit(X_pca)
# Calcular distancias al centroide
distancias = kmeans.transform(X_pca).max(axis=1) # Distancia al centroide más cercano
# Definir umbral (ej. el top 5% más lejano es anomalía)
umbral = np.percentile(distancias, 95) 
y_pred_kmeans = np.where(distancias > umbral, -1, 1)
resultados.append(calcular_metricas(y_true, y_pred_kmeans, "KMeans Distance"))

# ==========================================
# MOSTRAR TABLA FINAL COMPARATIVA
# ==========================================
df_resultados = pd.DataFrame(resultados)
print("\n=== TABLA COMPARATIVA DE MODELOS ===")
print(df_resultados.to_string(index=False))
