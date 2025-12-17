import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# --- 1. GENERAR DATOS DE PRUEBA (SIMULACIÓN) ---
# Vamos a crear 200 datos "normales" y 10 "anomalías" (como incendios o fallos)
# El paper usa variables como Temperatura, Humedad, Albedo, etc. [cite: 11]

np.random.seed(42)

# Generamos datos normales (ej. temperatura 25°C, humedad 60%)
datos_normales = np.random.normal(loc=[25, 60], scale=[2, 5], size=(200, 2))

# Generamos anomalías (ej. temperatura 45°C por incendio, humedad muy baja)
datos_anomalos = np.random.normal(loc=[45, 10], scale=[2, 2], size=(10, 2))

# Los juntamos en una sola tabla
datos = np.vstack([datos_normales, datos_anomalos])
df = pd.DataFrame(datos, columns=['Temperatura', 'Humedad'])

print("Primeros 5 datos (Simulados):")
print(df.head())

# --- 2. PASO TÉCNICO: PCA (Resumen de datos) ---
# El paper menciona usar PCA antes de detectar anomalías.
# Esto ayuda a simplificar la información.
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(df)

# --- 3. EL DETECTIVE: ISOLATION FOREST ---
# Este es el algoritmo que mejor funcionó en el estudio[cite: 31, 402].
# contamination=0.05 significa que esperamos que aprox el 5% de datos sean anomalías.
modelo = IsolationForest(contamination=0.05, random_state=42)
modelo.fit(datos_pca)

# Predecir: 1 es normal, -1 es anomalía
prediccion = modelo.predict(datos_pca)
df['Es_Anomalia'] = prediccion

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(10, 6))

# Dibujar datos normales (Azul)
plt.scatter(df.loc[df['Es_Anomalia'] == 1, 'Temperatura'], 
            df.loc[df['Es_Anomalia'] == 1, 'Humedad'], 
            c='blue', label='Normal (Bosque Sano)')

# Dibujar anomalías (Rojo)
plt.scatter(df.loc[df['Es_Anomalia'] == -1, 'Temperatura'], 
            df.loc[df['Es_Anomalia'] == -1, 'Humedad'], 
            c='red', label='Anomalía (Posible Incendio/Burst)')

plt.title('Detección de Anomalías (Simulación basada en el Paper)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Humedad (%)')
plt.legend()
plt.grid(True)
plt.show()