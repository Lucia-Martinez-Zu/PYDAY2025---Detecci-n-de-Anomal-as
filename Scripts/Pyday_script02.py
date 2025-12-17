import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. CARGAR DATOS ---
# Leemos el archivo que generamos en el paso anterior
df = pd.read_csv('datos_hidroclimaticos_24h.csv')

print("--- Paso 1: Datos Originales (Primeras 3 filas) ---")
print(df.head(3))

# --- 2. SELECCIÓN Y LIMPIEZA ---
# Seleccionamos solo las columnas numéricas de los sensores (Variables del paper)
columnas_sensores = ['temperature_C', 'RH_percent', 'soil_moisture_percent', 'sap_flow_gph', 'albedo']
datos_crudos = df[columnas_sensores]

# Verificamos si falta algún dato (Limpieza básica)
# dropna() elimina filas si algún sensor falló y dejó el campo vacío
datos_limpios = datos_crudos.dropna()

print(f"\n--- Paso 2: Datos listos para procesar ---")
print(f"Filas originales: {len(df)}, Filas tras limpieza: {len(datos_limpios)}")

# --- 3. NORMALIZACIÓN (STANDARD SCALER) ---
# Esto es vital. Fíjate que el Albedo es 0.15 y la Humedad es 80.
# Necesitamos ponerlos en la misma escala.
escalador = StandardScaler()
datos_normalizados = escalador.fit_transform(datos_limpios)

# Convertimos a DataFrame (opcional)
df_norm = pd.DataFrame(datos_normalizados, columns=columnas_sensores)
print("\n--- Paso 3: Datos Normalizados (Mira cómo cambiaron los valores) ---")
print("Ahora el promedio de cada columna es cercano a 0")
print(df_norm.head(3))

# --- 4. PCA (MÉTODO DEL PAPER) ---
# El paper menciona reducir la información antes de buscar anomalías.
# Vamos a comprimir las 5 variables en 2 "Componentes Principales".
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_normalizados)

# Creamos un DataFrame final con el resultado del PCA
df_final_pca = pd.DataFrame(data=datos_pca, columns=['Componente_1', 'Componente_2'])

print("\n--- Paso 4: Resultado final (PCA) ---")
print("Estos son los datos que realmente entrarán al 'Detective' (Isolation Forest)")
print(df_final_pca.head(3))