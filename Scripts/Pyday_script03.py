from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns # Para graficar la matriz bonita
import matplotlib.pyplot as plt

# --- 1. PREPARAR LOS DATOS (Repaso rápido) ---
# Usamos los datos del paso anterior (df_final_pca) y las etiquetas reales (df['anomaly_labels'])
# Entrenamos el modelo nuevamente para asegurar que tenemos las predicciones frescas
modelo = IsolationForest(contamination=0.05, random_state=42)
modelo.fit(datos_pca)
predicciones = modelo.predict(datos_pca)

# --- 2. DEFINIR LA REALIDAD VS PREDICCIÓN ---
y_real = df['anomaly_labels']  # Lo que realmente pasó (-1 es anomalía)
y_predicho = predicciones      # Lo que dijo el modelo (-1 es anomalía)

# --- 3. CALCULAR MÉTRICAS ---
# pos_label=-1 le dice a Python que la clase "Importante" es la Anomalía (-1)
exactitud = accuracy_score(y_real, y_predicho)
precision = precision_score(y_real, y_predicho, pos_label=-1)

print(f"--- RESULTADOS DEL EXAMEN ---")
print(f"Exactitud (Accuracy): {exactitud:.2f} (El {exactitud*100}% de las veces acertó)")
print(f"Precisión (Precision): {precision:.2f} (De sus alertas, el {precision*100}% eran reales)")

# --- 4. VISUALIZAR LA MATRIZ DE CONFUSIÓN ---
# Esto nos muestra exactamente dónde se equivocó
matriz = confusion_matrix(y_real, y_predicho, labels=[1, -1])

plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (1)', 'Anomalía (-1)'],
            yticklabels=['Normal (1)', 'Anomalía (-1)'])
plt.xlabel('Lo que PREDJO el modelo')
plt.ylabel('La REALIDAD')
plt.title('Matriz de Confusión')
plt.show()