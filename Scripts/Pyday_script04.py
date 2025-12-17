import matplotlib.pyplot as plt

# Configuramos una figura grande con 5 filas (una para cada variable)
variables = ['temperature_C', 'RH_percent', 'soil_moisture_percent', 'sap_flow_gph', 'albedo']
titulos = ['Temperatura (°C)', 'Humedad Relativa (%)', 'Humedad del Suelo (%)', 'Flujo de Savia (gph)', 'Albedo (0-1)']
colores = ['tab:red', 'tab:blue', 'tab:brown', 'tab:green', 'tab:orange']

fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

for i, var in enumerate(variables):
    # Graficamos la línea de tiempo normal
    axs[i].plot(df_sintetico['Timestamp'], df_sintetico[var], color=colores[i], label='Dato Sensor')
    
    # Resaltamos los puntos donde sabemos que hay anomalías
    datos_anomalos = df_sintetico[df_sintetico['anomaly_labels'] == -1]
    axs[i].scatter(datos_anomalos['Timestamp'], datos_anomalos[var], color='red', s=50, label='Anomalía Etiquetada', zorder=5)
    
    axs[i].set_ylabel(titulos[i], fontsize=10)
    axs[i].grid(True, linestyle='--', alpha=0.5)
    axs[i].legend(loc='upper right')

plt.xlabel('Hora del día')
plt.suptitle('Visualización de Variables Hidroclimáticas (24 Horas)', fontsize=16)
plt.tight_layout()
plt.show()