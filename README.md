# PYDAY2025 Detección de Anomalías
🌲 Detección de Anomalías en Redes de Sensores IoT Ambientales  Este repositorio contiene la implementación en Python del sistema de detección de anomalías presentado en el PyDay La Paz. El proyecto se centra en el procesamiento de datos provenientes de nodos IoT de bajo costo (basados en ESP32) desplegados para el monitoreo hidroclimático.
.

📝 Contexto del Proyecto

El monitoreo continuo de variables hidroclimáticas es clave para entender y gestionar la recarga de acuíferos, especialmente en zonas afectadas por incendios forestales (como la Chiquitania).

Los sensores de bajo costo desplegados en campo suelen presentar ruido, desconexiones o lecturas erróneas. Este proyecto implementa un pipeline de Machine Learning no supervisado para filtrar estos errores y detectar eventos hidrológicos significativos ("bursts") de manera automática.

🚀 Características

Preprocesamiento Robusto: Limpieza de datos, manejo de valores nulos (fillna) y sincronización de series temporales.

Normalización Estadística: Uso de StandardScaler para homogeneizar escalas entre sensores heterogéneos (Humedad de suelo, Albedo, Flujo de Savia).

Detección con Isolation Forest: Implementación del algoritmo Isolation Forest para la detección de outliers con alta precisión.

Reducción de Dimensionalidad: Integración de PCA (Principal Component Analysis) para visualización y optimización.

📧 Contacto

Lucía E. Martínez Zuzunaga

Ingeniera Mecatrónica

📧 22lucia.martinez.z@gmail.com

📞 +591 60631095

🔗 linkedin.com/in/lucia-martinez-z96

Presentado en PyDay La Paz 2025 🐍
