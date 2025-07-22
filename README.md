# 🔥 Automatización de análisis de caldera industrial con Python y OCR

Este proyecto fue desarrollado como entrega final del curso de Python. Consiste en un flujo automatizado para **extraer datos desde una imagen** con parámetros técnicos del funcionamiento de una **caldera industrial**, analizarlos y generar un informe profesional en formato Word (.docx).

---

## 🛠 Tecnologías y bibliotecas utilizadas

- `EasyOCR` – Extracción de texto desde imágenes (OCR)
- `Pandas`, `NumPy` – Procesamiento y limpieza de datos
- `Matplotlib`, `Seaborn` – Visualización y análisis gráfico
- `python-docx` – Generación automatizada de informes en Word

---

## 📁 Archivos incluidos

- `ocr_a_csv.py` → Script principal con todo el flujo automatizado
- `caldera_table_50_observations.png` → Imagen original con datos técnicos
- `caldera_datos.csv` → Datos estructurados extraídos desde la imagen
- Carpeta `graficos_analisis/` → Gráficos generados automáticamente:
  - `analisis_temporal.png`
  - `analisis_eficiencia_boxplots.png`
  - `analisis_correlacion.png`
  - `distribucion_y_anomalias.png`
- `informe_analisis_caldera.docx` → Informe final autogenerado

---

## ▶️ Cómo ejecutar el proyecto

1. Clonar este repositorio o descargar los archivos.

2. Instalar las dependencias necesarias:
```
pip install easyocr pandas numpy matplotlib seaborn python-docx
```
3. Ejecutar el script
```
python ocr_a_csv.py
```
4. El programa generará:

Un archivo .csv con los datos extraídos

Gráficos en la carpeta graficos_analisis/

Un informe .docx con el análisis completo

---

## 📊 Análisis realizados: 

- Evolución temporal de emisiones y consumo

- Boxplots por rangos de caudal

- Correlación entre variables técnicas

- Detección de anomalías y distribución de variables clave

---

👩‍🏫 Agradecimientos
Este proyecto fue desarrollado bajo la guía de la profesora Nuria Torres, en el marco del curso de Python 2025.

---

📬 Contacto
Lucas Lamiño – [https://www.linkedin.com/in/lucaslami%C3%B1o/]
