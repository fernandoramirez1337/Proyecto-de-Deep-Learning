# Clasificacion Multimodal de Documentos Academicos mediante Redes Hibridas CNN-LSTM con Mecanismo de Atencion

Proyecto de Deep Learning para clasificacion de papers cientificos de arXiv en 4 categorias CS usando arquitectura hibrida CNN-LSTM con mecanismo de atencion.

## Arquitectura

**Modelo Hibrido CNN-LSTM**

Componentes:
- **CNN 1D**: Extraccion de caracteristicas de abstracts (kernel sizes 3,4,5)
- **LSTM Bidireccional**: Procesamiento secuencial de titulos (2 layers)
- **Self-Attention**: Sobre outputs del LSTM para identificar palabras relevantes del titulo
- **Global Attention**: Sobre features de CNN para ponderar segmentos del abstract
- **Weighted Attention Fusion**: Aprende importancia relativa de titulo vs abstract
- **Regularizacion**: Dropout variacional y batch normalization
- **Visualizacion**: Mapas de atencion para interpretar predicciones

Implementacion completa en PyTorch.

## Uso

Abrir `Hybrid_CNN_LSTM_Colab.ipynb` en Google Colab:

1. Subir dataset `arxiv_papers_raw.csv`
2. Ejecutar todas las celdas
3. Entrenar modelo
4. Visualizar mapas de atencion

Todo el entrenamiento y evaluacion se realiza en el notebook.

## Dataset

12,000 papers de arXiv API en 4 categorias:
- cs.AI (Artificial Intelligence)
- cs.CL (Computation and Language)
- cs.CV (Computer Vision)
- cs.LG (Machine Learning)

Formato: CSV con columnas `title`, `abstract`, `categories`

## Configuracion del Modelo

```python
VOCAB_SIZE = 50000
EMBED_DIM = 300
NUM_FILTERS = 256
KERNEL_SIZES = [3, 4, 5]
LSTM_HIDDEN = 256
DROPOUT = 0.5
BATCH_SIZE = 64
MAX_TITLE_LEN = 30
MAX_ABSTRACT_LEN = 200
```
