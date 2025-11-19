# Clasificacion Multimodal de Documentos Academicos mediante Redes Hibridas CNN-LSTM

Clasificacion de papers cientificos de arXiv en 4 categorias CS usando arquitectura hibrida CNN-LSTM con mecanismo de atencion.

## Arquitectura

**Modelo Hibrido CNN-LSTM** (segun definicion del proyecto)

Componentes:
- **CNN 1D**: Extraccion de caracteristicas de abstracts (kernel sizes 3,4,5)
- **LSTM Bidireccional**: Procesamiento secuencial de titulos (2 layers)
- **Self-Attention**: Sobre outputs del LSTM para identificar palabras relevantes
- **Global Attention**: Sobre features de CNN para ponderar segmentos del abstract
- **Weighted Fusion**: Capa de atencion ponderada que aprende importancia relativa de cada modalidad
- **Regularizacion**: Dropout variacional y batch normalization
- **Visualizacion**: Mapas de atencion para interpretar predicciones

Implementacion completa en PyTorch desde cero (sin transformers).

## Uso

### Entrenamiento

```bash
./train_hybrid.sh
```

### Visualizacion de Mapas de Atencion

```python
from visualize_attention import AttentionVisualizer

viz = AttentionVisualizer('best_hybrid_model.pth', 'vocab_hybrid.pkl')

viz.visualize_sample(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture...",
    save_path='attention_viz.png'
)
```

## Estructura

```
Proyecto-de-Deep-Learning/
├── hybrid_cnn_lstm.py           # Arquitectura hibrida
├── preprocessing_hybrid.py      # Vocabulario y tokenizacion
├── train_hybrid.py              # Script de entrenamiento
├── train_hybrid.sh              # Shell script de ejecucion
├── visualize_attention.py       # Visualizacion de mapas de atencion
└── data/
    └── arxiv_papers_raw.csv     # Dataset (12,000 papers)
```

## Configuracion

```python
VOCAB_SIZE = 50000
EMBED_DIM = 300
NUM_FILTERS = 256
KERNEL_SIZES = [3, 4, 5]
LSTM_HIDDEN = 256
DROPOUT = 0.5
BATCH_SIZE = 32
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI minority class
MAX_TITLE_LEN = 30
MAX_ABSTRACT_LEN = 200
```

## Requisitos

```
Python 3.8+
torch>=2.0
scikit-learn
pandas
matplotlib
seaborn
numpy
```

## Categorias

cs.AI, cs.CL, cs.CV, cs.LG

## Objetivos

- Test Accuracy >= 60%
- cs.AI Recall > 30%

## Dataset

12,000 papers de arXiv API (CS.AI, CS.LG, CS.CV, CS.CL)
