# ArXiv Papers Classification - SciBERT

Clasificacion de papers cientificos de arXiv en 4 categorias de CS usando SciBERT.

## Objetivos

- Test Accuracy: >= 60%
- cs.AI Recall: > 30%
- Overfitting Gap: < 10%

## Resultados Finales

**Modelo Final: V3.7 + Threshold Tuning (threshold=0.40)**

| Metrica | Valor | Objetivo | Estado |
|---------|-------|----------|--------|
| Test Accuracy | 56.17% | >=60% | -3.83% |
| cs.AI Recall | 36.22% | >30% | CUMPLIDO |
| Gap Total | 3.83% | - | Mejor logrado |

### Por Clase

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| cs.AI | 0.3639 | 0.3622 | 0.3179 |
| cs.CL | 0.5757 | 0.8111 | 0.6734 |
| cs.CV | 0.6731 | 0.7733 | 0.7198 |
| cs.LG | 0.6433 | 0.4289 | 0.5147 |

## Quick Start

### Prediccion

```python
from predict_optimized import OptimizedPredictor

# Crear predictor con threshold optimizado
predictor = OptimizedPredictor(threshold_cs_ai=0.40)

# Predecir categoria
categoria = predictor.predict(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture..."
)
```

### Entrenar desde cero

```bash
./train_m2_optimized.sh
```

Tiempo: ~60-80 min en M2 MacBook Air

## Estructura del Proyecto

```
clasificacion_papers_dl/
├── Core Files
│   ├── train_scibert_optimized.py      # Training script
│   ├── predict_optimized.py            # Inference script
│   ├── preprocessing_scibert.py        # Data preparation
│   ├── model_scibert.py                # Model architecture
│   └── threshold_tuning.py             # Threshold optimization
│
├── Model & Data
│   ├── best_scibert_v3.7_final.pth    # Final model (1.1GB)
│   ├── scibert_label_encoder.pkl      # Label encoder
│   └── data/arxiv_papers_raw.csv      # 12,000 papers
│
├── Scripts
│   └── scripts/                        # Utility scripts
│
└── Documentation
    ├── README.md                       # This file
    └── SOLUTION_FINAL.md              # Complete solution docs
```

## Configuracion Final

### Modelo V3.7
```python
FREEZE_BERT_LAYERS = 3          # 9 capas descongeladas
DROPOUT = 0.35
LR = 5e-5
WEIGHT_DECAY = 0.01
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI x2
BATCH_SIZE = 12                 # M2 optimizado
```

### Threshold Tuning
```python
THRESHOLD_CS_AI = 0.40          # Optimizado experimentalmente
```

## Optimizaciones M2 MacBook Air

El proyecto esta optimizado para entrenar en M2 MacBook Air:

### Configuracion MPS
```python
# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

# M2 optimization
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# DataLoader
DataLoader(
    dataset, 
    batch_size=12,      # Reducido para M2
    num_workers=0,      # Sin multiprocessing en MPS
    pin_memory=False,   # No soportado en MPS
    persistent_workers=False
)
```

### Rendimiento
- Batch size: 12 (vs 32 en GPU T4)
- Velocidad: ~2-3x mas lento que T4 GPU
- Tiempo total (8 versiones): ~10-12 horas
- Memoria: Funciona en 8-16GB unified memory

## Clases

1. cs.AI - Inteligencia Artificial
2. cs.CL - Computacion y Lenguaje  
3. cs.CV - Vision por Computadora
4. cs.LG - Machine Learning

## Hallazgos Clave

1. **Threshold Tuning > Aggressive Weighting**: Ajustar threshold (0.40) supero al fine-tuning agresivo de pesos
2. **Class Weighting No-Lineal**: x2.0 optimo, x2.3 colapso de accuracy
3. **M2 Viable**: Suficiente para desarrollo y experimentacion
4. **Trade-off Aceptable**: -1.22% accuracy por +8% cs.AI recall

## Requisitos

```
Python 3.8+
torch>=2.0
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
```

## Hardware

**Optimizado para M2 MacBook Air**
- MPS backend
- Batch size: 12
- num_workers: 0

**Compatible con:**
- CUDA GPUs (aumentar batch_size)
- Google Colab T4

## Ver Documentacion Completa

Para detalles completos sobre el desarrollo, versiones probadas, y analisis:
- `SOLUTION_FINAL.md` - Documentacion completa de la solucion

---

Dataset: 12,000 papers arXiv  
Modelo: SciBERT (allenai/scibert_scivocab_uncased)  
Framework: PyTorch + Transformers
