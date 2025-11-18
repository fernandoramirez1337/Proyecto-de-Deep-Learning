# ArXiv Papers Classification - SciBERT

Clasificacion de papers cientificos de arXiv en 4 categorias de CS usando SciBERT.

## Objetivos

- Test Accuracy: >= 60%
- cs.AI Recall: > 30%
- Overfitting Gap: < 10%

## Resultados Finales

**Modelo Final: V5.0 - Cross-Attention + Back-Translation**

| Metrica | Valor | Objetivo | Estado |
|---------|-------|----------|--------|
| Test Accuracy | 57.01% | >=60% | -2.99% |
| cs.AI Recall | 41.89% | >30% | ✅ CUMPLIDO (+11.89%) |
| Gap Total | 2.99% | - | Mejor logrado |

### Por Clase

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| cs.AI | 0.4355 | 0.4189 | 0.4270 |
| cs.CL | 0.6098 | 0.7481 | 0.6721 |
| cs.CV | 0.6484 | 0.7000 | 0.6732 |
| cs.LG | 0.5571 | 0.5156 | 0.5356 |

**Mejora sobre V3.7+TT:**
- +0.84% accuracy (56.17% → 57.01%)
- +5.67% cs.AI recall (36.22% → 41.89%)
- Arquitectura avanzada con cross-attention title↔abstract
- Data augmentation con back-translation (EN→ES→EN)

## Quick Start

### Prediccion

```python
from predict_optimized import OptimizedPredictor

# Cargar modelo V5.0
predictor = OptimizedPredictor(
    model_path='best_scibert_v5_crossattn_aug.pth',
    model_type='cross_attention'
)

# Predecir categoria
categoria = predictor.predict(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture..."
)
```

### Entrenar desde cero

```bash
# V5.0: Cross-Attention + Back-Translation (RECOMENDADO)
./train_v5_crossattn_aug.sh
```

Tiempo:
- Data augmentation: ~50-60 min (450 cs.AI samples)
- Training: ~70-90 min (augmented dataset)
- Total: ~2-2.5 horas en M2 MacBook Air

**Alternativa (solo arquitectura base):**
```bash
# V3.7: Modelo base sin augmentation (más rápido)
./train_scibert_optimized.py
```
Tiempo: ~60-80 min en M2 MacBook Air

## Estructura del Proyecto

```
Proyecto-de-Deep-Learning/
├── V5.0 (Final Model - BEST)
│   ├── train_scibert_v5_crossattn_aug.py    # V5.0 training script
│   ├── train_v5_crossattn_aug.sh            # V5.0 shell script
│   ├── advanced_cross_attention.py          # Cross-attention architecture
│   └── advanced_data_augmentation.py        # Back-translation augmentation
│
├── Core Files
│   ├── preprocessing_scibert.py             # Data preparation
│   ├── predict_optimized.py                 # Inference script
│   └── model_scibert.py                     # Base model architectures
│
├── Model & Data
│   ├── best_scibert_v5_crossattn_aug.pth   # V5.0 model (1.1GB)
│   ├── data/arxiv_papers_augmented.csv     # Augmented dataset (12,450)
│   ├── data/arxiv_papers_raw.csv           # Original dataset (12,000)
│   └── scibert_label_encoder.pkl           # Label encoder
│
├── Historical (V3.7 baseline)
│   └── train_scibert_optimized.py          # V3.7 training script
│
├── Backups
│   └── backups/                             # Historical versions (V2-V3.7)
│
├── Scripts
│   └── scripts/                             # Utility scripts
│
└── Documentation
    ├── README.md                            # This file
    ├── V5_IMPLEMENTATION.md                # V5.0 detailed documentation
    ├── SOLUTION_FINAL.md                   # Complete solution history
    ├── CLAUDE.md                           # AI assistant guide
    └── V5_1_Training_Colab.ipynb          # Google Colab notebook
```

## Configuracion Final

### Modelo V5.0 - Cross-Attention + Back-Translation

**Arquitectura:**
```python
# CrossAttentionSciBERT
FREEZE_BERT_LAYERS = 3          # 9 capas descongeladas
DROPOUT = 0.35                  # Fusion network dropout
HIDDEN_DIM = 768                # SciBERT hidden dimension

# Cross-Attention Layers
- Title → Abstract cross-attention (8 heads)
- Abstract → Title cross-attention (8 heads)
- Bidirectional interaction between modalities
```

**Hyperparametros:**
```python
LR = 5e-5                       # Learning rate
WEIGHT_DECAY = 0.01             # L2 regularization
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI x2
BATCH_SIZE = 12                 # M2 optimizado
EPOCHS = 10                     # Con early stopping (patience=3)
LABEL_SMOOTHING = 0.1           # Soft labels
```

**Data Augmentation:**
```python
# Back-Translation: EN → ES → EN
AUGMENT_CATEGORY = 'cs.AI'
AUGMENT_SAMPLES = 450           # Duplicar muestras de cs.AI
TRANSLATION_MODEL = 'Helsinki-NLP/opus-mt-{en-es,es-en}'
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

1. **Cross-Attention Efectivo**: Interaccion bidirectional title↔abstract mejora la comprension semantica
2. **Back-Translation Funciona**: Augmentar 450 cs.AI samples mejoro recall +5.67% sin overfitting
3. **Law of Diminishing Returns**: V5.0 es el pico, intentos posteriores (V5.1) empeoraron resultados
4. **Arquitectura > Fine-Tuning**: Mejoras arquitectonicas (cross-attention) superan a ajustes de hiperparametros
5. **Data Augmentation Limitada**: 450 samples optimo (~50-60 min), 3000+ samples (~6 horas) innecesario
6. **M2 Viable**: Suficiente para desarrollo, experimentacion y entrenamiento completo
7. **Class Weighting No-Lineal**: x2.0 optimo, x2.3 colapso de accuracy (-7.78%)
8. **Sweet Spot Estrecho**: V5.0 (LR 5e-5) estable pero con crash; V5.1 (LR 3e-5) too conservative

**Journey: 15+ versiones probadas**
- V2: Over-regularized (59.17% acc, 13.78% cs.AI recall)
- V3.7: Balanced (57.39% acc, 28.22% cs.AI recall)
- V3.7+TT: Threshold tuning (56.17% acc, 36.22% cs.AI recall)
- **V5.0: BEST** (57.01% acc, 41.89% cs.AI recall) ← Cross-Attention + Augmentation
- V5.1: Over-stabilized (52.25% acc, failed)

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
