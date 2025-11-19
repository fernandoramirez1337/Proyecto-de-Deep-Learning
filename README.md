# ArXiv Papers Classification - SciBERT

Clasificacion de papers cientificos de arXiv en 4 categorias CS usando SciBERT.

## Resultados

**V5.0 - Cross-Attention + Back-Translation**

| Metrica | Valor | Objetivo |
|---------|-------|----------|
| Test Accuracy | 52.03% | 60% |
| cs.AI Recall | 37.45% | >30% |
| cs.CL Recall | 0.6667 | - |
| cs.CV Recall | 0.7044 | - |
| cs.LG Recall | 0.3578 | - |

**Nota:** Resultados obtenidos en Google Colab T4 GPU con dataset augmented.

## Uso

```python
from predict_optimized import OptimizedPredictor

predictor = OptimizedPredictor(
    model_path='best_scibert_v5_crossattn_aug.pth',
    model_type='cross_attention'
)

categoria = predictor.predict(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture..."
)
```

## Entrenamiento

```bash
./train_v5_crossattn_aug.sh  # V5.0 completo (~2-2.5h M2)
./train_scibert_optimized.py # V3.7 baseline (~1h M2)
```

## Estructura

```
Proyecto-de-Deep-Learning/
├── train_scibert_v5_crossattn_aug.py
├── train_v5_crossattn_aug.sh
├── advanced_cross_attention.py
├── advanced_data_augmentation.py
├── preprocessing_scibert.py
├── predict_optimized.py
├── model_scibert.py
├── train_scibert_optimized.py (V3.7 baseline)
├── backups/ (V2-V3.7)
├── scripts/
└── docs/
```

## Configuracion V5.0

```python
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.35
BATCH_SIZE = 12  # M2, 32 for GPU
LR = 5e-5
WEIGHT_DECAY = 0.01
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]
AUGMENT_SAMPLES = 450
```

## Requisitos

```
Python 3.8+
torch>=2.0
transformers
scikit-learn
pandas
```

## Categorias

- cs.AI - Inteligencia Artificial
- cs.CL - Computacion y Lenguaje
- cs.CV - Vision por Computadora
- cs.LG - Machine Learning

Ver `SOLUTION_FINAL.md` para historial completo.
