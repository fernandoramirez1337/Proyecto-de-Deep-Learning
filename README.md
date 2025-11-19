# ArXiv Papers Classification - SciBERT

Clasificacion de papers cientificos de arXiv en 4 categorias CS usando SciBERT.

**Modelo:** V5.0 - Cross-Attention + Back-Translation
**Test Accuracy:** 57.01%
**cs.AI Recall:** 41.89%

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
./train_v5_crossattn_aug.sh  # V5.0 (~2h M2)
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

cs.AI, cs.CL, cs.CV, cs.LG
