# ArXiv Papers Classification - SciBERT

Clasificación de papers científicos de arXiv en 4 categorías de CS usando SciBERT.

## 🎯 Objetivos del Proyecto

- **Test Accuracy:** ≥ 60%
- **cs.AI Recall:** > 30%
- **Overfitting Gap:** < 10%

## 📊 Resultados Actuales

| Versión | Test Acc | cs.AI Recall | Estado |
|---------|----------|--------------|--------|
| V2 | 59.17% | 13.78% | Mejor accuracy |
| V3 | 55.28% | 26.22% | Mejor cs.AI sin weights |
| V3.5 | 58.50% | 2.22% | Desastre |
| V3.6 | 49.72% | **51.11%** | Probó cs.AI detectable |
| **V3.7** | ? | ? | **PRÓXIMO** |

## 🚀 Quick Start

### Entrenar V3.7 (actual):
```bash
./train_m2_optimized.sh
```

Tiempo: ~10-11 min/época en M2 MacBook Air

## 📁 Estructura del Proyecto

```
clasificacion_papers_dl/
├── data/
│   └── arxiv_papers_raw.csv          # 12,000 papers (3,000 por clase)
│
├── backups/                           # Versiones anteriores
│   ├── train_scibert_v2_backup.py    # V2: 59.17% acc
│   ├── train_scibert_v3_backup.py    # V3: cs.AI 26.22%
│   ├── train_scibert_v3.5_backup.py  # V3.5: DESASTRE
│   └── train_scibert_v3.6_backup.py  # V3.6: cs.AI 51.11%
│
├── scripts/                           # Utilidades
│   ├── compare_models.py
│   ├── download_data.py
│   ├── eda.py
│   └── test_pipeline.py
│
├── Archivos principales:
├── train_scibert_optimized.py         # V3.7 ACTUAL
├── preprocessing_scibert.py           # Preparación de datos
├── model_scibert.py                   # Arquitectura SciBERT
├── train_m2_optimized.sh             # Script de entrenamiento M2
│
├── Resultados:
├── best_scibert_optimized.pth        # Mejor modelo guardado
├── scibert_optimized_history.png     # Gráficas de entrenamiento
└── scibert_optimized_confusion.png   # Matriz de confusión
```

## 📚 Documentación

- **COMPARATIVA_VERSIONES.md** - Comparativa detallada de todas las versiones
- **VERSION_CHANGELOG.md** - Changelog completo con razones de cada cambio
- **M2_OPTIMIZATIONS.md** - Optimizaciones específicas para M2 MacBook Air

## 🔧 Configuración Actual (V3.7)

```python
FREEZE_BERT_LAYERS = 3          # Descongelar 9 capas de BERT
DROPOUT = 0.35                  # Regularización moderada
LR = 5e-5                       # Learning rate
WEIGHT_DECAY = 0.01             # Regularización L2
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI x2
BATCH_SIZE = 12                 # Optimizado para M2
```

## 🎓 Clases

1. **cs.AI** - Inteligencia Artificial
2. **cs.CL** - Computación y Lenguaje
3. **cs.CV** - Visión por Computadora
4. **cs.LG** - Machine Learning

## 📈 Evolución del Proyecto

### Descubrimientos Clave:

1. **V2-V3:** Menos regularización mejora cs.AI recall
2. **V3.5:** El "punto medio" no funciona (relación no lineal)
3. **V3.6:** **cs.AI ES DETECTABLE** con class weighting (51% recall!)
4. **V3.7:** Busca balance con weight x2 (en vez de x3)

### Estrategia V3.7:

V3.6 demostró que cs.AI puede detectarse con weights, pero x3 fue excesivo.
V3.7 usa x2 para balancear: cs.AI ~35-40%, Accuracy ~55-57%

## 🛠️ Requisitos

- Python 3.8+
- PyTorch 2.0+
- transformers
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

## 💻 Hardware

**Optimizado para M2 MacBook Air:**
- MPS backend
- Batch size: 12
- num_workers: 0
- Sin pin_memory

**También compatible con:**
- CUDA GPUs (batch_size puede aumentarse)
- Google Colab T4 (~2-3x más rápido)
