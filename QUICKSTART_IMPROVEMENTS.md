# Quick Start - Mejoras Implementadas

**⏱️ Tiempo estimado:** 5-10 minutos lectura + 1-4 horas ejecución

---

## 🎯 Objetivo

Cerrar el gap de -3.83% y alcanzar **60% accuracy** manteniendo **cs.AI recall >30%**

**Estado actual (V3.7+TT):**
- Test Accuracy: 56.17% ❌ (necesitamos +3.83%)
- cs.AI Recall: 36.22% ✅ (objetivo cumplido)

---

## 🚀 Opción 1: Más Rápida (1-2 horas)

**Recomendación:** Si ya tienes V3.7 entrenado

### Paso 1: Optimizar Thresholds Multi-Clase
```bash
python -c "
from threshold_optimizer import ThresholdOptimizer
from preprocessing_scibert import prepare_scibert_data
from train_scibert_optimized import OptimizedSciBERTClassifier
import torch
from torch.utils.data import DataLoader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
_, val_dataset, _, tokenizer, le = prepare_scibert_data()
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

model = OptimizedSciBERTClassifier(num_classes=4, dropout=0.35, freeze_bert_layers=3)
checkpoint = torch.load('best_scibert_v3.7_final.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

optimizer = ThresholdOptimizer(model, val_loader, device, le)
best_thresholds, _, _ = optimizer.greedy_search()
print(f'\n✓ Optimized thresholds: {best_thresholds}')
" > optimize_thresholds.log 2>&1

tail -f optimize_thresholds.log
```

**Mejora esperada:** +1-2% accuracy → **~57-58%**

---

## 🔥 Opción 2: Focal Loss (2-3 horas) ⭐ RECOMENDADO

**Mejor opción para alcanzar 60%**

### Paso 1: Entrenar V4.0 con Focal Loss
```bash
# Método 1: Usando script shell (recomendado)
./train_v4_focal.sh

# Método 2: Python directo
python train_scibert_v4_focal.py
```

**Tiempo:** ~60-80 minutos en M2
**Mejora esperada:** +2-3% accuracy → **~58-59%**

### Paso 2: Optimizar Thresholds en V4.0
```bash
python -c "
from threshold_optimizer import ThresholdOptimizer
from preprocessing_scibert import prepare_scibert_data
from train_scibert_optimized import OptimizedSciBERTClassifier
import torch
from torch.utils.data import DataLoader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
_, val_dataset, _, tokenizer, le = prepare_scibert_data()
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

model = OptimizedSciBERTClassifier(num_classes=4, dropout=0.35, freeze_bert_layers=3)
checkpoint = torch.load('best_scibert_v4_focal.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

optimizer = ThresholdOptimizer(model, val_loader, device, le)
best_thresholds, metrics, _ = optimizer.greedy_search()
print(f'\n✓ V4.0 Optimized thresholds: {best_thresholds}')
print(f'✓ Accuracy: {metrics[\"accuracy\"]*100:.2f}%')
print(f'✓ cs.AI Recall: {metrics[\"cs_ai_recall\"]*100:.2f}%')
"
```

**Mejora esperada total:** +3-4% accuracy → **~59-60%** ✅

---

## 🏆 Opción 3: Máxima Precisión - Ensemble (3-5 horas)

**Solo si tienes V2 y V3.7 entrenados**

### Paso 1: Verificar modelos disponibles
```bash
ls -lh best_scibert_v2.pth best_scibert_v3.7_final.pth
```

### Paso 2: Evaluar Ensemble
```bash
python -c "
from ensemble_predictor import create_ensemble_v2_v37
from preprocessing_scibert import prepare_scibert_data
from torch.utils.data import DataLoader

_, _, test_dataset, _, le = prepare_scibert_data()
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

ensemble = create_ensemble_v2_v37(
    weights=[0.4, 0.6],              # V2: 40%, V3.7: 60%
    thresholds=[0.40, 0.5, 0.5, 0.5]
)

accuracy, preds, labels = ensemble.evaluate(test_loader)
print(f'\n✓ Ensemble accuracy: {accuracy*100:.2f}%')
"
```

**Mejora esperada:** +2-3% accuracy → **~58-59%**

---

## 📊 Evaluar Todo

Después de implementar mejoras:

```bash
python evaluate_all_improvements.py
```

**Outputs:**
- `model_comparison.csv` - Tabla de resultados
- `model_comparison.png` - Gráficas comparativas
- Reporte en consola

---

## 🆘 Si algo falla

### Error: "No module named 'torch'"
```bash
pip install torch transformers scikit-learn matplotlib seaborn pandas tqdm
```

### Error: "Dataset not found"
```bash
# Asegúrate de que existe data/arxiv_papers_raw.csv
mkdir -p data
# Coloca el dataset en data/arxiv_papers_raw.csv
```

### Error: "MPS out of memory"
```python
# En train_scibert_v4_focal.py, línea ~366
BATCH_SIZE = 8  # Reducir de 12 a 8
```

### Error: "Model not found"
```bash
# Primero entrena el modelo base
python train_scibert_optimized.py
```

---

## 📈 Tracking del Progreso

| Paso | Acción | Tiempo | Accuracy Esperada | Status |
|------|--------|--------|-------------------|--------|
| 0 | Baseline V3.7+TT | - | 56.17% | ✅ Done |
| 1 | Entrenar V4.0 Focal | 1-2h | 58-59% | ⏳ Pending |
| 2 | Optimizar thresholds V4.0 | 30min | 59-60% | ⏳ Pending |
| 3 | Evaluar todo | 15min | - | ⏳ Pending |

---

## 🎯 Decisión Rápida

### ¿Tienes 1 hora?
→ **Opción 1:** Multi-class threshold tuning en V3.7

### ¿Tienes 2-3 horas?
→ **Opción 2:** Focal Loss (V4.0) + threshold tuning ⭐

### ¿Tienes 4-5 horas?
→ **Opción 3:** Ensemble + Focal Loss + threshold tuning

---

## ✅ Checklist de Ejecución

```bash
# 1. Verificar dependencias
python -c "import torch, transformers, sklearn; print('✓ Dependencies OK')"

# 2. Verificar dataset
test -f data/arxiv_papers_raw.csv && echo "✓ Dataset OK" || echo "✗ Dataset missing"

# 3. Entrenar V4.0 (RECOMENDADO)
./train_v4_focal.sh

# 4. Optimizar thresholds
# (Copiar comando de Opción 2, Paso 2)

# 5. Evaluar todo
python evaluate_all_improvements.py

# 6. Revisar resultados
cat model_comparison.csv
open model_comparison.png
```

---

## 📚 Documentación Completa

Para detalles técnicos completos:
- **IMPROVEMENTS.md** - Explicación técnica de todas las mejoras
- **focal_loss.py** - Implementación de Focal Loss
- **ensemble_predictor.py** - Implementación de Ensemble
- **threshold_optimizer.py** - Optimizador multi-clase
- **evaluate_all_improvements.py** - Evaluador comprehensivo

---

**¿Listo para empezar?**

```bash
# El comando más simple para empezar:
./train_v4_focal.sh
```

🚀 **Good luck!**
