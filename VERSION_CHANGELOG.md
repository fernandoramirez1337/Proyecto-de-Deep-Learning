# SciBERT Training Versions - Changelog

## V3.7 - Balanced cs.AI Focus (ACTUAL)
**Fecha**: 2025-11-16
**Objetivo**: Balance entre cs.AI recall y overall accuracy

### Configuración:
```python
FREEZE_BERT_LAYERS = 3      # Igual V3.6
DROPOUT = 0.35              # Igual V3.6
LR = 5e-5                   # Igual V3.6
WEIGHT_DECAY = 0.01         # Igual V3.6
CLASS_WEIGHTS = [2,1,1,1]   # cs.AI x2 (AJUSTADO: era x3)
```

### Razón:
- V3.6 demostró: **cs.AI ES DETECTABLE** (51.11% recall!)
- Pero weight x3 fue excesivo (Acc cayó a 49.72%)
- Estrategia: x2 debería balancear cs.AI recall ~35-40% y Acc ~55-57%

---

## V3.6 - Aggressive cs.AI Focus
**Fecha**: 2025-11-16
**Resultados**: ✓ cs.AI logrado pero ✗ Accuracy colapsó

### Configuración:
```python
FREEZE_BERT_LAYERS = 3      # Más capacidad que V3 (era 4)
DROPOUT = 0.35              # Menos restricción que V3 (era 0.4)
LR = 5e-5                   # Igual V3
WEIGHT_DECAY = 0.01         # Igual V3
CLASS_WEIGHTS = [3,1,1,1]   # cs.AI x3 (NUEVO)
```

### Resultados:
- Test Accuracy: **49.72%** ✗✗✗ (peor de todas)
- cs.AI Recall: **51.11%** ✓✓✓ (¡OBJETIVO LOGRADO!)
- Test F1: 0.4915
- Overfitting: +27.73%

### Problemas:
1. Modelo obsesionado con cs.AI
2. cs.CL recall cayó: 89% → 47%
3. cs.LG recall colapsó: 59% → 24%
4. Overall accuracy inaceptable

**Lección crítica**: cs.AI PUEDE ser detectado, pero x3 es excesivo

**Archivo backup**: `train_scibert_v3.6_backup.py`

---

## V3.5 - Punto Medio
**Fecha**: 2025-11-16
**Resultados**: ✗✗✗ DESASTRE (cs.AI recall: 2.22%)

### Configuración:
```python
FREEZE_BERT_LAYERS = 6      # Punto medio (V2=8, V3=4)
DROPOUT = 0.45              # Punto medio (V2=0.5, V3=0.4)
LR = 4e-5                   # Punto medio (V2=3e-5, V3=5e-5)
WEIGHT_DECAY = 0.02         # Punto medio (V2=0.05, V3=0.01)
```

### Resultados:
- Test Accuracy: **58.50%** ✗
- cs.AI Recall: **2.22%** ✗✗✗ (peor de todas!)
- Test F1: 0.5094
- Overfitting: +21.10%

### Problemas:
1. Modelo ignora cs.AI completamente
2. Punto medio NO funciona (relación no lineal)
3. Hipótesis de interpolación falló

**Archivo backup**: `train_scibert_v3.5_backup.py`

---

## V3 - Balance Optimizado
**Fecha**: 2025-11-16
**Resultados**: Accuracy bajó pero ✓ **MEJOR cs.AI recall** (26.22%)

### Cambios vs V2:
```python
FREEZE_BERT_LAYERS: 8 → 4    # Descongelar 4 capas más
DROPOUT: 0.5 → 0.4           # Reducir dropout
LR: 3e-5 → 5e-5             # Duplicar learning rate
WEIGHT_DECAY: 0.05 → 0.01   # Reducir regularización L2 a 1/5
```

### Resultados:
- Test Accuracy: 55.28% ✗
- cs.AI Recall: **26.22%** ✓ (mejor de todas las versiones!)
- Overfitting: +24.04% ✗

### Razón:
V2 estaba **sobre-regularizado**:
- Test Acc: 59.17% (no alcanzó 60%)
- cs.AI Recall: 13.78% (muy bajo, objetivo >30%)
- Overfitting gap: +23.45% (severo)

V3 busca dar más capacidad al modelo sin perder control.

**Archivo backup**: `train_scibert_v3_backup.py`

---

## V2 - Anti-Overfitting Agresivo
**Fecha**: 2025-11-16
**Resultados**: ✗ No alcanzó objetivos

### Configuración:
```python
FREEZE_BERT_LAYERS = 8
DROPOUT = 0.5
LR = 3e-5
WEIGHT_DECAY = 0.05
BATCH_SIZE = 12 (M2 optimizado)
```

### Resultados:
- Test Accuracy: **59.17%** ✗
- Test F1: 0.5489
- cs.AI Recall: **13.78%** ✗ (objetivo >30%)
- Mejor epoch: 2 (early stopping en 5)

### Problemas:
1. Modelo sobre-regularizado
2. cs.AI prácticamente no detectado
3. Overfitting severo después de época 2

**Archivo backup**: `train_scibert_v2_backup.py`

---

## V1 - SciBERT Baseline
**Fecha**: Anterior
**Resultados**: 60.5% accuracy pero con overfitting +16.5%

### Configuración:
```python
FREEZE_BERT_LAYERS = 6
DROPOUT = 0.3
LR = 5e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
```

### Problemas:
- Overfitting alto
- No optimizado para M2

---

## Experimentos a Probar

### Si V3 no alcanza objetivos:

1. **V3.1 - BERT Completo**
   ```python
   FREEZE_BERT_LAYERS = 0
   LR = 2e-5
   DROPOUT = 0.3
   EPOCHS = 5
   ```

2. **V3.2 - Class Weights Agresivos**
   ```python
   # Mismo que V3 pero:
   class_weights = [2.5, 1.0, 1.0, 1.0]  # cs.AI x2.5
   ```

3. **V3.3 - Focal Loss**
   ```python
   # Implementar Focal Loss con gamma=3.0
   # Específicamente para cs.AI
   ```

---

## Métricas de Éxito

✓ Test Accuracy >= 60%
✓ cs.AI Recall > 30%
✓ Overfitting gap < 10%
✓ Stable training (no degradación severa)
