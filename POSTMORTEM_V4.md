# Post-Mortem: V4.0 Focal Loss

**Fecha:** 2025-11-18
**Resultado:** ❌ FALLÓ - Empeoró en lugar de mejorar

---

## 📊 Resultados

| Métrica | V3.7+TT (Baseline) | V4.0 Focal | Diferencia |
|---------|-------------------|------------|------------|
| Test Accuracy | **56.17%** | 53.33% | **-2.84%** ❌ |
| cs.AI Recall | **36.22%** | 28.00% | **-8.22%** ❌ |
| Gap Total | **3.83%** | 8.67% | **+4.84%** ❌ |

**Veredicto:** V4.0 es significativamente peor que V3.7+TT

---

## 🔍 Análisis del Fallo

### 1. **Overfitting Severo**

Evolución por época:

| Epoch | Train Acc | Val Acc | Gap | Trend |
|-------|-----------|---------|-----|-------|
| 1 | 47.42% | 54.47% | -7.05% | Normal |
| 2 | 55.98% | 51.25% | +4.73% | ⚠️ Invertido |
| 3 | 62.28% | **56.36%** | +5.92% | ⚠️ Gap crece |
| 4 | 68.03% | 53.08% | +14.95% | 🔴 Overfitting |
| 5 | 71.82% | 50.19% | +21.62% | 🔴 Severo |
| 6 | 75.22% | 46.47% | **+28.75%** | 🔴 Crítico |

**Early stopping en Epoch 6 → Mejor modelo en Epoch 3**

**Problema:** El modelo memorizó el training set en lugar de generalizar.

### 2. **Test Set Collapse**

- Val Acc (Epoch 3): 56.36%
- Test Acc: 53.33%
- **Diferencia:** -3.03%

**Normal es ~1-2% gap. -3.03% indica que el modelo no generalizó bien.**

### 3. **cs.AI Recall Disminuyó**

- V3.7+TT: 36.22% cs.AI recall
- V4.0: 28.00% cs.AI recall
- **Pérdida:** -8.22% (22.7% relativo)

**Focal Loss + Class Weights no mejoró cs.AI, empeoró.**

---

## 🧪 Causas del Fallo

### Causa Raíz: **Focal Loss muy agresivo**

**Focal Loss Formula:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Configuración V4.0:**
- γ (gamma) = **2.0** (muy alto)
- α (alpha) = **[2.0, 1.0, 1.0, 1.0]** (class weights)
- Label smoothing = 0.1

**Problema:** Focal Loss con γ=2.0 **reduce drásticamente** el peso de ejemplos fáciles:
- Si p_t = 0.9 (ejemplo fácil): (1 - 0.9)^2 = 0.01 → **99% de reducción**
- Si p_t = 0.5 (ejemplo difícil): (1 - 0.5)^2 = 0.25 → 75% de reducción

**Resultado:** El modelo solo aprende de ~1% de ejemplos fáciles, causando overfitting en difíciles.

### Factores Agravantes

#### 1. **Class Weights Duplicados**
- CrossEntropy ya tiene class weights: [2.0, 1.0, 1.0, 1.0]
- Focal Loss **también** tiene alpha (class weights)
- **Efecto combinado:** cs.AI tiene peso x2 **dos veces** → Overfitting en cs.AI

#### 2. **Batch Size Reducido**
- Original V3.7: batch_size = **12**
- V4.0: batch_size = **8** (por memoria MPS)
- **Impacto:** Menos estabilidad en gradientes → Mayor varianza

#### 3. **Learning Rate Constante**
- V3.7: LR = 5e-5 funciona con CrossEntropy
- V4.0: LR = 5e-5 con Focal Loss
- **Problema:** Focal Loss requiere LR más bajo (típicamente 0.3x-0.5x del original)

---

## 🔧 Cómo Arreglarlo (Si se reintenta)

### **Opción A: Focal Loss Suave** (γ más bajo)

```python
# En train_scibert_v4_focal.py, línea ~379
FOCAL_GAMMA = 1.0  # Cambiar de 2.0 a 1.0
# O incluso 0.5 (casi CrossEntropy pero con focus leve)
```

**Impacto γ:**
- γ = 0: CrossEntropy estándar
- γ = 0.5: Focal muy suave (+10-20% peso en difíciles)
- γ = 1.0: Focal moderado (+50% peso en difíciles)
- γ = 2.0: Focal agresivo (+300% peso en difíciles) ← **Demasiado**

### **Opción B: Sin Class Weights en Focal Loss**

```python
# Focal Loss YA tiene alpha (class weights incorporado)
# No duplicar con class_weights manual

# Línea ~406
if AGGRESSIVE_CS_AI:
    class_weights = None  # Dejar que Focal Loss maneje balance
else:
    class_weights = compute_class_weights_from_dataset(...)
```

### **Opción C: Learning Rate Reducido**

```python
# Línea ~368
LR = 3e-5  # Reducir de 5e-5 a 3e-5 (60% del original)
```

### **Opción D: Dropout Aumentado**

```python
# Línea ~365
DROPOUT = 0.45  # Aumentar de 0.35 a 0.45 para más regularización
```

### **Opción E: Combinación Segura** (RECOMENDADO para V4.1)

```python
# Configuración V4.1 - Focal Loss Conservador
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.40              # +0.05 vs V3.7
BATCH_SIZE = 8              # (o 12 si hay memoria)
EPOCHS = 10
LR = 3e-5                   # -40% vs V3.7 (5e-5)
WEIGHT_DECAY = 0.015        # +50% vs V3.7 (0.01)
PATIENCE = 3

# Focal Loss CONSERVADOR
FOCAL_GAMMA = 1.0           # Mucho más suave que 2.0
LABEL_SMOOTHING = 0.1
CLASS_WEIGHTS = None        # Dejar que Focal Loss maneje balance
USE_ADAPTIVE_FOCAL = True   # Gamma 3.0→1.5 adaptativo
```

---

## 📋 Lecciones Aprendidas

### 1. **Focal Loss != Siempre Mejor**
- Funciona bien para: Object detection, imbalance severo (1:100+)
- **No siempre mejor** para: Text classification, imbalance moderado (1:3)

### 2. **No Combinar Técnicas sin Ajustar**
- V3.7: Class Weights **O** Threshold Tuning
- V4.0: Class Weights **Y** Focal Loss **Y** Threshold Tuning
- **Error:** Demasiadas técnicas de balanceo acumuladas

### 3. **Hiperparámetros de Papers != Universales**
- γ=2.0 funciona en paper original (object detection)
- **No garantiza** funcionar en clasificación de texto

### 4. **Baseline Fuerte es Difícil de Superar**
- V3.7+TT ya está bien optimizado (8 versiones de ajuste)
- Mejoras marginales (+1-2%) requieren ajuste fino

---

## ✅ Alternativas Que SÍ Funcionan

### **Alternativa 1: Multi-Class Threshold Tuning en V3.7** ⭐⭐⭐

**Por qué funciona:**
- V3.7 ya es bueno (56.17%)
- Solo optimiza post-training (sin riesgo)
- Mejora esperada: +1-2% → **~57-58%**

**Comando:**
```bash
python improve_v37_multiclass.py
```

**Tiempo:** 30-60 minutos

### **Alternativa 2: Ensemble V2 + V3.7** ⭐⭐

**Por qué funciona:**
- Combina fortalezas: V2 (alta acc) + V3.7 (alta cs.AI)
- Sin reentrenamiento
- Mejora esperada: +1.5-2.5% → **~58-59%**

**Requiere:** Tener V2 entrenado

### **Alternativa 3: Data Augmentation** ⭐

**Por qué funciona:**
- Más datos de cs.AI (clase minoritaria)
- Back-translation, synonym replacement
- Sin cambiar arquitectura

**Tiempo:** 2-3 horas

---

## 🎯 Recomendación Final

### **CORTO PLAZO (Hoy):**
```bash
# Probar multi-class threshold en V3.7
python improve_v37_multiclass.py
```
**Esperado:** ~57-58% accuracy (mejor que V4.0)

### **MEDIANO PLAZO (Esta semana):**
Si quieres reintentar Focal Loss:
```bash
# Editar train_scibert_v4_focal.py con configuración V4.1
# Cambiar: FOCAL_GAMMA=1.0, LR=3e-5, CLASS_WEIGHTS=None
python train_scibert_v4_focal.py
```
**Esperado:** ~57-59% accuracy (si se ajusta bien)

### **LARGO PLAZO (Próxima iteración):**
- Probar otros loss functions: Dice Loss, Tversky Loss
- Data augmentation para cs.AI
- Modelos más grandes: RoBERTa, DeBERTa

---

## 📌 Conclusión

**V4.0 fracasó porque:**
1. Focal Loss γ=2.0 demasiado agresivo
2. Class weights duplicados (manual + Focal alpha)
3. Learning rate muy alto para Focal Loss
4. Batch size reducido afectó estabilidad

**Mejor estrategia actual:**
- ✅ Usar V3.7+TT como baseline (56.17%)
- ✅ Aplicar multi-class threshold tuning → **~57-58%**
- ✅ Si se reintenta Focal Loss, usar configuración V4.1 conservadora

**NO reintentar V4.0 con misma configuración.**

---

**Next steps:**
```bash
# 1. Rescatar lo que podamos de V4.0
python fix_v4_threshold.py

# 2. Mejor opción: Mejorar V3.7 (ya funciona)
python improve_v37_multiclass.py

# 3. Evaluar todo
python evaluate_all_improvements.py
```
