# Comparativa de Versiones - SciBERT

## Resultados Actuales

| Versión | Test Acc | cs.AI Recall | Overfit Gap | Estado |
|---------|----------|--------------|-------------|---------|
| **V2** | 59.17% | 13.78% ✗ | +23.45% ✗ | Sobre-regularizado |
| **V3** | 55.28% ✗ | 26.22% | +24.04% ✗ | Mejor cs.AI |
| **V3.5** | 58.50% ✗ | **2.22%** ✗✗✗ | +21.10% ✗ | Desastre |
| **V3.6** | **49.72%** ✗✗✗ | **51.11%** ✓✓✓ | +27.73% ✗ | cs.AI OK, Acc colapsó |
| **V3.7** | ? | ? | ? | **PRÓXIMO** |

## Configuración Detallada

| Parámetro | V2 | V3 | V3.5 | V3.6 | **V3.7** | Notas |
|-----------|----|----|------|------|----------|-------|
| `FREEZE_BERT_LAYERS` | 8 | 4 | 6 | 3 | **3** | Mantener |
| `DROPOUT` | 0.5 | 0.4 | 0.45 | 0.35 | **0.35** | Mantener |
| `LR` | 3e-5 | 5e-5 | 4e-5 | 5e-5 | **5e-5** | Mantener |
| `WEIGHT_DECAY` | 0.05 | 0.01 | 0.02 | 0.01 | **0.01** | Mantener |
| `CLASS_WEIGHTS` | balanced | balanced | balanced | [3,1,1,1] | **[2,1,1,1]** | x2 (era x3) |
| `BATCH_SIZE` | 12 | 12 | 12 | 12 | 12 | M2 optimizado |
| `PATIENCE` | 3 | 3 | 3 | 3 | 3 | Mantener |

## Análisis

### Por qué V2 falló:
✗ Demasiada regularización
✗ cs.AI recall muy bajo (13.78%)
✗ Modelo no pudo aprender suficiente

### Por qué V3 falló:
✗ Muy poca regularización
✗ Accuracy bajó a 55.28%
✗ Overfitting más severo (+24%)
✓ **PERO** mejor cs.AI recall (26.22%)

### Por qué V3.5 FALLÓ CATASTRÓFICAMENTE:
✗✗✗ cs.AI recall: **2.22%** (peor de todas!)
✗ Test Acc: 58.50% (no mejoró)
✗ Overfitting: +21.10%

**Lección aprendida**: El punto medio NO funciona. La relación no es lineal.

### Por qué V3.6 SOBRE-CORRIGIÓ:
✓✓✓ cs.AI recall: **51.11%** (¡OBJETIVO LOGRADO!)
✗✗✗ Test Acc: **49.72%** (colapsó, peor de todos)
✗ Overfitting: +27.73%

**Problema**: Weight x3 fue demasiado agresivo
- Modelo obsesionado con cs.AI
- cs.CL recall cayó: 89% → 47%
- cs.LG recall colapsó: 59% → 24%

**Lección**: cs.AI ES DETECTABLE, pero x3 es excesivo

### Por qué V3.7 debería funcionar:

**Ajuste fino**: V3.6 demostró que cs.AI puede detectarse

```
V3 (no weight): cs.AI 26%, Acc 55%
       ↓
V3.6 (x3): cs.AI 51%, Acc 50%  ← Demasiado
       ↓
V3.7 (x2): cs.AI ~35-40%, Acc ~55-57%  ← Balance
```

**V3.7 = V3.6 moderado**:
- Configuración igual que V3.6
- **AJUSTE**: Class weight cs.AI x2 (en vez de x3)
- Hipótesis: cs.AI 35-40%, Test Acc 55-57%

## Estrategia de Búsqueda

```
Intentado:
├── V2: freeze=8, drop=0.5, lr=3e-5, wd=0.05 → 59.17% acc, cs.AI 13.78%
├── V3: freeze=4, drop=0.4, lr=5e-5, wd=0.01 → 55.28% acc, cs.AI 26.22%
├── V3.5: freeze=6, drop=0.45, lr=4e-5, wd=0.02 → 58.50% acc, cs.AI 2.22% ✗✗✗
├── V3.6: freeze=3, drop=0.35, lr=5e-5, wd=0.01, cs.AI x3 → 49.72% acc, cs.AI 51.11% ✓
└── V3.7: igual V3.6 pero cs.AI x2 → ? (PRÓXIMO)

Si V3.7 falla:
├── V3.8: cs.AI x1.5 (ajuste fino)
├── V3.9: Full BERT (freeze=0), lr=2e-5, cs.AI x2
└── V4.0: Focal Loss para cs.AI
```

## Archivos del Proyecto

**Código Principal:**
- `train_scibert_optimized.py` - **V3.7 actual**
- `preprocessing_scibert.py` - Preparación de datos
- `model_scibert.py` - Arquitectura del modelo

**Backups:** (directorio `backups/`)
- `train_scibert_v2_backup.py` - V2 (59.17% acc, cs.AI 13.78%)
- `train_scibert_v3_backup.py` - V3 (55.28% acc, cs.AI 26.22%)
- `train_scibert_v3.5_backup.py` - V3.5 (58.50% acc, cs.AI 2.22%)
- `train_scibert_v3.6_backup.py` - V3.6 (49.72% acc, cs.AI 51.11%)

## Ejecutar V3.7

```bash
./train_m2_optimized.sh
```

Tiempo esperado: ~10-11 min/época en M2 Air
