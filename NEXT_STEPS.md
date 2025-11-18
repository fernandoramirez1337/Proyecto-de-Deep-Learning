# ¿Qué hacer ahora? - Guía Rápida

**Situación:** V4.0 Focal Loss **empeoró** (-2.84% accuracy) en lugar de mejorar.

**Mejor modelo actual:** V3.7+TT (56.17% accuracy, 36.22% cs.AI recall)

---

## 🎯 Objetivo

Alcanzar **60% accuracy** manteniendo **cs.AI recall >30%**

**Gap actual:** -3.83% (necesitamos ~4% más)

---

## ✅ OPCIÓN 1: Más Rápida y Segura (30-60 min) ⭐ RECOMENDADO

### Multi-Class Threshold Tuning en V3.7

**Ventajas:**
- ✅ Sin reentrenamiento (usa V3.7 que ya funciona)
- ✅ Rápido (30-60 minutos)
- ✅ Sin riesgo de empeorar
- ✅ Mejora esperada: +1-2% → **~57-58%**

**Comando:**
```bash
python improve_v37_multiclass.py
```

**Qué hace:**
- Optimiza thresholds para TODAS las clases (no solo cs.AI)
- Prueba 2 estrategias: Greedy y Priority Search
- Evalúa en test set y compara con baseline
- Te dice si mejoró o no

**Tiempo estimado:** 30-60 minutos en M2

---

## 🔧 OPCIÓN 2: Rescatar V4.0 (15-30 min)

### Aplicar Threshold Tuning a V4.0

Aunque V4.0 es peor, tal vez threshold tuning lo rescata:

**Comando:**
```bash
python fix_v4_threshold.py
```

**Qué hace:**
- Optimiza thresholds para V4.0
- Compara con/sin thresholds
- Compara con V3.7+TT baseline

**Probabilidad de éxito:** Baja (~30%)
- V4.0 tiene problemas fundamentales (overfitting)
- Threshold tuning probablemente no compense -2.84%

---

## 🔄 OPCIÓN 3: Reintentar Focal Loss (2-3 horas)

### V4.1 con Configuración Conservadora

**Solo si tienes tiempo y quieres experimentar.**

**Cambios necesarios en `train_scibert_v4_focal.py`:**

```python
# Línea ~366-379, reemplazar con:
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.40              # Aumentado de 0.35
BATCH_SIZE = 8              # O 12 si hay memoria
EPOCHS = 10
LR = 3e-5                   # REDUCIDO de 5e-5 (clave!)
WEIGHT_DECAY = 0.015        # Aumentado de 0.01
PATIENCE = 3

# Focal Loss CONSERVADOR
FOCAL_GAMMA = 1.0           # REDUCIDO de 2.0 (clave!)
LABEL_SMOOTHING = 0.1
CLASS_WEIGHTS = None        # SIN class weights (Focal ya los maneja)
USE_ADAPTIVE_FOCAL = False  # Gamma fijo 1.0
```

**Después de cambios:**
```bash
./train_v4_focal.sh
```

**Mejora esperada:** ~56-58% (si funciona mejor que V4.0)

---

## 📊 OPCIÓN 4: Evaluar Todo (15 min)

### Comparación Comprehensiva

Después de ejecutar Opción 1 o 2:

```bash
python evaluate_all_improvements.py
```

**Qué hace:**
- Compara V3.7+TT (baseline) vs V4.0 vs V3.7+Multi-TT
- Genera gráficas y tablas
- Identifica el mejor modelo

---

## 🎯 Mi Recomendación Personal

### **EJECUTA EN ESTE ORDEN:**

#### 1️⃣ **PRIMERO:** Mejora V3.7 con multi-class thresholds
```bash
python improve_v37_multiclass.py
```
**¿Por qué?** Es rápido, seguro, y probablemente alcance ~57-58%

#### 2️⃣ **SI FUNCIONA (>57%):** Documenta y cierra
```bash
# Actualizar SOLUTION_FINAL.md con resultados
# Hacer commit
git add .
git commit -m "V3.7+Multi-TT: Mejora a X.XX% accuracy"
git push
```

#### 3️⃣ **SI NO ALCANZA 60%:** Considera otras opciones
- Data augmentation para cs.AI
- Ensemble (si tienes V2)
- Modelo más grande (RoBERTa, DeBERTa)

---

## ⏱️ Comparación de Opciones

| Opción | Tiempo | Riesgo | Mejora Esperada | Target |
|--------|--------|--------|-----------------|--------|
| **1. Multi-class Threshold V3.7** | 30-60min | Bajo | +1-2% | ~57-58% |
| 2. Threshold V4.0 | 15-30min | Medio | +0-1% | ~53-54% |
| 3. Reentrenar V4.1 | 2-3h | Alto | +0-3% | ~56-59% |
| 4. Evaluar Todo | 15min | Ninguno | - | - |

**Color code:**
- 🟢 Verde: Muy probable que mejore
- 🟡 Amarillo: Puede mejorar
- 🔴 Rojo: Poco probable que mejore

---

## 🚫 Lo Que NO Debes Hacer

❌ **NO reintentar V4.0 con misma configuración**
- Ya demostró que no funciona
- Gastará 2-3 horas sin mejora

❌ **NO combinar múltiples técnicas a la vez**
- V4.0 falló por combinar Class Weights + Focal Loss
- Mejor: una técnica a la vez, bien calibrada

❌ **NO ignorar el análisis del fallo**
- Lee POSTMORTEM_V4.md para entender qué salió mal
- Evita repetir errores

---

## ✅ Checklist de Ejecución

```bash
# Paso 1: Entender qué pasó
cat POSTMORTEM_V4.md

# Paso 2: Ejecutar mejor opción (Opción 1)
python improve_v37_multiclass.py

# Paso 3: Revisar resultados
# Buscar línea "FINAL COMPARISON" en output

# Paso 4a: Si mejoró (>57%)
echo "SUCCESS! Documenting results..."
# Actualizar SOLUTION_FINAL.md

# Paso 4b: Si NO mejoró
echo "Multi-class threshold didn't help. Trying fix_v4_threshold.py..."
python fix_v4_threshold.py
```

---

## 📈 Qué Esperar

### Escenario Optimista (70% probabilidad)
```
V3.7+Multi-TT alcanza ~57-58% accuracy
- Gap reducido de 3.83% a ~2-3%
- cs.AI recall mantiene >30%
- Mejor modelo hasta ahora
```

### Escenario Realista (20% probabilidad)
```
Multi-class threshold mejora ligeramente (~+0.5-1%)
- V3.7+Multi-TT: ~56.7-57.2%
- Aún mejor que V4.0
- Necesita otras técnicas para 60%
```

### Escenario Pesimista (10% probabilidad)
```
Multi-class threshold no mejora
- V3.7+TT sigue siendo mejor
- Necesita data augmentation o modelo más grande
```

---

## 🎓 Aprendizajes Clave

1. **Focal Loss no es mágico**
   - Requiere calibración cuidadosa (γ, α, LR)
   - No siempre supera a técnicas más simples

2. **Baseline fuerte es difícil de superar**
   - V3.7+TT ya está muy optimizado (8 versiones)
   - Mejoras incrementales requieren técnicas sofisticadas

3. **Post-training optimization es poderosa**
   - Threshold tuning dio +8% cs.AI recall en V3.7
   - Multi-class puede dar +1-2% más

4. **Iteración rápida > Soluciones complejas**
   - Mejor probar 3 técnicas simples (1h cada una)
   - Que 1 técnica compleja (3h) que puede fallar

---

## 💡 Próximos Pasos Sugeridos

### **HOY (próximos 1-2 horas):**
```bash
python improve_v37_multiclass.py
```

### **ESTA SEMANA (si no alcanzas 60%):**
- Implementar data augmentation para cs.AI
- Probar Ensemble V2+V3.7 (si tienes V2)
- Considerar modelo más grande (RoBERTa)

### **LARGO PLAZO (proyecto futuro):**
- Fine-tuning en domain-specific data
- Active learning para cs.AI
- Hybrid models (BERT + tradicional ML)

---

## 📞 ¿Necesitas Ayuda?

### Si improve_v37_multiclass.py falla:
```bash
# Verificar que V3.7 existe
ls -lh best_scibert_v3.7_final.pth best_scibert_optimized.pth

# Si no existe, entrenar V3.7 primero
python train_scibert_optimized.py
```

### Si estás confundido:
1. Lee POSTMORTEM_V4.md (explica qué salió mal)
2. Lee QUICKSTART_IMPROVEMENTS.md (guía original)
3. Consulta IMPROVEMENTS.md (documentación técnica)

---

## 🏁 Comando Para Empezar AHORA

```bash
# El comando más simple y recomendado:
python improve_v37_multiclass.py
```

**Tiempo:** 30-60 minutos
**Probabilidad de éxito:** Alta (70-80%)
**Mejora esperada:** +1-2% accuracy

🚀 **¡Adelante!**
