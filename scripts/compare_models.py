# comparacion_modelos.py
"""
Script para comparar resultados de diferentes modelos
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resultados de los diferentes modelos
resultados = {
    'Modelo': [
        'Baseline (modelo inicial)',
        'Modelo v2 (regularizacin)',
        'Paso 1 (Focal Loss + Balanced Sampling)',
        'SciBERT (embeddings pre-entrenados)'
    ],
    'Test Accuracy': [
        0.588,  # Del modelo original
        0.588,  # Modelo v2 (similar)
        0.570,  # Paso 1 (baj ligeramente)
        0.6019  # SciBERT poca 1 (puede mejorar)
    ],
    'Val Accuracy': [
        0.58,
        0.58,
        0.581,
        0.6019
    ],
    'Parmetros (M)': [
        1.5,    # MultimodalAttentionClassifier
        0.5,    # ImprovedMultimodalClassifier
        0.5,    # ImprovedMultimodalClassifier con Focal Loss
        110.8   # SciBERT completo
    ],
    'Tiempo por poca (min)': [
        2,
        1.5,
        1.5,
        8
    ]
}

df = pd.DataFrame(resultados)

# Visualizacin
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy Comparison
colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1']
x = range(len(df))

ax1.bar(x, df['Test Accuracy'], color=colors, alpha=0.7, label='Test Acc')
ax1.bar(x, df['Val Accuracy'], color=colors, alpha=0.4, label='Val Acc')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Modelo'], rotation=45, ha='right')
ax1.set_ylabel('Accuracy')
ax1.set_title('Comparacin de Accuracy por Modelo', fontsize=14, fontweight='bold')
ax1.axhline(y=0.60, color='red', linestyle='--', label='Objetivo (60%)')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre las barras
for i, (test, val) in enumerate(zip(df['Test Accuracy'], df['Val Accuracy'])):
    ax1.text(i, test + 0.005, f'{test:.3f}', ha='center', fontweight='bold')

# 2. Mejora relativa
mejora = ((df['Test Accuracy'] - df['Test Accuracy'].iloc[0]) * 100).round(2)
colors_mejora = ['green' if m > 0 else 'red' if m < 0 else 'gray' for m in mejora]

ax2.barh(range(len(df)), mejora, color=colors_mejora, alpha=0.7)
ax2.set_yticks(range(len(df)))
ax2.set_yticklabels(df['Modelo'])
ax2.set_xlabel('Mejora sobre Baseline (%)')
ax2.set_title('Mejora Relativa vs Baseline', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='x')

# Agregar valores
for i, m in enumerate(mejora):
    ax2.text(m + 0.1 if m >= 0 else m - 0.1, i, f'{m:+.2f}%',
             va='center', fontweight='bold')

# 3. Parmetros vs Accuracy
ax3.scatter(df['Parmetros (M)'], df['Test Accuracy'],
           s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

for i, modelo in enumerate(df['Modelo']):
    ax3.annotate(modelo.split('(')[0].strip(),
                (df['Parmetros (M)'].iloc[i], df['Test Accuracy'].iloc[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

ax3.set_xlabel('Parmetros del Modelo (Millones)')
ax3.set_ylabel('Test Accuracy')
ax3.set_title('Parmetros vs Accuracy', fontsize=14, fontweight='bold')
ax3.axhline(y=0.60, color='red', linestyle='--', alpha=0.5)
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

# 4. Eficiencia (Accuracy / Tiempo)
eficiencia = df['Test Accuracy'] / df['Tiempo por poca (min)']

ax4.bar(range(len(df)), eficiencia, color=colors, alpha=0.7)
ax4.set_xticks(range(len(df)))
ax4.set_xticklabels(df['Modelo'], rotation=45, ha='right')
ax4.set_ylabel('Accuracy / Tiempo (1/min)')
ax4.set_title('Eficiencia de Entrenamiento', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Agregar valores
for i, e in enumerate(eficiencia):
    ax4.text(i, e + 0.005, f'{e:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('comparacion_modelos.png', dpi=150, bbox_inches='tight')
plt.show()

# Tabla resumen
print("\n" + "="*80)
print(" RESUMEN COMPARATIVO DE MODELOS")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Anlisis
print("\n ANLISIS:")
print("-"*80)

mejor_modelo_idx = df['Test Accuracy'].idxmax()
mejor_modelo = df.loc[mejor_modelo_idx, 'Modelo']
mejor_acc = df.loc[mejor_modelo_idx, 'Test Accuracy']

print(f" Mejor modelo: {mejor_modelo}")
print(f"   Test Accuracy: {mejor_acc:.4f} ({mejor_acc*100:.2f}%)")
print(f"   Mejora vs baseline: +{((mejor_acc - df['Test Accuracy'].iloc[0])*100):.2f}%")

if mejor_acc >= 0.60:
    print(f"\n OBJETIVO ALCANZADO! Accuracy >= 60%")
else:
    print(f"\n Objetivo no alcanzado. Faltan {((0.60 - mejor_acc)*100):.2f}% para llegar al 60%")

# Modelo ms eficiente
mas_eficiente_idx = eficiencia.idxmax()
print(f"\n Modelo ms eficiente: {df.loc[mas_eficiente_idx, 'Modelo']}")
print(f"   Eficiencia: {eficiencia.iloc[mas_eficiente_idx]:.4f} acc/min")

print("\n" + "="*80)
