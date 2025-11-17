# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('arxiv_papers_raw.csv')

print("=== ANLISIS EXPLORATORIO ===")
print(f"Total papers: {len(df)}")
print(f"\nLongitud promedio de ttulos: {df['title'].str.split().str.len().mean():.1f} palabras")
print(f"Longitud promedio de abstracts: {df['abstract'].str.split().str.len().mean():.1f} palabras")

# Distribucin de categoras
plt.figure(figsize=(10, 6))
df['category'].value_counts().plot(kind='bar')
plt.title('Distribucin de Papers por Categora')
plt.xlabel('Categora')
plt.ylabel('Cantidad')
plt.tight_layout()
plt.savefig('distribucion_categorias.png')
plt.show()

# Verificar datos faltantes
print(f"\nDatos faltantes:")
print(df.isnull().sum())

# Ejemplos de datos
print("\n=== EJEMPLOS DE DATOS ===")
for cat in df['category'].unique():
    sample = df[df['category'] == cat].iloc[0]
    print(f"\n[{cat}]")
    print(f"Ttulo: {sample['title'][:100]}...")
    print(f"Abstract: {sample['abstract'][:200]}...")