"""
Data Augmentation para cs.AI - Estrategia Final

SITUACIÓN:
- V3.7+TT: 56.17% accuracy (mejor modelo)
- V4.0 Focal: Falló (53.33%)
- Multi-class threshold: Falló (52.06%)

OBJETIVO:
- Aumentar datos de cs.AI (clase minoritaria)
- Mejorar representación sin overfitting
- Target: 58-60% accuracy

TÉCNICAS:
1. Back-Translation (inglés → español → inglés)
2. Synonym Replacement (EDA)
3. Paraphrase Generation (usando T5/BART)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

print("="*70)
print("DATA AUGMENTATION STRATEGY FOR cs.AI")
print("="*70)

# Analizar dataset actual
print("\nAnalyzing current dataset...")
df = pd.read_csv('data/arxiv_papers_raw.csv')

print(f"\nDataset statistics:")
print(f"Total papers: {len(df)}")
print(f"\nClass distribution:")
class_counts = df['category'].value_counts()
for category, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {category}: {count} ({percentage:.1f}%)")

cs_ai_count = class_counts.get('cs.AI', 0)
cs_ai_percentage = (cs_ai_count / len(df)) * 100

print(f"\n🎯 Target class: cs.AI")
print(f"   Current: {cs_ai_count} samples ({cs_ai_percentage:.1f}%)")
print(f"   Target: ~{cs_ai_count * 2} samples (double)")

# Estrategia de augmentation
print("\n" + "="*70)
print("AUGMENTATION STRATEGIES")
print("="*70)

print("""
Strategy 1: BACK-TRANSLATION ⭐⭐⭐
- Translate: English → Spanish → English
- Creates paraphrases naturally
- Preserves semantic meaning
- Tools: googletrans, deep-translator, MarianMT

Pros:
✅ Simple to implement
✅ High quality paraphrases
✅ No training required

Cons:
❌ Slow (API calls)
❌ May need manual review

Expected improvement: +1-2% accuracy


Strategy 2: SYNONYM REPLACEMENT (EDA) ⭐⭐
- Replace words with synonyms from WordNet
- Random insertion/deletion/swap
- Tool: nlpaug library

Pros:
✅ Fast (offline)
✅ Configurable intensity

Cons:
❌ May change meaning if too aggressive
❌ Limited to English synonyms

Expected improvement: +0.5-1.5% accuracy


Strategy 3: PARAPHRASE GENERATION ⭐⭐⭐
- Use T5/PEGASUS model fine-tuned on paraphrasing
- High quality semantic-preserving rewrites
- Tool: Hugging Face transformers

Pros:
✅ High quality
✅ Semantic preservation

Cons:
❌ Requires GPU (slow on CPU)
❌ Model download (~1GB)

Expected improvement: +1-2% accuracy


Strategy 4: MIXUP in Embedding Space ⭐
- Interpolate embeddings of cs.AI samples
- Creates "virtual" samples
- Implemented during training

Pros:
✅ No external tools
✅ Proven technique

Cons:
❌ Requires retraining
❌ Risk of overfitting

Expected improvement: +0.5-1% accuracy
""")

# Recomendación
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("""
🏆 BEST APPROACH: Combination of Strategy 1 + 2

1. Back-Translation (using MarianMT - offline)
   - Augment 50% of cs.AI samples
   - ~150 new samples

2. Synonym Replacement (using nlpaug)
   - Augment remaining 50%
   - ~150 new samples

Total: ~300 new cs.AI samples (double current)

EXPECTED RESULTS:
- Dataset: 12,000 → 12,300 samples
- cs.AI: 25% → 29% of dataset (more balanced)
- Accuracy improvement: +1.5-2.5%
- Target: 56.17% → 58-59% ✅

TIME ESTIMATE:
- Implementation: 1-2 hours
- Augmentation: 30-60 minutes
- Retraining: 60-80 minutes
Total: 3-4 hours
""")

# Código de ejemplo
print("\n" + "="*70)
print("IMPLEMENTATION EXAMPLE")
print("="*70)

print("""
# Install dependencies
pip install nlpaug transformers sentencepiece

# Example 1: Synonym Replacement (EDA)
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')

original = "Deep learning for artificial intelligence"
augmented = aug.augment(original)
print(f"Original: {original}")
print(f"Augmented: {augmented}")

# Example 2: Back-Translation (MarianMT)
from transformers import MarianMTModel, MarianTokenizer

# Load models (once)
model_en_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
tokenizer_en_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
model_es_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
tokenizer_es_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')

def back_translate(text):
    # English → Spanish
    inputs = tokenizer_en_es(text, return_tensors="pt", padding=True)
    translated = model_en_es.generate(**inputs)
    spanish = tokenizer_en_es.decode(translated[0], skip_special_tokens=True)

    # Spanish → English
    inputs = tokenizer_es_en(spanish, return_tensors="pt", padding=True)
    back_translated = model_es_en.generate(**inputs)
    english = tokenizer_es_en.decode(back_translated[0], skip_special_tokens=True)

    return english

original = "We propose a novel neural network architecture"
augmented = back_translate(original)
print(f"Original: {original}")
print(f"Augmented: {augmented}")

# Apply to cs.AI samples
cs_ai_df = df[df['category'] == 'cs.AI'].copy()

augmented_samples = []
for idx, row in cs_ai_df.iterrows():
    # Augment abstract with back-translation
    aug_abstract = back_translate(row['abstract'])

    augmented_samples.append({
        'title': row['title'],
        'abstract': aug_abstract,
        'category': 'cs.AI'
    })

# Create augmented dataset
augmented_df = pd.DataFrame(augmented_samples)
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Save
combined_df.to_csv('data/arxiv_papers_augmented.csv', index=False)
print(f"Augmented dataset saved: {len(combined_df)} samples")
""")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("""
1. Install dependencies:
   pip install nlpaug transformers sentencepiece

2. Run augmentation script:
   python augment_cs_ai_data.py

3. Retrain V3.7 with augmented data:
   python train_scibert_optimized.py --augmented

4. Evaluate:
   python evaluate_all_improvements.py

Expected timeline: 3-4 hours
Expected result: 58-59% accuracy
""")

print("\n" + "="*70)
print("\nWould you like me to implement the full augmentation pipeline?")
print("="*70 + "\n")
