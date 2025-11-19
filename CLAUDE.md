# CLAUDE.md - AI Assistant Guide

**Project:** ArXiv Papers Classification using SciBERT
**Status:** COMPLETED

## Project Overview

Multi-class classification of ArXiv papers into 4 CS categories using SciBERT.

### Categories
1. cs.AI - Artificial Intelligence (minority class)
2. cs.CL - Computation and Language
3. cs.CV - Computer Vision
4. cs.LG - Machine Learning

### Goals
- Test Accuracy: >= 60% (Current: 57.01%, gap: -2.99%)
- cs.AI Recall: > 30% (Current: 41.89%, ACHIEVED +11.89%)
- Overfitting Gap: < 10%

### Dataset
- Original: 12,000 papers (`data/arxiv_papers_raw.csv`)
- Augmented: 12,450 papers (`data/arxiv_papers_augmented.csv`)
- Split: 70% train, 15% val, 15% test

### Final Solution
**V5.0 - Cross-Attention + Back-Translation**
- Cross-attention architecture (title ↔ abstract)
- Back-translation augmentation (EN→ES→EN, 450 cs.AI samples)
- Class weighting (cs.AI x2.0)
- Test acc: 57.01%, cs.AI recall: 41.89%
- All subsequent attempts failed (V5.0+TT, V5.1)

## Repository Structure

```
Proyecto-de-Deep-Learning/
├── V5.0 (FINAL)
│   ├── train_scibert_v5_crossattn_aug.py
│   ├── train_v5_crossattn_aug.sh
│   ├── advanced_cross_attention.py
│   └── advanced_data_augmentation.py
├── Core
│   ├── preprocessing_scibert.py
│   ├── model_scibert.py
│   └── predict_optimized.py
├── Historical
│   └── train_scibert_optimized.py (V3.7)
├── backups/ (V2-V3.7)
├── scripts/
└── docs/
    ├── README.md
    ├── V5_IMPLEMENTATION.md
    ├── SOLUTION_FINAL.md
    ├── CLAUDE.md
    └── V5_0_Training_Colab.ipynb
```

## Key Files

### train_scibert_v5_crossattn_aug.py
V5.0 training script with cross-attention + augmentation.

Config:
```python
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.35
BATCH_SIZE = 12
LR = 5e-5
WEIGHT_DECAY = 0.01
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]
AUGMENT_SAMPLES = 450
```

Output: `best_scibert_v5_crossattn_aug.pth`

### advanced_cross_attention.py
CrossAttentionSciBERT model.
- Bidirectional cross-attention (title ↔ abstract)
- 8 attention heads per direction
- Attention pooling + fusion network

### advanced_data_augmentation.py
Back-translation augmentation.
- EN → ES → EN paraphrasing
- 450 cs.AI samples (~50-60 min)

### preprocessing_scibert.py
Data pipeline.
- SciBERTDataset (dual-encoder)
- prepare_scibert_data() function
- Tokenization: title (32), abstract (128)

### predict_optimized.py
Inference pipeline.
- OptimizedPredictor class
- Loads model and label encoder

## Development Workflow

### Train New Version

```bash
# Edit hyperparameters in training script
vim train_scibert_v5_crossattn_aug.py

# Run training
./train_v5_crossattn_aug.sh

# Backup
cp train_scibert_v5_crossattn_aug.py backups/
```

### Make Predictions

```python
from predict_optimized import OptimizedPredictor

predictor = OptimizedPredictor(
    model_path='best_scibert_v5_crossattn_aug.pth',
    model_type='cross_attention'
)

category = predictor.predict(
    title="Your paper title",
    abstract="Your paper abstract..."
)
```

## Model Architecture

### CrossAttentionSciBERT

Base: SciBERT (allenai/scibert_scivocab_uncased)
- 12 transformer layers
- 768 hidden dims
- ~31K scientific vocab

Flow:
```
Input: Title (32 tokens) + Abstract (128 tokens)
↓
[SciBERT Encoder]
↓
Title [B,32,768]    Abstract [B,128,768]
↓                   ↓
[Cross-Attention: Title→Abstract, Abstract→Title]
↓                   ↓
[Attention Pooling]
↓
[Concat] → [1536]
↓
[Fusion: 512→256→128→4]
↓
[4 logits]
```

Features:
- Layer freezing (first 3 layers)
- Attention pooling
- GELU activation
- Progressive dropout
- Label smoothing
- Early stopping

## Training Pipeline

Data:
```
arxiv_papers_augmented.csv
↓
[prepare_scibert_data()]
↓
LabelEncoder + stratified split
↓
SciBERTDataset (tokenization)
↓
DataLoader (batch=12, shuffle)
↓
[Training loop]
```

Loss:
```python
CrossEntropyLoss(
    label_smoothing=0.1,
    weight=[2.0, 1.0, 1.0, 1.0]
)
```

Optimizer:
```python
AdamW([
    {'params': bert, 'lr': 5e-5, 'weight_decay': 0.01},
    {'params': classifier, 'lr': 2.5e-4, 'weight_decay': 0.02}
])
```

Scheduler: Linear warmup (10%) + decay

## Hardware: M2 MacBook Air

Config:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

DataLoader(
    batch_size=12,
    num_workers=0,
    pin_memory=False
)
```

Performance:
- ~2-3x slower than T4 GPU
- ~60-80 min per version
- Viable for development

CUDA adaptation:
```python
BATCH_SIZE = 32  # or 64
NUM_WORKERS = 4
pin_memory=True
```

## Version History

| Ver | Strategy | Freeze | Drop | Wt | Acc | csAI | Gap |
|-----|----------|--------|------|-----|-----|------|-----|
| V2 | Over-reg | 8 | 0.5 | - | 59.17 | 13.78 | 19.05 |
| V3 | Under-reg | 3 | 0.35 | - | 55.28 | 26.22 | 8.50 |
| V3.5 | Midpoint | 5-6 | 0.42 | - | 58.50 | 2.22 | 29.28 |
| V3.6 | Aggr-wt | 3 | 0.35 | 3.0 | 49.72 | 51.11 | 10.28 |
| V3.7 | Balanced | 3 | 0.35 | 2.0 | 57.39 | 28.22 | 4.39 |
| V3.7+TT | Threshold | 3 | 0.35 | 2.0 | 56.17 | 36.22 | 3.83 |
| V3.8 | Fine-wt | 3 | 0.35 | 2.3 | 49.61 | 39.78 | 10.39 |
| V4.0 | Focal | 3 | 0.35 | - | 53.33 | 28.00 | 10.45 |
| Multi-TT | Multi-th | - | - | - | 52.06 | 34.89 | 8.05 |
| Ensemble | V2+V3.7 | - | - | - | 55.33 | 31.33 | 5.34 |
| **V5.0** | **Cr-Att+Aug** | **3** | **0.35** | **2.0** | **57.01** | **41.89** | **2.99** |
| V5.0+TT | Threshold | 3 | 0.35 | 2.0 | 58.40 | 26.83 | - |
| V5.1 (C) | Stab | 3 | 0.30 | 1.4 | 54.12 | 38.03 | - |
| V5.1 (M2) | Stab | 3 | 0.30 | 1.4 | 52.25 | 38.03 | - |

Total: 15+ versions

## Key Learnings

1. Architecture > Hyperparameters
   - Cross-attention real improvement
   - Hyperparameter-only versions failed
2. Data augmentation works (450 samples optimal)
3. Law of diminishing returns (V5.0 peak)
4. Class weighting non-linear (x2.0 optimal, x2.3 collapse)
5. Don't over-optimize (V5.1 hurt performance)
6. Layer freezing critical (3 layers best)

## Constants

```python
# Data
DATA_PATH_ORIGINAL = 'data/arxiv_papers_raw.csv'
DATA_PATH_AUGMENTED = 'data/arxiv_papers_augmented.csv'

# Models
FINAL_MODEL = 'best_scibert_v5_crossattn_aug.pth'
LABEL_ENCODER = 'scibert_label_encoder.pkl'

# Config
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
TRANSLATION_EN_ES = 'Helsinki-NLP/opus-mt-en-es'
TRANSLATION_ES_EN = 'Helsinki-NLP/opus-mt-es-en'
NUM_CLASSES = 4
HIDDEN_SIZE = 768

# Hyperparameters (V5.0)
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.35
BATCH_SIZE = 12
EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 0.01
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]
PATIENCE = 3
LABEL_SMOOTHING = 0.1
AUGMENT_SAMPLES = 450

# Tokenization
MAX_TITLE_LEN = 32
MAX_ABSTRACT_LEN = 128
MAX_COMBINED_LEN = 160

# Splits
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TRAIN_SIZE = 0.70
RANDOM_STATE = 42

# Categories
CATEGORY_TO_IDX = {
    'cs.AI': 0,
    'cs.CL': 1,
    'cs.CV': 2,
    'cs.LG': 3
}
```

## Troubleshooting

### MPS Out of Memory
```python
BATCH_SIZE = 8
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.mps.empty_cache()
```

### Model Not Loading
```python
checkpoint = torch.load(path, map_location=device)
# Match architecture config
model = CrossAttentionSciBERT(dropout=0.35, freeze_bert_layers=3)
```

### Poor cs.AI Recall
```python
# Increase weight
CLASS_WEIGHTS = [2.5, 1.0, 1.0, 1.0]

# Unfreeze more layers
FREEZE_BERT_LAYERS = 2
```

### High Overfitting
```python
DROPOUT = 0.4
WEIGHT_DECAY = 0.02
FREEZE_BERT_LAYERS = 4
```

---

**Last Updated:** 2025-11-18
**Final Model:** V5.0 (57.01% acc, 41.89% cs.AI recall)
**Status:** COMPLETED
