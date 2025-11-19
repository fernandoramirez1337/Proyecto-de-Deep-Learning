# V5.0 Implementation

## Results

**Status:** Tested - Colab T4 GPU

| Metric | Value |
|--------|-------|
| Test Acc | 52.03% |
| cs.AI Recall | 37.45% |
| cs.CL Recall | 66.67% |
| cs.CV Recall | 70.44% |
| cs.LG Recall | 35.78% |

**Note:** Results from Google Colab with augmented dataset.

## Components

### 1. Back-Translation Augmentation

`advanced_data_augmentation.py`

```python
EN → ES → EN paraphrasing
Models: Helsinki-NLP/opus-mt-{en-es,es-en}
Process: 450 cs.AI samples, ~50-60 min
Result: 12,000 → 12,450 samples
```

### 2. Cross-Attention Architecture

`advanced_cross_attention.py`

```python
Title ↔ Abstract bidirectional attention
8 attention heads per direction
Residual connections + LayerNorm
Attention pooling for sequence aggregation
```

### 3. Training

`train_scibert_v5_crossattn_aug.py`

```python
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.35
BATCH_SIZE = 12  # M2, 32 for GPU
LR = 5e-5
WEIGHT_DECAY = 0.01
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]
PATIENCE = 3
LABEL_SMOOTHING = 0.1
AUGMENT_SAMPLES = 450
```

## Usage

### Inference

```python
from advanced_cross_attention import CrossAttentionSciBERT

model = CrossAttentionSciBERT(num_classes=4, dropout=0.35, freeze_bert_layers=3)
model.load_state_dict(torch.load('best_scibert_v5_crossattn_aug.pth'))
```

### Training

```bash
./train_v5_crossattn_aug.sh
# Augmentation: ~50-60 min
# Training: ~70-90 min
# Total: ~2-2.5h M2
```

### Colab

Use `V5_0_Training_Colab.ipynb` (~40-50 min T4 GPU)

## Key Findings

1. Architecture > Hyperparameters
2. Data augmentation (450 samples optimal)
3. Class weights x2.0 effective
4. Layer freezing critical (3 layers)

---

Created: 2025-11-18
Model: V5.0 Cross-Attention + Back-Translation
