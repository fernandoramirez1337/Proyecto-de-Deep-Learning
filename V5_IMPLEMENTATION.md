# V5.0 Implementation

## Results

**Status:** COMPLETED - Final model

| Metric | V3.7+TT | V5.0 | Change |
|--------|---------|------|--------|
| Test Acc | 56.17% | 57.01% | +0.84% |
| cs.AI Recall | 36.22% | 41.89% | +5.67% |

**Subsequent attempts failed:**
- V5.0+TT: 58.40% acc, 26.83% cs.AI recall
- V5.1: 52.25% acc (M2), 54.12% acc (Colab)

## Components

### 1. Back-Translation Augmentation

`advanced_data_augmentation.py`

```python
EN -> ES -> EN paraphrasing
Models: Helsinki-NLP/opus-mt-{en-es,es-en}
Process: 450 cs.AI samples, ~50-60 min
Result: 12,000 -> 12,450 samples
```

### 2. Cross-Attention Architecture

`advanced_cross_attention.py`

```python
Title <-> Abstract bidirectional attention
8 attention heads per direction
Residual connections + LayerNorm
Attention pooling for sequence aggregation
```

### 3. Training

`train_scibert_v5_crossattn_aug.py`

```python
FREEZE_BERT_LAYERS = 3
DROPOUT = 0.35
BATCH_SIZE = 12
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

## Version History

| Version | Strategy | Acc | cs.AI | Status |
|---------|----------|-----|-------|--------|
| V2 | Over-reg | 59.17% | 13.78% | Failed |
| V3 | Under-reg | 55.28% | 26.22% | Failed |
| V3.7 | Balanced | 57.39% | 28.22% | Good |
| V3.7+TT | Threshold | 56.17% | 36.22% | Previous best |
| V3.8 | Over-weight | 49.61% | 39.78% | Failed |
| V4.0 | Focal Loss | 53.33% | 28.00% | Failed |
| Multi-TT | Multi-threshold | 52.06% | 34.89% | Failed |
| Ensemble | V2+V3.7 | 55.33% | 31.33% | Failed |
| **V5.0** | **Cross-Attn+Aug** | **57.01%** | **41.89%** | **Best** |
| V5.0+TT | Threshold | 58.40% | 26.83% | Failed |
| V5.1 | Stabilized | 52.25% | 38.03% | Failed |

Total: 15+ versions

## Key Findings

1. Architecture > Hyperparameters
2. Data augmentation (450 samples optimal)
3. Law of diminishing returns (V5.0 peak)
4. Class weights x2.0 optimal
5. Sweet spot narrow (easy overshoot)
6. Layer freezing critical (3 layers)

---

Created: 2025-11-18
Model: V5.0 Cross-Attention + Back-Translation
Results: 57.01% acc, 41.89% cs.AI recall
