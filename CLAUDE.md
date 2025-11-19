# Hybrid CNN-LSTM ArXiv Paper Classification

## Project Overview

Academic paper classification system using an optimized hybrid CNN-LSTM architecture with multi-head attention mechanisms to classify arXiv papers into Computer Science categories.

**Current Status:** Optimized architecture implemented and ready for testing
**Baseline Performance:** 59.33% test accuracy (previous implementation)
**Target Performance:** 65-70%+ test accuracy

## Project Definition (Requirements)

Title: "Clasificación Multimodal de Documentos Académicos mediante Redes Híbridas CNN-LSTM con Mecanismo de Atención"

**Required Components:**
- ✓ CNN 1D for abstract feature extraction
- ✓ Bidirectional LSTM for title processing
- ✓ Self-attention over LSTM outputs
- ✓ Global attention over CNN features
- ✓ Weighted attention fusion between title and abstract
- ✓ Variational dropout + batch normalization
- ✓ Attention visualization capability
- ✓ Pure PyTorch implementation (no Transformers/BERT)

## Dataset

**Source:** `arxiv_papers_raw.csv`
**Total Samples:** 12,000 papers

**Categories (3-class):**
- cs.AI-LG: 6,000 papers (merged cs.AI + cs.LG)
- cs.CL: 3,000 papers (Computational Linguistics)
- cs.CV: 3,000 papers (Computer Vision)

**Splits:**
- Train: 70% (8,399 samples)
- Validation: 15% (1,801 samples)
- Test: 15% (1,800 samples)

**Class Imbalance:** 2:1 ratio (cs.AI-LG has 2x more samples than cs.CL/cs.CV)

**Why 3-Class?** cs.AI papers had severe performance issues (20-27% recall) in 4-class models due to generic vocabulary overlap with cs.LG. Merging improved accuracy from 33.94% → 59.00% (+25pp).

## Current Architecture (Optimized)

### Overview

**Model:** HybridCNNLSTM
**Parameters:** ~10M (optimized from 13.3M baseline)
**Embedding:** GloVe 300d (53.8% vocabulary coverage)

### Component Details

**Embeddings:**
- GloVe 300d pre-trained embeddings
- Trainable for domain adaptation
- Uniform dropout: 0.6

**Title Processing (LSTM + Multi-Head Self-Attention):**
- Bidirectional LSTM (2 layers, 128 hidden units)
- Multi-head self-attention (4 heads) over LSTM outputs
  - Query, Key, Value projections
  - Scaled dot-product attention per head
  - Output projection with residual connection
  - Layer normalization
- Mean pooling across sequence
- Output: 256-dimensional title representation

**Abstract Processing (Residual CNN + Multi-Head Global Attention):**
- Three residual CNN blocks with kernel sizes [3, 4, 5]
  - Each block: Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm
  - Skip connections for gradient flow
  - 128 filters per kernel
  - Dropout: 0.3 within blocks
- Concatenated CNN features: 384 dimensions
- Multi-head global attention (4 heads) over CNN features
  - Attention weights computed per head
  - Weighted sum of features per head
  - Combined head outputs with projection
  - Layer normalization with residual
- Output: 384-dimensional abstract representation

**Fusion (Gated Mechanism):**
- Input: Title (256d) + Abstract (384d) = 640d
- Title gate: Linear(640→256) → Sigmoid
- Abstract gate: Linear(640→384) → Sigmoid
- Gated outputs: title * title_gate + abstract * abstract_gate
- Layer normalization
- Output: 640-dimensional fused representation

**Classifier:**
- LayerNorm(640) → Dropout(0.6) → Linear(640→256) → ReLU
- LayerNorm(256) → Dropout(0.6) → Linear(256→3)
- CrossEntropyLoss with class weights and label smoothing

### Key Improvements Over Baseline

| Component | Baseline | Optimized | Impact |
|-----------|----------|-----------|--------|
| **LSTM Attention** | Single-head | Multi-head (4) | Better semantic pattern capture |
| **CNN Attention** | Single-head | Multi-head (4) | Diverse feature importance |
| **CNN Architecture** | Vanilla | Residual blocks | Improved gradient flow |
| **Fusion** | Weighted sum | Gated mechanism | Dynamic importance learning |
| **Dropout** | Inconsistent (0.4/0.3/0.5) | Uniform (0.6) | Stronger regularization |
| **Model Capacity** | 160 filters (13.3M) | 128 filters (10M) | Reduced overfitting |
| **Class Weights** | [1.0, 1.0, 1.8] | [1.0, 2.0, 1.8] | Fixed cs.CL underperformance |
| **Weight Decay** | 5e-4 | 1e-3 | Stronger L2 regularization |
| **Initialization** | Standard | Xavier + Orthogonal | Better gradient propagation |

## Baseline Performance Analysis

### Previous Implementation (59.33% accuracy)

```
              precision    recall  f1-score   support
  cs.AI-LG     62.46%    66.00%    64.18%       900
     cs.CL     51.09%    41.78%    45.97%       450  ← CRITICAL
     cs.CV     59.46%    63.56%    61.44%       450
```

**Critical Issues:**

1. **Catastrophic Overfitting:**
   - Epoch 3: Train 66.31%, Val 63.96% (gap: 2.34%) ✓ healthy
   - Epoch 9: Train 86.24%, Val 59.30% (gap: 26.94%) ✗ collapsed
   - Model memorizes instead of learning

2. **cs.CL Collapse:**
   - Recall: 41.78% (vs 80% in earlier version)
   - 262 out of 450 cs.CL papers misclassified → cs.AI-LG
   - Root cause: Class weight 1.0 despite having 50% less training data

3. **Poor Embedding Coverage:**
   - GloVe (2014): 53.8% vocabulary coverage
   - 46.2% words randomly initialized
   - Missing: "BERT", "GPT", "transformer", "LLM", "ResNet" (post-2014 terms)

## Optimization Strategy

### Problem → Solution Mapping

| Problem | Root Cause | Solution | Expected Impact |
|---------|------------|----------|-----------------|
| Overfitting (27% gap) | Excessive capacity (13.3M params) | Reduce to 10M, dropout 0.6, weight decay 1e-3 | Gap <10% |
| cs.CL low recall (42%) | Class weight 1.0 despite 2:1 imbalance | Increase class weight to 2.0 | +20-30pp recall |
| Limited attention | Single-head captures limited patterns | Multi-head (4) for diversity | +2-3pp accuracy |
| Vanilla CNN | No gradient shortcuts | Residual connections | Better training stability |
| Simple fusion | Fixed weighted sum | Gated mechanism | +1-2pp accuracy |

### Expected Performance (Optimized)

**Overall Accuracy:** 65-70% (+6-11pp improvement)

**Per-Class Predictions:**
- **cs.AI-LG:** 65-68% recall (stable, slight improvement)
- **cs.CL:** 60-70% recall (+20-30pp from 42%) ← Major fix
- **cs.CV:** 62-68% recall (maintains ~64%)

**Training Behavior:**
- Train/Val gap: <10% (vs 27% baseline)
- Early stopping: epoch 10-15 (vs epoch 3 baseline)
- Best val accuracy: 68-72%

## Hyperparameters

```python
# Training
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
PATIENCE = 7

# Regularization
DROPOUT = 0.6
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1

# Class Balancing
CLASS_WEIGHTS = [1.0, 2.0, 1.8]  # [cs.AI-LG, cs.CL, cs.CV]

# Model Architecture
NUM_FILTERS = 128
LSTM_HIDDEN = 128
KERNEL_SIZES = [3, 4, 5]
NUM_ATTENTION_HEADS = 4

# Sequence Lengths
MAX_ABSTRACT_LEN = 300
MAX_TITLE_LEN = 30
```

## File Structure

```
Proyecto-de-Deep-Learning/
├── Hybrid_CNN_LSTM_Colab.ipynb   # Main training notebook (optimized architecture)
├── README.md                      # User-facing documentation
├── CLAUDE.md                      # This file (for Claude context)
├── arxiv_papers_raw.csv          # Dataset (upload to Drive)
└── glove.6B.300d.txt             # GloVe embeddings (upload to Drive)
```

## How to Use

### Google Colab Execution

1. **Upload Files to Google Drive:**
   - Path: `/content/drive/MyDrive/ArXiv_Project/`
   - Files: `arxiv_papers_raw.csv` (dataset), `glove.6B.300d.txt` (~822MB)

2. **Run Notebook:**
   - Open via Colab badge (cell 0)
   - Mount Google Drive (cell 5)
   - Execute cells sequentially (0-22)
   - Training: ~8-12 minutes on GPU

3. **Expected Output:**
   - Best validation accuracy: ~68-72%
   - Test accuracy: ~65-70%
   - Model saved: `hybrid_model.pth`
   - Confusion matrix: `confusion_matrix.png`

## Architecture Compliance

All optimizations maintain 100% compliance with project requirements:

- [x] **CNN 1D for abstracts** (with residual blocks)
- [x] **Bidirectional LSTM for titles**
- [x] **Self-attention over LSTM** (multi-head variant)
- [x] **Global attention over CNN** (multi-head variant)
- [x] **Weighted fusion** (gated variant)
- [x] **Variational dropout** (uniform 0.6)
- [x] **Batch/Layer normalization**
- [x] **Attention visualization** (attention_maps returned)
- [x] **Pure PyTorch** (no Transformers)

**Note:** Multi-head attention and gated fusion are advanced implementations that enhance the original requirements while maintaining their core intent.

## Known Limitations

1. **GloVe Embeddings:**
   - From 2014, only 53.8% vocabulary coverage
   - Future: Train embeddings from scratch or use subword models

2. **3-Class Taxonomy:**
   - cs.AI-LG merge loses granularity
   - Necessary due to vocabulary overlap

3. **Class Imbalance:**
   - 2:1 ratio persists
   - Addressed with weights, but data augmentation could help further

## Git Workflow

**Branch:** `claude/improve-implementation-018rMkv8JP1bb2KNiHNbvF1o`
**Origin:** `fernandoramirez1337/Proyecto-de-Deep-Learning`

**Latest Commit:**
```
commit 0b931a9
Implement optimized architecture for maximum performance

Major improvements:
- Multi-head self-attention (4 heads) over LSTM outputs
- Multi-head global attention (4 heads) over CNN features
- Residual CNN blocks with skip connections
- Gated fusion mechanism with layer normalization
- Improved LSTM initialization (Xavier + Orthogonal)

Hyperparameter optimizations:
- Dropout: 0.5 → 0.6 (stronger regularization)
- Model capacity: 160 → 128 (reduce overfitting)
- Class weights: [1.0, 1.0, 1.8] → [1.0, 2.0, 1.8] (fix cs.CL)
- Weight decay: 5e-4 → 1e-3 (stronger L2)

Expected: 65-70% test accuracy (vs 59.33% baseline)
```

## Next Steps

1. **Execute Training:** Run notebook in Colab and collect results
2. **Evaluate Performance:**
   - If 65-70%: Success, document and finalize
   - If 60-65%: Good improvement, consider minor tweaks
   - If <60%: Investigate further (embeddings from scratch, data augmentation)

3. **Further Improvements (if needed):**
   - Train embeddings from scratch on arXiv corpus
   - Implement data augmentation (EDA, back-translation)
   - Experiment with focal loss (γ=2.5)
   - Model ensemble

## Important Notes

1. **Project Compliance:** All optimizations maintain 100% project definition compliance
2. **Not a Transformer:** Despite multi-head attention, this is NOT a Transformer architecture
3. **Pure PyTorch:** All implementations use standard PyTorch modules
4. **Reproducibility:** SEED=42 for deterministic results

## Session Context

**Last Modified:** 2025-11-19
**Status:** Optimized architecture implemented, committed, and pushed
**Current Task:** Ready for testing in Google Colab

**Recent Implementation:**
- Multi-head self-attention (4 heads) for LSTM
- Multi-head global attention (4 heads) for CNN
- Residual CNN blocks with skip connections
- Gated fusion with learnable gates
- Optimized hyperparameters (dropout 0.6, class weights [1.0, 2.0, 1.8], weight decay 1e-3)
- Reduced model capacity (128 filters, ~10M params)
- Better initialization (Xavier + Orthogonal for LSTM)

**Expected Results:**
- Test accuracy: 65-70%
- cs.CL recall: 60-70% (major improvement from 42%)
- Reduced overfitting: train/val gap <10%
