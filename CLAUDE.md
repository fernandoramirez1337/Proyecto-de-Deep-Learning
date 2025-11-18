# CLAUDE.md - AI Assistant Guide

**Last Updated:** 2025-11-18
**Project:** ArXiv Papers Classification using SciBERT

This document provides comprehensive guidance for AI assistants (like Claude) working with this codebase. It covers structure, conventions, workflows, and important context.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Key Files Reference](#key-files-reference)
4. [Development Workflow](#development-workflow)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Coding Conventions](#coding-conventions)
8. [Hardware Considerations](#hardware-considerations)
9. [Common Tasks](#common-tasks)
10. [Version History](#version-history)
11. [Important Constants](#important-constants)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose
Multi-class classification of scientific papers from ArXiv into 4 Computer Science categories using SciBERT transformer model.

### Categories
1. **cs.AI** - Artificial Intelligence (minority class, main challenge)
2. **cs.CL** - Computation and Language
3. **cs.CV** - Computer Vision
4. **cs.LG** - Machine Learning

### Project Goals
- **Test Accuracy:** ≥ 60% (Current: 56.17%, gap: -3.83%)
- **cs.AI Recall:** > 30% (Current: 36.22%, **ACHIEVED** ✓)
- **Overfitting Gap:** < 10%

### Dataset
- **Size:** 12,000 ArXiv papers
- **Location:** `data/arxiv_papers_raw.csv`
- **Split:** 70% train, 15% validation, 15% test
- **Features:** Title + Abstract (text)
- **Format:** CSV with columns: `title`, `abstract`, `category`

### Current Solution
**Model V3.7 + Threshold Tuning (threshold=0.40)**
- Uses class weighting (cs.AI x2.0) during training
- Applies threshold tuning at inference (threshold=0.40 for cs.AI)
- Best gap total: 3.83%
- Successfully meets cs.AI recall objective

---

## Repository Structure

```
Proyecto-de-Deep-Learning/
├── Core Training Files
│   ├── train_scibert_optimized.py      # Main training script (V3.8 config)
│   ├── model_scibert.py                # Model architecture definitions
│   ├── preprocessing_scibert.py        # Data loading and preprocessing
│   └── threshold_tuning.py             # Post-training threshold optimization
│
├── Inference
│   └── predict_optimized.py            # Production inference with threshold tuning
│
├── Scripts
│   ├── scripts/compare_models.py       # Compare model versions
│   ├── scripts/eda.py                  # Exploratory data analysis
│   ├── scripts/download_data.py        # Download ArXiv dataset
│   └── scripts/test_pipeline.py        # Pipeline testing (legacy)
│
├── Backups
│   ├── backups/train_scibert_v2_backup.py    # V2: Over-regularized
│   ├── backups/train_scibert_v3_backup.py    # V3: Under-regularized
│   ├── backups/train_scibert_v3.5_backup.py  # V3.5: Failed midpoint
│   ├── backups/train_scibert_v3.6_backup.py  # V3.6: Aggressive weighting
│   └── backups/train_scibert_v3.7_backup.py  # V3.7: Best base model
│
├── Configuration
│   ├── train_m2_optimized.sh           # Training shell script with env vars
│   └── .gitignore                      # Excludes models, data, artifacts
│
├── Documentation
│   ├── README.md                       # User-facing documentation
│   ├── SOLUTION_FINAL.md              # Detailed solution documentation
│   └── CLAUDE.md                      # This file (AI assistant guide)
│
└── Artifacts (gitignored)
    ├── best_scibert_v3.7_final.pth    # Final model checkpoint (1.1GB)
    ├── scibert_label_encoder.pkl      # Label encoder for categories
    ├── data/arxiv_papers_raw.csv      # Dataset
    └── *.png                          # Training plots and confusion matrices
```

### Directory Purpose

- **Root:** Core training, inference, and data processing files
- **scripts/:** Utility scripts for analysis and testing
- **backups/:** Historical versions of training scripts for reference
- **data/:** Dataset files (gitignored, must be downloaded)
- **results/:** Training outputs, plots, metrics (gitignored)

---

## Key Files Reference

### 1. `train_scibert_optimized.py` (Main Training Script)

**Purpose:** Complete training pipeline with class weighting and regularization

**Key Components:**
- `OptimizedSciBERTClassifier`: Model with configurable dropout and layer freezing
- `OptimizedTrainer`: Training loop with early stopping, scheduler, metrics
- `main()`: Entry point with hyperparameter configuration

**Current Configuration (V3.8):**
```python
FREEZE_BERT_LAYERS = 3      # 9 layers unfrozen
DROPOUT = 0.35              # Moderate regularization
BATCH_SIZE = 12             # M2 optimized
LR = 5e-5                   # Learning rate
WEIGHT_DECAY = 0.01         # L2 regularization
CLASS_WEIGHTS = [2.3, 1.0, 1.0, 1.0]  # cs.AI emphasis
PATIENCE = 3                # Early stopping
```

**Output Files:**
- `best_scibert_optimized.pth`: Best model checkpoint
- `scibert_optimized_history.png`: Training curves
- `scibert_optimized_confusion.png`: Confusion matrix

**Usage:**
```bash
./train_m2_optimized.sh
# OR
python train_scibert_optimized.py
```

### 2. `model_scibert.py` (Model Architectures)

**Classes:**
1. **SciBERTClassifier** (Base model, not currently used)
   - Dual-encoder architecture
   - Separate processing for title and abstract
   - Attention pooling for each modality

2. **LightSciBERTClassifier** (Lightweight alternative)
   - Single-encoder architecture
   - Concatenates title + abstract
   - Uses [CLS] token for classification

**Note:** Current training uses `OptimizedSciBERTClassifier` from `train_scibert_optimized.py`, which is an enhanced version with:
- Configurable layer freezing
- Embedding dropout
- GELU activation (better for transformers)
- 3-layer fusion network (512→256→128→4)

### 3. `preprocessing_scibert.py` (Data Pipeline)

**Key Classes:**
- `SciBERTDataset`: Dual-encoder dataset (title + abstract separate)
- `SciBERTLightDataset`: Single-encoder dataset (combined text)

**Function:**
- `prepare_scibert_data()`: Main data preparation function
  - Loads CSV from `data/arxiv_papers_raw.csv`
  - Creates train/val/test splits (70/15/15)
  - Tokenizes using SciBERT tokenizer
  - Saves label encoder to `scibert_label_encoder.pkl`

**Tokenization Lengths:**
- Title: max_length=32
- Abstract: max_length=128
- Combined: max_length=160 (light model)

### 4. `predict_optimized.py` (Inference)

**Purpose:** Production-ready inference with threshold tuning

**Class:** `OptimizedPredictor`
- Loads trained model and label encoder
- Applies threshold=0.40 for cs.AI detection
- Supports single and batch prediction
- Returns predictions with optional probabilities

**Key Innovation - Threshold Tuning:**
```python
# Standard argmax (implicit threshold=0.5)
prediction = argmax(probabilities)

# Threshold tuning (threshold=0.40)
if probability[cs.AI] >= 0.40:
    prediction = cs.AI
else:
    prediction = argmax(probabilities)
```

**Usage:**
```python
from predict_optimized import OptimizedPredictor

predictor = OptimizedPredictor(threshold_cs_ai=0.40)
category = predictor.predict(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture..."
)
```

### 5. `threshold_tuning.py` (Post-Training Optimization)

**Purpose:** Find optimal threshold for cs.AI class

**Process:**
1. Load trained model V3.7
2. Test thresholds from 0.30 to 0.50
3. Evaluate metrics for each threshold
4. Select threshold with best gap total

**Results:**
- threshold=0.50 (default): 57.39% acc, 28.22% cs.AI recall
- threshold=0.40 (optimal): 56.17% acc, 36.22% cs.AI recall
- Trade-off: -1.22% accuracy for +8.00% cs.AI recall

### 6. `train_m2_optimized.sh` (Environment Setup)

**Purpose:** Configure environment variables for M2 MacBook Air training

**Key Variables:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1      # Enable CPU fallback for unsupported ops
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
TOKENIZERS_PARALLELISM=false       # Avoid multiprocessing issues
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
PYTORCH_MPS_PREFER_METAL=1
```

---

## Development Workflow

### Standard Training Workflow

1. **Prepare Data**
   ```bash
   # Ensure data/arxiv_papers_raw.csv exists
   python scripts/download_data.py  # If needed
   ```

2. **Configure Hyperparameters**
   - Edit `train_scibert_optimized.py` main() function
   - Update FREEZE_BERT_LAYERS, DROPOUT, CLASS_WEIGHTS, etc.
   - Update version comment and echo in `train_m2_optimized.sh`

3. **Run Training**
   ```bash
   ./train_m2_optimized.sh
   ```

4. **Monitor Progress**
   - Watch epoch metrics: Train Acc, Val Acc, F1, Gap
   - Early stopping triggers after 3 epochs without improvement
   - Best model saved to `best_scibert_optimized.pth`

5. **Evaluate Results**
   - Review test metrics printed at end
   - Check classification report for per-class metrics
   - Examine confusion matrix plot
   - Verify cs.AI recall > 30% objective

6. **Apply Threshold Tuning (Optional)**
   ```bash
   python threshold_tuning.py
   ```

7. **Backup Version**
   ```bash
   cp train_scibert_optimized.py backups/train_scibert_v3.X_backup.py
   ```

### Experiment Tracking

**Version Naming Convention:**
- V2, V3, V3.5, V3.6, V3.7, V3.8, etc.
- Use incremental version numbers
- Document strategy in comments

**What to Track:**
- Hyperparameters (freeze layers, dropout, LR, weights)
- Metrics (test acc, cs.AI recall, gap total)
- Strategy description
- Issues observed

**Where to Document:**
- Code comments in training script
- Echo statements in shell script
- SOLUTION_FINAL.md for major findings

---

## Model Architecture

### OptimizedSciBERTClassifier

**Base Model:** SciBERT (`allenai/scibert_scivocab_uncased`)
- 12 transformer layers
- 768 hidden dimensions
- Vocabulary: ~31K tokens specialized for scientific text

**Architecture Flow:**

```
Input: Title (max 32 tokens) + Abstract (max 128 tokens)
    ↓
[SciBERT Encoder] (same weights, separate passes)
    ↓
Title: [batch, 32, 768]    Abstract: [batch, 128, 768]
    ↓                           ↓
[Attention Pooling]       [Attention Pooling]
    ↓                           ↓
[batch, 768]              [batch, 768]
    ↓─────────────────────────────↓
           [Concatenate]
                ↓
          [batch, 1536]
                ↓
         [Fusion Network]
         512 → 256 → 128 → 4
                ↓
         [batch, 4] logits
```

**Key Features:**

1. **Layer Freezing** (FREEZE_BERT_LAYERS=3)
   - First 3 layers frozen (general language features)
   - Last 9 layers trainable (task-specific features)
   - Reduces overfitting and training time

2. **Attention Pooling**
   - Learned attention weights over sequence
   - Better than [CLS] or mean pooling
   - Returns pooled vector + attention weights

3. **Fusion Network**
   - 3 hidden layers with LayerNorm
   - GELU activation (standard for transformers)
   - Progressive dropout (0.35 → 0.35 → 0.28)

4. **Regularization Techniques**
   - Embedding dropout (0.1)
   - Layer dropout (0.35)
   - Label smoothing (0.1)
   - Weight decay (0.01)
   - Gradient clipping (1.0)
   - Early stopping (patience=3)

---

## Training Pipeline

### Data Flow

```
arxiv_papers_raw.csv
    ↓
[prepare_scibert_data()]
    ↓
LabelEncoder: category → {0,1,2,3}
    ↓
Train/Val/Test Split (70/15/15, stratified)
    ↓
SciBERTDataset (tokenization)
    ↓
DataLoader (batch_size=12, shuffle=True)
    ↓
[OptimizedTrainer.train()]
```

### Training Loop

```python
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        outputs = model(title_ids, title_mask, abstract_ids, abstract_mask)
        loss = criterion(outputs, labels)  # CrossEntropy + label_smoothing + class_weights
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Validation phase
    model.eval()
    for batch in val_loader:
        outputs = model(...)
        # Calculate accuracy, F1

    # Early stopping check
    if val_acc > best_val_acc:
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break
```

### Loss Function

```python
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,           # Soften class boundaries
    weight=class_weights           # [2.0, 1.0, 1.0, 1.0] for cs.AI emphasis
)
```

### Optimizer Configuration

```python
# Differential learning rates
optimizer = AdamW([
    {'params': bert_params, 'lr': 5e-5, 'weight_decay': 0.01},
    {'params': classifier_params, 'lr': 2.5e-4, 'weight_decay': 0.02}
])
```

**Rationale:**
- BERT layers: Lower LR (5e-5) to preserve pre-trained knowledge
- Classifier layers: Higher LR (5x) for faster adaptation
- Higher weight decay on classifier (2x) to prevent overfitting

### Learning Rate Schedule

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps
)
```

**Schedule:**
- 10% warmup (gradual increase)
- 90% linear decay to zero

---

## Coding Conventions

### Python Style

1. **Imports:** Standard library → Third-party → Local
   ```python
   import torch
   import torch.nn as nn
   from transformers import AutoTokenizer
   from preprocessing_scibert import prepare_scibert_data
   ```

2. **Docstrings:** Use for classes and main functions
   ```python
   def prepare_scibert_data(use_light_model=False):
       """
       Prepara datos para modelos SciBERT

       Args:
           use_light_model: Si True, usa modelo ligero

       Returns:
           train_dataset, val_dataset, test_dataset, tokenizer, label_encoder
       """
   ```

3. **Comments:** Mix of English and Spanish
   - Technical terms: English (dropout, pooling, attention)
   - Explanations: Spanish
   - Keep existing style when editing

4. **Variable Naming:**
   - Classes: PascalCase (`OptimizedSciBERTClassifier`)
   - Functions: snake_case (`prepare_scibert_data`)
   - Constants: UPPER_SNAKE_CASE (`FREEZE_BERT_LAYERS`)
   - Private: Leading underscore (`_init_weights`)

### Model Development

1. **Always inherit from nn.Module**
   ```python
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()

       def forward(self, x):
           return output
   ```

2. **Use LayerNorm over BatchNorm** for transformers

3. **Prefer GELU over ReLU** for transformer-based models

4. **Initialize custom layers** with normal distribution
   ```python
   module.weight.data.normal_(mean=0.0, std=0.02)
   module.bias.data.zero_()
   ```

### Training Script Structure

```python
# 1. Imports
import torch
...

# 2. Model class
class OptimizedSciBERTClassifier(nn.Module):
    ...

# 3. Trainer class
class OptimizedTrainer:
    ...

# 4. Utility functions
def compute_class_weights_from_dataset():
    ...

# 5. Main function
def main():
    # Configuration
    FREEZE_BERT_LAYERS = 3
    ...

    # Setup device
    device = torch.device(...)

    # Prepare data
    train_dataset, val_dataset, test_dataset, tokenizer, le = prepare_scibert_data()

    # Create model and trainer
    model = OptimizedSciBERTClassifier(...)
    trainer = OptimizedTrainer(...)

    # Train
    trainer.train(...)

    # Evaluate
    ...

# 6. Entry point
if __name__ == "__main__":
    main()
```

---

## Hardware Considerations

### M2 MacBook Air Optimizations

**Why M2-specific configuration?**
- This project was developed on M2 MacBook Air
- MPS (Metal Performance Shaders) backend has limitations
- Memory constraints require smaller batch sizes

**Key Optimizations:**

1. **Device Selection**
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **MPS Fallback** (Critical!)
   ```python
   import os
   os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
   ```
   Without this, some operations crash on MPS

3. **DataLoader Configuration**
   ```python
   DataLoader(
       dataset,
       batch_size=12,          # Reduced for M2 (32 on CUDA)
       num_workers=0,          # MPS doesn't support multiprocessing
       pin_memory=False,       # Not supported on MPS
       persistent_workers=False
   )
   ```

4. **Batch Size Constraints**
   - M2: 12 (maximum stable)
   - T4 GPU: 32
   - A100 GPU: 64+

**Performance Expectations:**
- M2 is ~2-3x slower than T4 GPU
- Total training time: ~60-80 min per version (10 epochs)
- 8 versions total: ~10-12 hours
- Viable for development and experimentation

### Adapting to Different Hardware

**For CUDA GPUs:**
```python
# Increase batch size
BATCH_SIZE = 32  # or 64

# Enable multiprocessing
NUM_WORKERS = 4

# Enable pinned memory
DataLoader(..., num_workers=4, pin_memory=True)
```

**For CPU (not recommended):**
```python
# Further reduce batch size
BATCH_SIZE = 8

# Training will be very slow (~10x slower)
```

---

## Common Tasks

### Task 1: Train a New Model Version

```bash
# 1. Edit hyperparameters in train_scibert_optimized.py
vim train_scibert_optimized.py
# Modify FREEZE_BERT_LAYERS, DROPOUT, CLASS_WEIGHTS, etc.

# 2. Update version info in train_m2_optimized.sh
vim train_m2_optimized.sh
# Update echo statement with version number and strategy

# 3. Run training
./train_m2_optimized.sh

# 4. Backup the version
cp train_scibert_optimized.py backups/train_scibert_v3.X_backup.py

# 5. Document results in SOLUTION_FINAL.md
```

### Task 2: Make Predictions on New Papers

```python
from predict_optimized import OptimizedPredictor

# Initialize predictor
predictor = OptimizedPredictor(threshold_cs_ai=0.40)

# Single prediction
category = predictor.predict(
    title="Your paper title",
    abstract="Your paper abstract..."
)

# With probabilities
category, probs = predictor.predict(
    title="Your paper title",
    abstract="Your paper abstract...",
    return_probs=True
)

# Batch predictions
papers = [
    {'title': 'Title 1', 'abstract': 'Abstract 1'},
    {'title': 'Title 2', 'abstract': 'Abstract 2'}
]
predictions = predictor.predict_batch(papers)
```

### Task 3: Tune Threshold for Different Use Cases

```python
# Conservative (high precision, low recall)
predictor = OptimizedPredictor(threshold_cs_ai=0.45)

# Balanced (current optimal)
predictor = OptimizedPredictor(threshold_cs_ai=0.40)

# Aggressive (low precision, high recall)
predictor = OptimizedPredictor(threshold_cs_ai=0.35)
```

### Task 4: Analyze Model Performance

```python
# After training, examine outputs:
# 1. Training curves: scibert_optimized_history.png
# 2. Confusion matrix: scibert_optimized_confusion.png
# 3. Classification report: printed to console

# Key metrics to check:
# - Test accuracy vs 60% target
# - cs.AI recall vs 30% target
# - Overfitting gap (train_acc - val_acc)
# - Per-class precision and recall
```

### Task 5: Compare Model Versions

```python
# Use compare_models.py script
python scripts/compare_models.py

# Manual comparison:
# 1. Check SOLUTION_FINAL.md for version comparison table
# 2. Load multiple checkpoints and evaluate
```

### Task 6: Debug Training Issues

**Overfitting (train >> val):**
- Increase dropout
- Increase weight decay
- Freeze more BERT layers
- Add label smoothing
- Reduce class weights

**Underfitting (both low):**
- Decrease dropout
- Unfreeze more BERT layers
- Increase model capacity
- Train longer (more epochs)

**cs.AI Ignored:**
- Increase class weight for cs.AI
- Apply threshold tuning
- Check class distribution in data

**Accuracy Collapse:**
- Reduce class weights (probably too high)
- Check for NaN gradients
- Verify data loading correctness

---

## Version History

### Evolution Summary (8 Versions)

| Version | Strategy | Freeze | Dropout | CS.AI Weight | Test Acc | CS.AI Recall | Gap |
|---------|----------|--------|---------|--------------|----------|--------------|-----|
| V2 | Over-regularized | 8 | 0.5 | - | 59.17% | 13.78% | 19.05% |
| V3 | Under-regularized | 3 | 0.35 | - | 55.28% | 26.22% | 8.50% |
| V3.5 | Midpoint (failed) | 5-6 | 0.42 | - | 58.50% | 2.22% | 29.28% |
| V3.6 | Aggressive weight | 3 | 0.35 | 3.0 | 49.72% | 51.11% | 10.28% |
| V3.7 | Balanced weight | 3 | 0.35 | 2.0 | 57.39% | 28.22% | 4.39% |
| V3.7+TT | Threshold=0.40 | 3 | 0.35 | 2.0 | **56.17%** | **36.22%** | **3.83%** |
| V3.8 | Fine-tuned weight | 3 | 0.35 | 2.3 | 49.61% | 39.78% | 10.39% |

**Final Solution:** V3.7 + Threshold Tuning (threshold=0.40)
- **Objectives:** 1/2 met (cs.AI recall ✓, accuracy close)
- **Best gap total:** 3.83%

### Key Learnings

1. **Threshold Tuning > Aggressive Weighting**
   - V3.7+TT (gap 3.83%) better than V3.8 (gap 10.39%)
   - No retraining required
   - Adjustable at inference time

2. **Class Weighting is Non-Linear**
   - x2.0: Optimal balance
   - x2.3: Accuracy collapse (-7.78%)
   - Sweet spot: 2.0-2.15

3. **"Midpoint" Strategy Failed**
   - V3.5 averaging V2 and V3 hyperparameters
   - Worst cs.AI recall (2.22%)
   - Non-linear hyperparameter relationships

4. **Layer Freezing Impact**
   - Freeze 8 layers: High accuracy, ignores cs.AI
   - Freeze 3 layers: Lower accuracy, better cs.AI detection
   - Trade-off between generalization and task-specific learning

---

## Important Constants

### File Paths
```python
DATA_PATH = 'data/arxiv_papers_raw.csv'
MODEL_PATH = 'best_scibert_optimized.pth'
FINAL_MODEL_PATH = 'best_scibert_v3.7_final.pth'
LABEL_ENCODER_PATH = 'scibert_label_encoder.pkl'
```

### Model Configuration
```python
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
NUM_CLASSES = 4
HIDDEN_SIZE = 768  # SciBERT hidden dimension
```

### Optimal Hyperparameters (V3.7)
```python
FREEZE_BERT_LAYERS = 3       # First 3 layers frozen
DROPOUT = 0.35               # Fusion network dropout
BATCH_SIZE = 12              # M2 optimized
EPOCHS = 10                  # Maximum epochs
LR = 5e-5                    # BERT learning rate
WEIGHT_DECAY = 0.01          # L2 regularization
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI, cs.CL, cs.CV, cs.LG
PATIENCE = 3                 # Early stopping patience
THRESHOLD_CS_AI = 0.40       # Inference threshold
```

### Tokenization
```python
MAX_TITLE_LEN = 32           # Title max tokens (dual model)
MAX_ABSTRACT_LEN = 128       # Abstract max tokens (dual model)
MAX_COMBINED_LEN = 160       # Combined max tokens (light model)
```

### Data Splits
```python
TEST_SIZE = 0.15             # 15% test
VAL_SIZE = 0.15              # 15% validation
TRAIN_SIZE = 0.70            # 70% training
RANDOM_STATE = 42            # Reproducibility
STRATIFY = True              # Balanced class distribution
```

### Category Mapping
```python
# LabelEncoder creates this mapping:
CATEGORY_TO_IDX = {
    'cs.AI': 0,
    'cs.CL': 1,
    'cs.CV': 2,
    'cs.LG': 3
}
```

---

## Troubleshooting

### Common Issues

#### 1. MPS/CUDA Out of Memory
**Symptoms:** RuntimeError: MPS backend out of memory

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 8  # or even 4

# Enable MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Clear cache between runs
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

#### 2. Model Not Loading
**Symptoms:** FileNotFoundError or KeyError when loading checkpoint

**Solutions:**
```python
# Check model path
assert os.path.exists('best_scibert_v3.7_final.pth')

# Load with map_location
checkpoint = torch.load(model_path, map_location=device)

# Match architecture
model = OptimizedSciBERTClassifier(
    num_classes=4,
    dropout=0.35,        # Must match training config
    freeze_bert_layers=3
)
```

#### 3. Data Not Found
**Symptoms:** FileNotFoundError: data/arxiv_papers_raw.csv

**Solutions:**
```bash
# Download dataset
python scripts/download_data.py

# OR ensure data directory exists
mkdir -p data
# Place arxiv_papers_raw.csv in data/
```

#### 4. Slow Training on M2
**Symptoms:** Training takes hours per epoch

**Solutions:**
- Expected behavior: M2 is 2-3x slower than T4 GPU
- Reduce batch size won't help (already optimized at 12)
- Consider using Google Colab with T4 GPU for faster iteration
- Use early stopping (patience=3) to avoid wasted epochs

#### 5. Poor cs.AI Recall
**Symptoms:** cs.AI recall < 20%

**Solutions:**
```python
# Increase class weight
CLASS_WEIGHTS = [2.5, 1.0, 1.0, 1.0]  # Try 2.5 instead of 2.0

# Apply threshold tuning
predictor = OptimizedPredictor(threshold_cs_ai=0.35)  # Lower threshold

# Unfreeze more layers
FREEZE_BERT_LAYERS = 2  # or 1
```

#### 6. High Overfitting Gap
**Symptoms:** train_acc - val_acc > 15%

**Solutions:**
```python
# Increase regularization
DROPOUT = 0.4           # Was 0.35
WEIGHT_DECAY = 0.02     # Was 0.01

# Freeze more layers
FREEZE_BERT_LAYERS = 4  # Was 3

# Reduce class weights
CLASS_WEIGHTS = [1.5, 1.0, 1.0, 1.0]  # Was 2.0
```

#### 7. NaN Loss
**Symptoms:** Loss becomes NaN during training

**Solutions:**
```python
# Reduce learning rate
LR = 3e-5  # Was 5e-5

# Check for inf/nan in data
assert not df.isnull().any().any()

# Enable gradient clipping (already enabled at 1.0)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Reduce class weights
CLASS_WEIGHTS = [1.5, 1.0, 1.0, 1.0]  # Very high weights can cause instability
```

---

## Best Practices for AI Assistants

### When Modifying Code

1. **Read existing version first** before making changes
2. **Preserve coding style** (mixed English/Spanish comments)
3. **Update version numbers** in comments and scripts
4. **Backup before major changes** to backups/ directory
5. **Document changes** in comments and commit messages

### When Analyzing Results

1. **Check all three metrics:** accuracy, cs.AI recall, gap
2. **Compare to objectives:** 60% accuracy, 30% cs.AI recall
3. **Examine confusion matrix** for class-specific issues
4. **Look at training curves** for overfitting signs
5. **Consider trade-offs** between metrics

### When Suggesting Improvements

1. **Reference version history** to avoid repeating failed experiments
2. **Justify hyperparameter changes** based on observed issues
3. **Consider hardware constraints** (M2 limitations)
4. **Propose incremental changes** rather than drastic overhauls
5. **Estimate training time impact** (~60-80 min per version on M2)

### When Explaining Code

1. **Reference this document** for standard explanations
2. **Link to relevant sections** of key files
3. **Include line numbers** when referring to specific code: `train_scibert_optimized.py:364`
4. **Explain context** from version history when relevant
5. **Highlight M2-specific optimizations** if relevant to question

---

## Additional Resources

### External Documentation
- [SciBERT Paper](https://arxiv.org/abs/1903.10676)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

### Internal Documentation
- `README.md`: User-facing quick start and results
- `SOLUTION_FINAL.md`: Complete solution documentation with detailed analysis
- `backups/`: Historical training scripts for reference

### Key External Dependencies
```
Python 3.8+
torch>=2.0              # PyTorch with MPS support
transformers            # Hugging Face transformers
scikit-learn           # Metrics, preprocessing
pandas                 # Data manipulation
numpy                  # Numerical operations
matplotlib             # Plotting
seaborn                # Enhanced plotting
tqdm                   # Progress bars
```

---

## Conclusion

This codebase represents an iterative deep learning project focused on multi-class text classification with class imbalance. The key insight is that **threshold tuning** (post-training optimization) can outperform aggressive fine-tuning techniques while being more efficient.

**Core Strengths:**
- Well-documented version history
- Hardware-optimized for M2 (portable, no GPU required)
- Production-ready inference pipeline
- Clear separation of concerns (data/model/training/inference)

**Limitations:**
- Accuracy slightly below 60% target (56.17%)
- Threshold tuning is class-specific (only cs.AI)
- M2 training is slow (~60-80 min per version)

**Future Directions:**
- Focal Loss for better class balance
- Data augmentation for cs.AI
- Ensemble methods
- Larger models (RoBERTa, DeBERTa) on GPU

For any questions or clarifications, refer to the detailed analysis in `SOLUTION_FINAL.md` or examine the version backups in `backups/` directory.

---

**Generated for AI assistants working with this codebase**
Last updated: 2025-11-18
