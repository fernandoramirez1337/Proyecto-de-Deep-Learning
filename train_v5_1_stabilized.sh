#!/bin/bash

# V5.1 Training: STABILIZED Cross-Attention + Back-Translation
# Fixes the crash issue from V5.0

echo "======================================================================"
echo "V5.1 TRAINING: STABILIZED"
echo "======================================================================"
echo ""
echo "Problem in V5.0:"
echo "  Epoch 1: 57.60% val acc (PEAK)"
echo "  Epoch 2: 46.84% val acc (CRASH -10.76%)"
echo ""
echo "Stabilization Fixes:"
echo "  - Lower LR: 5e-5 → 3e-5 (40% reduction)"
echo "  - Softer class weights: 2.0 → 1.4 (30% reduction)"
echo "  - Warmup: 10% of steps (prevent early overfitting)"
echo "  - Lower dropout: 0.35 → 0.30"
echo ""
echo "Configuration:"
echo "  - Architecture: CrossAttentionSciBERT (same as V5.0)"
echo "  - Dataset: Augmented (450 cs.AI duplicated, already created)"
echo "  - FREEZE_LAYERS: 3"
echo "  - DROPOUT: 0.30"
echo "  - CLASS_WEIGHTS: [1.4, 1.0, 1.0, 1.0]"
echo "  - LR: 3e-5"
echo ""
echo "Expected Results:"
echo "  - Stable training (no crash)"
echo "  - Test accuracy: 59-60%"
echo "  - cs.AI recall: 35-40%"
echo "  - cs.LG recall: > 45% (was 37.33%)"
echo ""
echo "Time estimate: ~70-90 minutes"
echo "======================================================================"
echo ""

# Set M2 environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_MPS_PREFER_METAL=1

echo "✓ Environment variables set for M2"
echo ""

# Check if augmented dataset exists
if [ ! -f "data/arxiv_papers_augmented.csv" ]; then
    echo "ERROR: Augmented dataset not found: data/arxiv_papers_augmented.csv"
    echo "Please run V5.0 augmentation first:"
    echo "  ./train_v5_crossattn_aug.sh"
    exit 1
fi

echo "✓ Augmented dataset found"
echo ""

echo "======================================================================"
echo "TRAINING V5.1 (Stabilized Hyperparameters)"
echo "======================================================================"
echo ""
echo "Training on augmented dataset..."
echo "This will take ~70-90 minutes"
echo ""

python train_scibert_v5_1_stabilized.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ V5.1 TRAINING COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "Results saved:"
    echo "  - Model: best_scibert_v5_1_stabilized.pth"
    echo "  - History: scibert_v5_1_history.png"
    echo "  - Confusion matrix: scibert_v5_1_confusion.png"
    echo ""
    echo "Next steps:"
    echo "  1. Review test metrics"
    echo "  2. Compare with V5.0 baseline (57.01% acc, 41.89% cs.AI recall)"
    echo "  3. Check if training was stable (no crash)"
    echo ""
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ V5.1 TRAINING FAILED"
    echo "======================================================================"
    echo ""
    echo "Please check error messages above"
    exit 1
fi
