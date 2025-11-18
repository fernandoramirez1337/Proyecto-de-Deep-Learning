#!/bin/bash

# Quick script to train V2 and create ensemble

echo "======================================================================"
echo "ENSEMBLE EXPERIMENT - Training V2 for Ensemble"
echo "======================================================================"
echo ""
echo "Goal: Train V2 (59.17% acc, 13.78% cs.AI) for ensemble with V3.7"
echo "Expected result: ~58-59% accuracy"
echo "Time: ~60-80 minutes"
echo ""
echo "This is FOR FUN - no hard requirement!"
echo "======================================================================"
echo ""

# Check if V3.7 exists
if [ ! -f "best_scibert_v3.7_final.pth" ] && [ ! -f "best_scibert_optimized.pth" ]; then
    echo "ERROR: V3.7 model not found!"
    echo "Need V3.7 model for ensemble"
    exit 1
fi

echo "✓ V3.7 model found"
echo ""

# Setup V2 training script
echo "Setting up V2 training script..."
cp backups/train_scibert_v2_backup.py train_scibert_v2.py

# Update to save as v2
sed -i.bak "s/best_scibert_optimized.pth/best_scibert_v2.pth/g" train_scibert_v2.py
sed -i.bak "s/scibert_optimized_history.png/scibert_v2_history.png/g" train_scibert_v2.py
sed -i.bak "s/scibert_optimized_confusion.png/scibert_v2_confusion.png/g" train_scibert_v2.py

echo "✓ V2 script configured"
echo ""

# Set M2 environment
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_MPS_PREFER_METAL=1

echo "Environment variables set for M2"
echo ""

# Train V2
echo "======================================================================"
echo "TRAINING V2"
echo "======================================================================"
echo ""
echo "Configuration V2:"
echo "  - FREEZE_LAYERS: 8 (over-regularized)"
echo "  - DROPOUT: 0.5"
echo "  - LR: 3e-5"
echo "  - WEIGHT_DECAY: 0.05"
echo ""
echo "Expected: High accuracy (59%), Low cs.AI recall (14%)"
echo "Complements V3.7: Medium accuracy (57%), High cs.AI recall (28%)"
echo ""
echo "Starting training..."
echo ""

python train_scibert_v2.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ V2 Training Complete!"
    echo "======================================================================"
    echo ""
    echo "Next: Create ensemble V2 + V3.7"
    echo ""
    echo "Run:"
    echo "  python test_ensemble_v2_v37.py"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ V2 Training Failed"
    echo "======================================================================"
    exit 1
fi
