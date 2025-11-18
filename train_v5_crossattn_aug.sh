#!/bin/bash

# V5.0 Complete Pipeline: Cross-Attention + Back-Translation
# Expected improvement: +3-4% accuracy
# Target: 59-60% test accuracy

echo "======================================================================"
echo "V5.0 TRAINING PIPELINE"
echo "Cross-Attention Architecture + Back-Translation Augmentation"
echo "======================================================================"
echo ""
echo "Strategy:"
echo "  1. Back-Translation: Duplicate cs.AI samples via EN→ES→EN (450→900)"
echo "  2. Cross-Attention: Bidirectional title↔abstract interaction"
echo "  3. Expected improvement: +3-4% total"
echo ""
echo "Configuration:"
echo "  - Architecture: CrossAttentionSciBERT"
echo "  - Dataset: Augmented (cs.AI x2)"
echo "  - FREEZE_LAYERS: 3"
echo "  - DROPOUT: 0.35"
echo "  - CLASS_WEIGHTS: [2.0, 1.0, 1.0, 1.0]"
echo "  - LR: 5e-5"
echo ""
echo "Time estimate:"
echo "  - Data augmentation: ~30-40 min (MarianMT translation)"
echo "  - Training: ~70-90 min (slightly longer due to larger dataset)"
echo "  - Total: ~2 hours"
echo ""
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

# Check if original dataset exists
if [ ! -f "data/arxiv_papers_raw.csv" ]; then
    echo "ERROR: Original dataset not found: data/arxiv_papers_raw.csv"
    echo "Please download dataset first"
    exit 1
fi

echo "✓ Original dataset found"
echo ""

# Step 1: Data Augmentation
echo "======================================================================"
echo "STEP 1: DATA AUGMENTATION (Back-Translation)"
echo "======================================================================"
echo ""
echo "Augmenting cs.AI samples via English→Spanish→English translation..."
echo "This will take ~30-40 minutes"
echo ""

if [ ! -f "data/arxiv_papers_augmented.csv" ]; then
    python advanced_data_augmentation.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Data augmentation failed"
        exit 1
    fi
else
    echo "⚠️  Augmented dataset already exists: data/arxiv_papers_augmented.csv"
    echo ""
    read -p "Use existing augmented dataset? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Regenerating augmented dataset..."
        rm data/arxiv_papers_augmented.csv
        python advanced_data_augmentation.py

        if [ $? -ne 0 ]; then
            echo ""
            echo "ERROR: Data augmentation failed"
            exit 1
        fi
    fi
fi

echo ""
echo "✓ Data augmentation complete!"
echo ""

# Step 2: Training
echo "======================================================================"
echo "STEP 2: TRAINING (Cross-Attention on Augmented Data)"
echo "======================================================================"
echo ""
echo "Training V5.0 model..."
echo "This will take ~70-90 minutes"
echo ""

python train_scibert_v5_crossattn_aug.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ V5.0 TRAINING COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "Results saved:"
    echo "  - Model: best_scibert_v5_crossattn_aug.pth"
    echo "  - History: scibert_v5_history.png"
    echo "  - Confusion matrix: scibert_v5_confusion.png"
    echo ""
    echo "Next steps:"
    echo "  1. Review test metrics (should be ~59-60% accuracy)"
    echo "  2. If accuracy < 60%, consider threshold tuning:"
    echo "     python threshold_tuning_v5.py"
    echo "  3. If accuracy >= 60%, celebrate! 🎉"
    echo ""
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ V5.0 TRAINING FAILED"
    echo "======================================================================"
    echo ""
    echo "Please check error messages above"
    exit 1
fi
