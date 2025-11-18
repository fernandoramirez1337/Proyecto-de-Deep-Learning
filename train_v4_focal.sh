#!/bin/bash

# Training Script for V4.0 - Focal Loss
# Optimized for M2 MacBook Air

echo "======================================================================"
echo "Training SciBERT V4.0 - Focal Loss Improvement"
echo "======================================================================"
echo ""
echo "Version: V4.0"
echo "Strategy: Focal Loss (gamma=2.0) + Class Weighting (cs.AI x2.0)"
echo "Base: V3.7 architecture"
echo ""
echo "Expected improvements:"
echo "  - Accuracy: +2-3% (target: 58-59%)"
echo "  - cs.AI Recall: Maintain or improve (>30%)"
echo "  - Gap Total: < 3.83%"
echo ""
echo "Estimated training time: 60-80 minutes on M2"
echo "======================================================================"
echo ""

# M2 MacBook Air optimizations
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_MPS_PREFER_METAL=1

echo "Environment variables set for M2 optimization"
echo ""

# Verificar que existe el dataset
if [ ! -f "data/arxiv_papers_raw.csv" ]; then
    echo "ERROR: Dataset not found at data/arxiv_papers_raw.csv"
    echo "Please download the dataset first:"
    echo "  python scripts/download_data.py"
    exit 1
fi

echo "✓ Dataset found"
echo ""

# Ejecutar entrenamiento
echo "Starting training..."
echo ""

python train_scibert_v4_focal.py

# Verificar si el entrenamiento fue exitoso
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Training completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Output files:"
    echo "  - best_scibert_v4_focal.pth (model checkpoint)"
    echo "  - scibert_v4_focal_history.png (training curves)"
    echo "  - scibert_v4_focal_confusion.png (confusion matrix)"
    echo ""
    echo "Next steps:"
    echo "  1. Optimize thresholds: python -c 'from threshold_optimizer ...'"
    echo "  2. Evaluate all improvements: python evaluate_all_improvements.py"
    echo "  3. Compare with baseline: Check metrics in output"
    echo ""
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ Training failed!"
    echo "======================================================================"
    echo ""
    echo "Check the error messages above for details."
    echo "Common issues:"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo "  - Out of memory: Reduce BATCH_SIZE in train_scibert_v4_focal.py"
    echo "  - Dataset not found: Run scripts/download_data.py"
    echo ""
    exit 1
fi
