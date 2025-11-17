#!/bin/bash
# M2 MacBook Air - SciBERT V3.5 Training Script
# V3.5: Punto medio entre V2 y V3

# Set MPS optimizations
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_MPS_PREFER_METAL=1

echo "SciBERT V3.8 - Fine-Tuning cs.AI Weight"
echo "=========================================="
echo "V3.7: cs.AI 28.22%, Acc 57.39% (SO CLOSE!)"
echo "V3.8: cs.AI weight x2.3 (was x2.0) - final push"
echo ""

# Run the training script
python train_scibert_optimized.py
