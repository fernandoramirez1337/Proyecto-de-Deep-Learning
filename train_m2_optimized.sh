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

echo "SciBERT V3.7 - Balanced cs.AI Focus"
echo "======================================"
echo "V3: cs.AI 26%, Acc 55% | V3.6: cs.AI 51%, Acc 50%"
echo "V3.7: cs.AI weight x2 (was x3) - seeking balance"
echo ""

# Run the training script
python train_scibert_optimized.py
