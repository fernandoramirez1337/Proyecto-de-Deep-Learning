#!/bin/bash

echo "="
echo "HYBRID CNN-LSTM MODEL TRAINING"
echo "="
echo ""
echo "Architecture (according to project definition):"
echo "  - CNN 1D for abstract feature extraction"
echo "  - Bidirectional LSTM for title processing"
echo "  - Self-attention over LSTM outputs"
echo "  - Global attention over CNN features"
echo "  - Weighted attention fusion"
echo "  - Variational dropout + batch normalization"
echo ""

# Check dataset
if [ ! -f "data/arxiv_papers_raw.csv" ]; then
    echo "ERROR: Dataset not found at data/arxiv_papers_raw.csv"
    exit 1
fi

echo "OK Dataset found"
echo ""

# M2 optimization if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - setting MPS backend"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Run training
echo "Starting training..."
echo ""
python train_hybrid.py

echo ""
echo "="
echo "TRAINING COMPLETE"
echo "="
echo ""
echo "Outputs:"
echo "  - best_hybrid_model.pth"
echo "  - vocab_hybrid.pkl"
echo "  - hybrid_training_history.png"
echo ""
echo "To visualize attention maps:"
echo "  python visualize_attention.py"
