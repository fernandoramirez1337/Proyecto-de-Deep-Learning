"""
Quick fix: Apply threshold tuning to V4.0 to see if it can be rescued
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from threshold_optimizer import ThresholdOptimizer
from preprocessing_scibert import prepare_scibert_data
from train_scibert_optimized import OptimizedSciBERTClassifier

print("="*70)
print("V4.0 THRESHOLD OPTIMIZATION - Rescue Attempt")
print("="*70)
print("\nV4.0 baseline (no threshold): 53.33% accuracy, 28.00% cs.AI recall")
print("Goal: Improve with threshold tuning\n")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}\n")

# Load data
print("Loading validation data...")
_, val_dataset, test_dataset, tokenizer, le = prepare_scibert_data(use_light_model=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
print(f"✓ Val set: {len(val_dataset)} samples")
print(f"✓ Test set: {len(test_dataset)} samples\n")

# Load V4.0 model
print("Loading V4.0 model...")
model = OptimizedSciBERTClassifier(
    num_classes=4,
    dropout=0.35,
    freeze_bert_layers=3
)
checkpoint = torch.load('best_scibert_v4_focal.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print("✓ Model loaded\n")

# Create optimizer
print("Creating threshold optimizer...")
optimizer = ThresholdOptimizer(model, val_loader, device, le)

# Try greedy search
print("\n" + "="*70)
print("GREEDY SEARCH - Optimizing thresholds")
print("="*70)

best_thresholds, best_metrics, history = optimizer.greedy_search(
    threshold_range=(0.25, 0.65),
    step=0.05,
    optimize_metric='gap_total',
    minimize=True
)

print("\n" + "="*70)
print("EVALUATION ON TEST SET")
print("="*70)

# Evaluate on test set with optimized thresholds
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        title_ids = batch['title_input_ids'].to(device)
        title_mask = batch['title_attention_mask'].to(device)
        abstract_ids = batch['abstract_input_ids'].to(device)
        abstract_mask = batch['abstract_attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()

        outputs = model(title_ids, title_mask, abstract_ids, abstract_mask)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels)

all_probs = np.vstack(all_probs)
all_labels = np.array(all_labels)

# Apply thresholds
def apply_thresholds(probs, thresholds):
    num_samples = probs.shape[0]
    predictions = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        sample_probs = probs[i]
        candidates = []
        for class_idx, (prob, threshold) in enumerate(zip(sample_probs, thresholds)):
            if prob >= threshold:
                candidates.append((class_idx, prob))

        if candidates:
            predictions[i] = max(candidates, key=lambda x: x[1])[0]
        else:
            predictions[i] = sample_probs.argmax()

    return predictions

from sklearn.metrics import accuracy_score, recall_score

# Without thresholds
preds_no_threshold = all_probs.argmax(axis=1)
acc_no_threshold = accuracy_score(all_labels, preds_no_threshold)
recalls_no_threshold = recall_score(all_labels, preds_no_threshold, average=None)
cs_ai_idx = list(le.classes_).index('cs.AI')
cs_ai_recall_no_threshold = recalls_no_threshold[cs_ai_idx]

# With thresholds
preds_with_threshold = apply_thresholds(all_probs, best_thresholds)
acc_with_threshold = accuracy_score(all_labels, preds_with_threshold)
recalls_with_threshold = recall_score(all_labels, preds_with_threshold, average=None)
cs_ai_recall_with_threshold = recalls_with_threshold[cs_ai_idx]

print("\nRESULTS COMPARISON:")
print("="*70)
print(f"{'Metric':<25} {'No Threshold':<20} {'With Threshold':<20} {'Δ':<10}")
print("-"*70)
print(f"{'Test Accuracy':<25} {acc_no_threshold*100:>6.2f}% {acc_with_threshold*100:>15.2f}% {(acc_with_threshold-acc_no_threshold)*100:>+9.2f}%")
print(f"{'cs.AI Recall':<25} {cs_ai_recall_no_threshold*100:>6.2f}% {cs_ai_recall_with_threshold*100:>15.2f}% {(cs_ai_recall_with_threshold-cs_ai_recall_no_threshold)*100:>+9.2f}%")
print("="*70)

print(f"\nOptimized Thresholds: {best_thresholds}")
print(f"  cs.AI: {best_thresholds[0]:.2f}")
print(f"  cs.CL: {best_thresholds[1]:.2f}")
print(f"  cs.CV: {best_thresholds[2]:.2f}")
print(f"  cs.LG: {best_thresholds[3]:.2f}")

# Compare with V3.7+TT baseline
print("\n" + "="*70)
print("COMPARISON WITH V3.7+TT BASELINE")
print("="*70)
print(f"{'Model':<30} {'Accuracy':<15} {'cs.AI Recall':<15} {'Status'}")
print("-"*70)
print(f"{'V3.7+TT (baseline)':<30} {'56.17%':<15} {'36.22%':<15} {'Best'}")
print(f"{'V4.0 (no threshold)':<30} {f'{acc_no_threshold*100:.2f}%':<15} {f'{cs_ai_recall_no_threshold*100:.2f}%':<15} {'Worse'}")
print(f"{'V4.0+TT (optimized)':<30} {f'{acc_with_threshold*100:.2f}%':<15} {f'{cs_ai_recall_with_threshold*100:.2f}%':<15} {'???'}")
print("="*70)

# Verdict
if acc_with_threshold >= 0.5617:
    print("\n✅ SUCCESS! V4.0+TT improved over baseline!")
else:
    improvement = (acc_with_threshold - acc_no_threshold) * 100
    still_behind = (0.5617 - acc_with_threshold) * 100
    print(f"\n⚠️  V4.0+TT improved by {improvement:+.2f}% but still {still_behind:.2f}% behind baseline")
    print("   Recommendation: Use V3.7+TT or try alternative approaches")

print("\n" + "="*70)
