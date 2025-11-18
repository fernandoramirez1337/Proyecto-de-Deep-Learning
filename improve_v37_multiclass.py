"""
BETTER APPROACH: Multi-class threshold tuning on V3.7 (already good model)

V3.7+TT (single-class): 56.17% accuracy, 36.22% cs.AI recall
Goal: Improve to ~57-58% with multi-class thresholds
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from threshold_optimizer import ThresholdOptimizer
from preprocessing_scibert import prepare_scibert_data
from train_scibert_optimized import OptimizedSciBERTClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report

print("="*70)
print("V3.7 MULTI-CLASS THRESHOLD OPTIMIZATION")
print("="*70)
print("\nCurrent V3.7+TT (single-class):")
print("  - Accuracy: 56.17%")
print("  - cs.AI Recall: 36.22%")
print("  - Thresholds: [0.40, 0.5, 0.5, 0.5] (only cs.AI optimized)")
print("\nGoal: Optimize ALL class thresholds")
print("  - Expected improvement: +1-2%")
print("  - Target: ~57-58% accuracy")
print("="*70 + "\n")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}\n")

# Load data
print("Loading data...")
_, val_dataset, test_dataset, tokenizer, le = prepare_scibert_data(use_light_model=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
print(f"✓ Val set: {len(val_dataset)} samples")
print(f"✓ Test set: {len(test_dataset)} samples\n")

# Load V3.7 model
print("Loading V3.7 model...")
try:
    model = OptimizedSciBERTClassifier(
        num_classes=4,
        dropout=0.35,
        freeze_bert_layers=3
    )
    checkpoint = torch.load('best_scibert_v3.7_final.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✓ V3.7 model loaded\n")
except FileNotFoundError:
    print("ERROR: best_scibert_v3.7_final.pth not found!")
    print("Using best_scibert_optimized.pth instead...")
    checkpoint = torch.load('best_scibert_optimized.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("✓ Model loaded\n")

# Create threshold optimizer
print("Creating threshold optimizer...")
optimizer = ThresholdOptimizer(model, val_loader, device, le)

# Baseline: Single-class threshold (current V3.7+TT)
print("\n" + "="*70)
print("BASELINE: V3.7 with single-class threshold")
print("="*70)

baseline_thresholds = np.array([0.40, 0.5, 0.5, 0.5])
baseline_metrics = optimizer.evaluate_thresholds(baseline_thresholds, verbose=True)

# Strategy 1: Greedy Search (fast)
print("\n" + "="*70)
print("STRATEGY 1: GREEDY SEARCH - Multi-class optimization")
print("="*70)

greedy_thresholds, greedy_metrics, greedy_history = optimizer.greedy_search(
    initial_thresholds=baseline_thresholds,  # Start from current best
    threshold_range=(0.25, 0.65),
    step=0.05,
    optimize_metric='gap_total',
    minimize=True
)

# Strategy 2: Priority Search (cs.AI first)
print("\n" + "="*70)
print("STRATEGY 2: PRIORITY SEARCH - cs.AI first, then others")
print("="*70)

priority_thresholds, priority_metrics, priority_history = optimizer.class_priority_search(
    priority_order=None,  # cs.AI first automatically
    threshold_range=(0.25, 0.65),
    step=0.05,
    optimize_metric='gap_total',
    minimize=True
)

# Choose best strategy
print("\n" + "="*70)
print("STRATEGY COMPARISON")
print("="*70)

strategies = {
    'Baseline (single-class)': (baseline_thresholds, baseline_metrics),
    'Greedy Search': (greedy_thresholds, greedy_metrics),
    'Priority Search': (priority_thresholds, priority_metrics)
}

best_strategy = None
best_gap = float('inf')

print(f"\n{'Strategy':<25} {'Accuracy':<12} {'cs.AI Recall':<15} {'Gap Total':<12} {'Best?'}")
print("-"*75)

for name, (thresholds, metrics) in strategies.items():
    acc = metrics['accuracy']
    cs_ai_recall = metrics['cs_ai_recall']
    gap = metrics['gap_total']

    is_best = ""
    if gap < best_gap:
        best_gap = gap
        best_strategy = name
        is_best = " ★"

    print(f"{name:<25} {acc*100:>6.2f}% {cs_ai_recall*100:>11.2f}% {gap:>11.4f}{is_best}")

print("="*75)

# Evaluate best on test set
print("\n" + "="*70)
print(f"FINAL EVALUATION: {best_strategy}")
print("="*70)

best_thresholds = strategies[best_strategy][0]

# Get test predictions
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

test_preds = apply_thresholds(all_probs, best_thresholds)
test_acc = accuracy_score(all_labels, test_preds)
test_recalls = recall_score(all_labels, test_preds, average=None)
cs_ai_idx = list(le.classes_).index('cs.AI')
cs_ai_recall_test = test_recalls[cs_ai_idx]

print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"cs.AI Recall: {cs_ai_recall_test:.4f} ({cs_ai_recall_test*100:.2f}%)")
print(f"\nOptimized Thresholds: {best_thresholds}")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {best_thresholds[i]:.2f}")

print("\nClassification Report:")
print(classification_report(all_labels, test_preds, target_names=le.classes_, digits=4))

# Final comparison
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

gap_acc = abs(test_acc - 0.60)
gap_cs_ai = max(0, 0.30 - cs_ai_recall_test)
gap_total = gap_acc + gap_cs_ai

print(f"\n{'Model':<35} {'Accuracy':<12} {'cs.AI Recall':<15} {'Gap Total'}")
print("-"*75)
print(f"{'V3.7+TT (single-class, baseline)':<35} {'56.17%':<12} {'36.22%':<15} {'3.83%'}")
print(f"{f'V3.7+Multi-TT ({best_strategy})':<35} {f'{test_acc*100:.2f}%':<12} {f'{cs_ai_recall_test*100:.2f}%':<15} {f'{gap_total*100:.2f}%'}")
print(f"{'Improvement':<35} {f'{(test_acc-0.5617)*100:+.2f}%':<12} {f'{(cs_ai_recall_test-0.3622)*100:+.2f}%':<15} {f'{(gap_total-0.0383)*100:+.2f}%'}")
print("="*75)

# Objectives
acc_target_met = test_acc >= 0.60
cs_ai_target_met = cs_ai_recall_test > 0.30

print(f"\n{'='*70}")
print("OBJECTIVES CHECK")
print(f"{'='*70}")
print(f"Test Accuracy >= 60%: {'✅ YES' if acc_target_met else '❌ NO'} ({test_acc*100:.2f}%, gap: {gap_acc*100:+.2f}%)")
print(f"cs.AI Recall > 30%:   {'✅ YES' if cs_ai_target_met else '❌ NO'} ({cs_ai_recall_test*100:.2f}%, gap: {gap_cs_ai*100:+.2f}%)")
print(f"{'='*70}")

if test_acc > 0.5617:
    print("\n🎉 SUCCESS! Multi-class threshold tuning improved V3.7!")
    print(f"   New best model: V3.7+Multi-TT")
    print(f"   Improvement: {(test_acc-0.5617)*100:+.2f}% accuracy")
else:
    print("\n⚠️  Multi-class threshold didn't improve over single-class")
    print("   Stick with V3.7+TT (single-class) as best model")

print(f"\n{'='*70}\n")
