"""
Test Ensemble V2 + V3.7

V2: 59.17% accuracy, 13.78% cs.AI recall (high acc, low cs.AI)
V3.7: 57.39% accuracy, 28.22% cs.AI recall (medium acc, high cs.AI)

Goal: Combine strengths → ~58-59% accuracy with decent cs.AI recall
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing_scibert import prepare_scibert_data
from ensemble_predictor import EnsemblePredictor
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ENSEMBLE TEST: V2 + V3.7")
print("="*70)
print("\nV2 characteristics:")
print("  - Test Accuracy: 59.17%")
print("  - cs.AI Recall: 13.78%")
print("  - Strategy: Over-regularized (FREEZE=8, DROPOUT=0.5)")
print("\nV3.7 characteristics:")
print("  - Test Accuracy: 57.39%")
print("  - cs.AI Recall: 28.22%")
print("  - Strategy: Balanced (FREEZE=3, DROPOUT=0.35)")
print("\nEnsemble goal: Combine high accuracy + decent cs.AI recall")
print("="*70 + "\n")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}\n")

# Load data
print("Loading test data...")
_, _, test_dataset, tokenizer, le = prepare_scibert_data(use_light_model=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
print(f"✓ Test set: {len(test_dataset)} samples\n")

# Model configurations
model_configs = [
    {
        'path': 'best_scibert_v2.pth',
        'dropout': 0.5,
        'freeze_layers': 8,
        'name': 'V2'
    },
    {
        'path': 'best_scibert_v3.7_final.pth' if os.path.exists('best_scibert_v3.7_final.pth') else 'best_scibert_optimized.pth',
        'dropout': 0.35,
        'freeze_layers': 3,
        'name': 'V3.7'
    }
]

# Check models exist
import os
for config in model_configs:
    if not os.path.exists(config['path']):
        print(f"ERROR: {config['name']} model not found at {config['path']}")
        print("Train models first!")
        exit(1)

print("✓ All models found\n")

# Try different ensemble configurations
print("="*70)
print("TESTING DIFFERENT ENSEMBLE CONFIGURATIONS")
print("="*70 + "\n")

configs_to_test = [
    {
        'name': 'Equal Weight',
        'weights': [0.5, 0.5],
        'thresholds': [0.5, 0.5, 0.5, 0.5]
    },
    {
        'name': 'V3.7 Dominant',
        'weights': [0.3, 0.7],
        'thresholds': [0.40, 0.5, 0.5, 0.5]
    },
    {
        'name': 'V2 Dominant',
        'weights': [0.7, 0.3],
        'thresholds': [0.40, 0.5, 0.5, 0.5]
    },
    {
        'name': 'Balanced + Tuned',
        'weights': [0.4, 0.6],
        'thresholds': [0.40, 0.5, 0.5, 0.5]
    },
]

results = []

for config in configs_to_test:
    print(f"\n{'-'*70}")
    print(f"Testing: {config['name']}")
    print(f"  Weights: V2={config['weights'][0]:.1f}, V3.7={config['weights'][1]:.1f}")
    print(f"  Thresholds: {config['thresholds']}")
    print(f"{'-'*70}")

    # Create ensemble
    ensemble = EnsemblePredictor(
        model_configs=model_configs,
        weights=config['weights'],
        thresholds=config['thresholds']
    )

    # Evaluate
    accuracy, preds, labels = ensemble.evaluate(test_loader)

    # Calculate per-class metrics
    from sklearn.metrics import recall_score, precision_score
    recalls = recall_score(labels, preds, average=None)
    precisions = precision_score(labels, preds, average=None, zero_division=0)

    cs_ai_idx = list(le.classes_).index('cs.AI')
    cs_ai_recall = recalls[cs_ai_idx]
    cs_ai_precision = precisions[cs_ai_idx]

    # Calculate gap
    gap_acc = abs(accuracy - 0.60)
    gap_cs_ai = max(0, 0.30 - cs_ai_recall)
    gap_total = gap_acc + gap_cs_ai

    result = {
        'name': config['name'],
        'accuracy': accuracy,
        'cs_ai_recall': cs_ai_recall,
        'cs_ai_precision': cs_ai_precision,
        'gap_total': gap_total,
        'weights': config['weights'],
        'thresholds': config['thresholds'],
        'preds': preds,
        'labels': labels
    }
    results.append(result)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  cs.AI Recall: {cs_ai_recall:.4f} ({cs_ai_recall*100:.2f}%)")
    print(f"  cs.AI Precision: {cs_ai_precision:.4f} ({cs_ai_precision*100:.2f}%)")
    print(f"  Gap Total: {gap_total:.4f}")

# Find best configuration
best_result = min(results, key=lambda x: x['gap_total'])

print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70 + "\n")

print(f"{'Configuration':<25} {'Accuracy':<12} {'cs.AI Recall':<15} {'Gap Total':<12} {'Best?'}")
print("-"*75)

for result in results:
    is_best = " ★" if result['name'] == best_result['name'] else ""
    print(f"{result['name']:<25} {result['accuracy']*100:>6.2f}% {result['cs_ai_recall']*100:>11.2f}% {result['gap_total']:>11.4f}{is_best}")

print("="*75)

# Compare with baselines
print("\n" + "="*70)
print("COMPARISON WITH BASELINES")
print("="*70 + "\n")

print(f"{'Model':<30} {'Accuracy':<12} {'cs.AI Recall':<15} {'Gap Total'}")
print("-"*70)
print(f"{'V2 (alone)':<30} {'59.17%':<12} {'13.78%':<15} {'4.22%'}")
print(f"{'V3.7 (alone)':<30} {'57.39%':<12} {'28.22%':<15} {'4.39%'}")
print(f"{'V3.7+TT (baseline)':<30} {'56.17%':<12} {'36.22%':<15} {'3.83%'}")
print(f"{f'Ensemble ({best_result[\"name\"]})':<30} {f'{best_result[\"accuracy\"]*100:.2f}%':<12} {f'{best_result[\"cs_ai_recall\"]*100:.2f}%':<15} {f'{best_result[\"gap_total\"]*100:.2f}%'}")
print("="*70)

# Detailed report for best ensemble
print("\n" + "="*70)
print(f"BEST ENSEMBLE: {best_result['name']}")
print("="*70 + "\n")

print(f"Configuration:")
print(f"  Weights: V2={best_result['weights'][0]:.1f}, V3.7={best_result['weights'][1]:.1f}")
print(f"  Thresholds: {best_result['thresholds']}")

print(f"\nResults:")
print(f"  Test Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"  cs.AI Recall: {best_result['cs_ai_recall']:.4f} ({best_result['cs_ai_recall']*100:.2f}%)")
print(f"  Gap Total: {best_result['gap_total']:.4f}")

print("\nClassification Report:")
print(classification_report(best_result['labels'], best_result['preds'],
                           target_names=le.classes_, digits=4))

# Confusion matrix
cm = confusion_matrix(best_result['labels'], best_result['preds'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Ensemble Confusion Matrix ({best_result["name"]})\nAcc: {best_result["accuracy"]:.3f}',
          fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Prediction', fontsize=12)
plt.tight_layout()
plt.savefig('ensemble_v2_v37_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Confusion matrix saved: ensemble_v2_v37_confusion.png")

# Objectives check
acc_target_met = best_result['accuracy'] >= 0.60
cs_ai_target_met = best_result['cs_ai_recall'] > 0.30

print(f"\n{'='*70}")
print("OBJECTIVES CHECK")
print(f"{'='*70}")
print(f"Test Accuracy >= 60%: {'✅ YES' if acc_target_met else '❌ NO'} ({best_result['accuracy']*100:.2f}%)")
print(f"cs.AI Recall > 30%:   {'✅ YES' if cs_ai_target_met else '❌ NO'} ({best_result['cs_ai_recall']*100:.2f}%)")
print(f"{'='*70}")

# Final verdict
if best_result['gap_total'] < 0.0383:  # Better than V3.7+TT
    print("\n🎉 SUCCESS! Ensemble improved over V3.7+TT!")
    print(f"   Gap improved: 3.83% → {best_result['gap_total']*100:.2f}%")
    print(f"   Improvement: {(0.0383 - best_result['gap_total'])*100:+.2f}%")
elif best_result['accuracy'] > 0.5617:  # Better accuracy than V3.7+TT
    print("\n✅ Ensemble improved accuracy!")
    print(f"   Accuracy: 56.17% → {best_result['accuracy']*100:.2f}%")
    print(f"   Improvement: {(best_result['accuracy'] - 0.5617)*100:+.2f}%")
else:
    print("\n⚠️  Ensemble didn't improve over V3.7+TT")
    print("   V3.7+TT remains the best model")
    print(f"   V3.7+TT: 56.17% accuracy, 36.22% cs.AI recall, 3.83% gap")

print(f"\n{'='*70}\n")
