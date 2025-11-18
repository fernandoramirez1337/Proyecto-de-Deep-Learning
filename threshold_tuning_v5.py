"""
Multi-Class Threshold Tuning for V5.0

PROBLEM IDENTIFIED:
- cs.CV over-predicted: 88.44% recall (too aggressive)
- cs.LG under-predicted: 37.33% recall (too conservative)
- Class weights 2.0 + augmented data → imbalanced probabilities

SOLUTION:
- Optimize decision thresholds per class
- Rebalance predictions without retraining
- Expected: +2-3% accuracy improvement

APPROACH:
- Greedy search: Optimize each class threshold sequentially
- Metric: Test accuracy
- Validate on val set, evaluate on test set
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os

from advanced_cross_attention import CrossAttentionSciBERT
from preprocessing_scibert import SciBERTDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


def load_augmented_data():
    """Load augmented dataset with same splits as training"""
    df = pd.read_csv('data/arxiv_papers_augmented.csv')

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    # Split data (same as training)
    X = df[['title', 'abstract']]
    y = df['label']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/(1-0.15), random_state=42, stratify=y_temp
    )

    # Create datasets
    val_dataset = SciBERTDataset(
        X_val['title'].tolist(),
        X_val['abstract'].tolist(),
        y_val.tolist(),
        tokenizer,
        max_title_len=32,
        max_abstract_len=128
    )

    test_dataset = SciBERTDataset(
        X_test['title'].tolist(),
        X_test['abstract'].tolist(),
        y_test.tolist(),
        tokenizer,
        max_title_len=32,
        max_abstract_len=128
    )

    return val_dataset, test_dataset, tokenizer, le


def get_predictions(model, dataloader, device):
    """Get probability predictions from model"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Getting predictions'):
            title_ids = batch['title_input_ids'].to(device)
            title_mask = batch['title_attention_mask'].to(device)
            abstract_ids = batch['abstract_input_ids'].to(device)
            abstract_mask = batch['abstract_attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(title_ids, title_mask, abstract_ids, abstract_mask)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    return all_probs, all_labels


def apply_thresholds(probs, thresholds):
    """
    Apply per-class thresholds to probabilities

    Strategy:
    - For each sample, check if any class exceeds its threshold
    - If multiple classes exceed, pick highest probability
    - If none exceed, pick highest probability (fallback)
    """
    predictions = []

    for prob in probs:
        # Check which classes exceed their thresholds
        exceeds = prob >= thresholds

        if exceeds.any():
            # Among classes that exceed threshold, pick highest prob
            masked_probs = np.where(exceeds, prob, -np.inf)
            predictions.append(np.argmax(masked_probs))
        else:
            # Fallback: no class exceeds threshold, pick highest
            predictions.append(np.argmax(prob))

    return np.array(predictions)


def optimize_thresholds_greedy(val_probs, val_labels, num_classes=4,
                               threshold_range=(0.20, 0.60), step=0.05):
    """
    Greedy threshold optimization

    Optimize each class threshold sequentially to maximize accuracy
    """
    best_thresholds = np.array([0.25] * num_classes)  # Start at 0.25 (default argmax)

    print("="*70)
    print("GREEDY THRESHOLD OPTIMIZATION")
    print("="*70)
    print(f"\nInitial thresholds: {best_thresholds}")
    print(f"Search range: {threshold_range}, step: {step}")
    print()

    # Baseline accuracy
    baseline_preds = apply_thresholds(val_probs, best_thresholds)
    baseline_acc = accuracy_score(val_labels, baseline_preds)
    print(f"Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print()

    # Optimize each class sequentially
    for class_idx in range(num_classes):
        print(f"Optimizing class {class_idx}...")

        best_class_threshold = best_thresholds[class_idx]
        best_class_acc = baseline_acc

        # Try different thresholds for this class
        thresholds_to_try = np.arange(threshold_range[0], threshold_range[1] + step, step)

        for threshold in thresholds_to_try:
            # Test this threshold
            test_thresholds = best_thresholds.copy()
            test_thresholds[class_idx] = threshold

            preds = apply_thresholds(val_probs, test_thresholds)
            acc = accuracy_score(val_labels, preds)

            if acc > best_class_acc:
                best_class_acc = acc
                best_class_threshold = threshold

        # Update threshold for this class
        best_thresholds[class_idx] = best_class_threshold
        improvement = best_class_acc - baseline_acc

        print(f"  Best threshold: {best_class_threshold:.2f}")
        print(f"  Accuracy: {best_class_acc:.4f} ({best_class_acc*100:.2f}%)")
        print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print()

        # Update baseline for next class
        baseline_acc = best_class_acc

    print("="*70)
    print(f"✓ Optimization complete!")
    print(f"Final thresholds: {best_thresholds}")
    print(f"Final val accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print("="*70)
    print()

    return best_thresholds


def evaluate_with_thresholds(probs, labels, thresholds, label_encoder):
    """Evaluate predictions with custom thresholds"""
    predictions = apply_thresholds(probs, thresholds)

    acc = accuracy_score(labels, predictions)

    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(labels, predictions,
                               target_names=label_encoder.classes_, digits=4))

    # Per-class metrics
    from sklearn.metrics import recall_score, precision_score
    recalls = recall_score(labels, predictions, average=None)
    precisions = precision_score(labels, predictions, average=None, zero_division=0)

    print("\nPer-Class Metrics:")
    print("-" * 60)
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name:<10} Precision: {precisions[idx]:.4f}  "
              f"Recall: {recalls[idx]:.4f}  Threshold: {thresholds[idx]:.2f}")

    return acc, predictions


def plot_comparison(baseline_cm, tuned_cm, label_encoder, save_path='v5_threshold_comparison.png'):
    """Plot confusion matrices comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Baseline
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=axes[0])
    axes[0].set_title('V5.0 Baseline (Default Thresholds)', fontsize=14, pad=20)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Prediction', fontsize=12)

    # Tuned
    sns.heatmap(tuned_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=axes[1])
    axes[1].set_title('V5.0 + Threshold Tuning', fontsize=14, pad=20)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Prediction', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved: {save_path}")


def main():
    print("="*70)
    print("V5.0 MULTI-CLASS THRESHOLD TUNING")
    print("="*70)
    print("\nGoal: Rebalance predictions to maximize accuracy")
    print("Problem: cs.CV over-predicted, cs.LG under-predicted")
    print("Expected: +2-3% accuracy improvement")
    print("="*70 + "\n")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    print("Loading augmented dataset...")
    val_dataset, test_dataset, tokenizer, le = load_augmented_data()

    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    # Load V5.0 model
    print("Loading V5.0 model...")
    model = CrossAttentionSciBERT(num_classes=4, dropout=0.35, freeze_bert_layers=3)
    model.load_state_dict(torch.load('best_scibert_v5_crossattn_aug.pth',
                                     map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")

    # Get predictions
    print("Getting validation predictions...")
    val_probs, val_labels = get_predictions(model, val_loader, device)
    print(f"✓ Got {len(val_probs)} validation predictions\n")

    print("Getting test predictions...")
    test_probs, test_labels = get_predictions(model, test_loader, device)
    print(f"✓ Got {len(test_probs)} test predictions\n")

    # Baseline (default thresholds = 0.25)
    print("="*70)
    print("BASELINE EVALUATION (Default Thresholds)")
    print("="*70)

    default_thresholds = np.array([0.25, 0.25, 0.25, 0.25])
    baseline_acc, baseline_preds = evaluate_with_thresholds(
        test_probs, test_labels, default_thresholds, le
    )
    baseline_cm = confusion_matrix(test_labels, baseline_preds)

    # Optimize thresholds on validation set
    print("\n")
    optimized_thresholds = optimize_thresholds_greedy(val_probs, val_labels)

    # Evaluate on test set with optimized thresholds
    print("="*70)
    print("FINAL EVALUATION (Optimized Thresholds)")
    print("="*70)

    tuned_acc, tuned_preds = evaluate_with_thresholds(
        test_probs, test_labels, optimized_thresholds, le
    )
    tuned_cm = confusion_matrix(test_labels, tuned_preds)

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    improvement = tuned_acc - baseline_acc

    print(f"\n{'Metric':<30} {'Baseline':<12} {'Tuned':<12} {'Change'}")
    print("-"*70)
    print(f"{'Test Accuracy':<30} {baseline_acc*100:>6.2f}% {tuned_acc*100:>11.2f}% {improvement*100:>11.2f}%")

    # cs.AI metrics
    from sklearn.metrics import recall_score
    baseline_cs_ai_recall = recall_score(test_labels, baseline_preds, average=None)[0]
    tuned_cs_ai_recall = recall_score(test_labels, tuned_preds, average=None)[0]
    cs_ai_change = tuned_cs_ai_recall - baseline_cs_ai_recall

    print(f"{'cs.AI Recall':<30} {baseline_cs_ai_recall*100:>6.2f}% {tuned_cs_ai_recall*100:>11.2f}% {cs_ai_change*100:>11.2f}%")

    # Objective check
    print("\n" + "="*70)
    print("OBJECTIVES CHECK")
    print("="*70)

    target_acc = 0.60
    target_cs_ai = 0.30

    acc_met = tuned_acc >= target_acc
    cs_ai_met = tuned_cs_ai_recall > target_cs_ai

    print(f"\nTest Accuracy >= 60%: {'✅ YES' if acc_met else '❌ NO'} ({tuned_acc*100:.2f}%)")
    print(f"cs.AI Recall > 30%:   {'✅ YES' if cs_ai_met else '❌ NO'} ({tuned_cs_ai_recall*100:.2f}%)")

    if acc_met and cs_ai_met:
        print("\n" + "🎉"*20)
        print("✅ SUCCESS! BOTH OBJECTIVES MET!")
        print("🎉"*20)
    elif tuned_acc > baseline_acc:
        print("\n✅ Improvement over baseline!")
    else:
        print("\n⚠️  No improvement from threshold tuning")

    # Plot comparison
    plot_comparison(baseline_cm, tuned_cm, le)

    # Save optimized thresholds
    np.save('v5_optimized_thresholds.npy', optimized_thresholds)
    print(f"\n✓ Optimized thresholds saved: v5_optimized_thresholds.npy")

    print("\n" + "="*70)
    print("✓ THRESHOLD TUNING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
