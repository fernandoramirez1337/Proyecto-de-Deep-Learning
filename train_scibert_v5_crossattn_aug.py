"""
Training SciBERT V5.0: Cross-Attention + Back-Translation Augmentation

STRATEGY:
- Cross-Attention architecture for better title<->abstract interaction
- Back-Translation augmented cs.AI samples (450 -> 900)
- Expected improvement: +3-4% total (+1-2% Cross-Attention, +1.5-2.5% augmentation)
- Target: 59-60% accuracy

CONFIGURATION:
- Architecture: CrossAttentionSciBERT (bidirectional title<->abstract attention)
- Dataset: data/arxiv_papers_augmented.csv (cs.AI duplicated via EN->ES->EN)
- FREEZE_LAYERS: 3 (same as V3.7)
- DROPOUT: 0.35 (same as V3.7)
- CLASS_WEIGHTS: [2.0, 1.0, 1.0, 1.0] (same as V3.7)
- BATCH_SIZE: 12 (M2 optimized)
- LR: 5e-5 (same as V3.7)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pickle
import os

# Import Cross-Attention architecture
from advanced_cross_attention import CrossAttentionSciBERT
from preprocessing_scibert import SciBERTDataset


def prepare_augmented_data(data_path='data/arxiv_papers_augmented.csv',
                          test_size=0.15, val_size=0.15):
    """
    Prepare augmented dataset

    Args:
        data_path: Path to augmented CSV
        test_size: Fraction for test set
        val_size: Fraction for validation set

    Returns:
        train_dataset, val_dataset, test_dataset, tokenizer, label_encoder
    """
    print("="*70)
    print("LOADING AUGMENTED DATASET")
    print("="*70)

    # Load augmented data
    if not os.path.exists(data_path):
        print(f"ERROR: Augmented dataset not found: {data_path}")
        print("Please run: python advanced_data_augmentation.py")
        exit(1)

    df = pd.read_csv(data_path)
    print(f"\nOK Loaded {len(df)} samples")
    print(f"\nClass distribution:")
    print(df['category'].value_counts())
    print()

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    # Load SciBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    # Split data
    X = df[['title', 'abstract']]
    y = df['label']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Adjusted val size
    adjusted_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=42, stratify=y_temp
    )

    # Create datasets
    train_dataset = SciBERTDataset(
        X_train['title'].tolist(),
        X_train['abstract'].tolist(),
        y_train.tolist(),
        tokenizer,
        max_title_len=32,
        max_abstract_len=128
    )

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

    # Save label encoder
    with open('scibert_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print("="*70 + "\n")

    return train_dataset, val_dataset, test_dataset, tokenizer, le


def compute_class_weights_from_dataset(dataset, num_classes=4):
    """Compute class weights from dataset"""
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=num_classes)
    weights = len(labels) / (num_classes * counts)
    return torch.FloatTensor(weights)


class OptimizedTrainer:
    """Training loop with early stopping, scheduler, metrics"""

    def __init__(self, model, device, lr=5e-5, weight_decay=0.01,
                 class_weights=None, patience=3):
        self.model = model.to(device)
        self.device = device
        self.patience = patience

        # Optimizer con differential learning rates
        bert_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if 'bert' in name and param.requires_grad:
                bert_params.append(param)
            elif param.requires_grad:
                classifier_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': lr * 5, 'weight_decay': weight_decay * 2}
        ])

        # Loss con class weights y label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1,
            weight=class_weights.to(device) if class_weights is not None else None
        )

        self.best_val_acc = 0
        self.best_model_state = None
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            title_ids = batch['title_input_ids'].to(self.device)
            title_mask = batch['title_attention_mask'].to(self.device)
            abstract_ids = batch['abstract_input_ids'].to(self.device)
            abstract_mask = batch['abstract_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(title_ids, title_mask, abstract_ids, abstract_mask)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                title_ids = batch['title_input_ids'].to(self.device)
                title_mask = batch['title_attention_mask'].to(self.device)
                abstract_ids = batch['abstract_input_ids'].to(self.device)
                abstract_mask = batch['abstract_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(title_ids, title_mask, abstract_ids, abstract_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, acc, f1, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=10):
        """Full training loop with early stopping"""

        # Setup scheduler
        num_training_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        print("="*70)
        print("TRAINING V5.0: Cross-Attention + Back-Translation")
        print("="*70)
        print(f"\nEpochs: {epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Total steps: {num_training_steps}")
        print("="*70 + "\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader)

            # Metrics
            gap = abs(train_acc - val_acc)

            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"  Val F1:     {val_f1:.4f}")
            print(f"  Gap (Overfit): {gap:.4f} ({gap*100:.2f}%)")

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"  OK New best model! (val_acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")

            if self.patience_counter >= self.patience:
                print(f"\n⚠ Early stopping triggered (patience={self.patience})")
                break

        # Restore best model
        print(f"\nOK Training complete!")
        print(f"  Best val accuracy: {self.best_val_acc:.4f} ({self.best_val_acc*100:.2f}%)")
        self.model.load_state_dict(self.best_model_state)

        return self.history


def plot_history(history, save_path='scibert_v5_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"OK History plot saved: {save_path}")


def main():
    # Configuration
    FREEZE_BERT_LAYERS = 3
    DROPOUT = 0.35
    BATCH_SIZE = 12
    EPOCHS = 10
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI emphasis
    PATIENCE = 3

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Enable MPS fallback
    if device.type == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Prepare augmented data
    train_dataset, val_dataset, test_dataset, tokenizer, le = prepare_augmented_data()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Create Cross-Attention model
    print("="*70)
    print("CREATING CROSS-ATTENTION MODEL")
    print("="*70)
    print(f"\nArchitecture: CrossAttentionSciBERT")
    print(f"  Freeze layers: {FREEZE_BERT_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Innovation: Bidirectional title<->abstract cross-attention")
    print("="*70 + "\n")

    model = CrossAttentionSciBERT(
        num_classes=4,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS
    )

    # Compute class weights
    class_weights_tensor = torch.FloatTensor(CLASS_WEIGHTS)
    print(f"Class weights: {CLASS_WEIGHTS}")
    print(f"  (cs.AI x{CLASS_WEIGHTS[0]})\n")

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        device=device,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights_tensor,
        patience=PATIENCE
    )

    # Train
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)

    # Plot history
    plot_history(history)

    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70 + "\n")

    test_loss, test_acc, test_f1, test_preds, test_labels = trainer.evaluate(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1: {test_f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                               target_names=le.classes_, digits=4))

    # Per-class metrics
    from sklearn.metrics import recall_score, precision_score
    recalls = recall_score(test_labels, test_preds, average=None)
    precisions = precision_score(test_preds, test_labels, average=None, zero_division=0)

    cs_ai_idx = list(le.classes_).index('cs.AI')
    cs_ai_recall = recalls[cs_ai_idx]
    cs_ai_precision = precisions[cs_ai_idx]

    print(f"\ncs.AI specific metrics:")
    print(f"  Recall: {cs_ai_recall:.4f} ({cs_ai_recall*100:.2f}%)")
    print(f"  Precision: {cs_ai_precision:.4f} ({cs_ai_precision*100:.2f}%)")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'V5.0 Confusion Matrix (Cross-Attention + Augmentation)\\nTest Acc: {test_acc:.3f}',
              fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig('scibert_v5_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nOK Confusion matrix saved: scibert_v5_confusion.png")

    # Save model
    torch.save(model.state_dict(), 'best_scibert_v5_crossattn_aug.pth')
    print("OK Model saved: best_scibert_v5_crossattn_aug.pth")

    # Objectives check
    print("\n" + "="*70)
    print("OBJECTIVES CHECK")
    print("="*70)

    gap_acc = abs(test_acc - 0.60)
    gap_cs_ai = max(0, 0.30 - cs_ai_recall)
    gap_total = gap_acc + gap_cs_ai

    acc_target_met = test_acc >= 0.60
    cs_ai_target_met = cs_ai_recall > 0.30

    print(f"\nTest Accuracy >= 60%: {'YES YES' if acc_target_met else 'NO NO'} ({test_acc*100:.2f}%)")
    print(f"cs.AI Recall > 30%:   {'YES YES' if cs_ai_target_met else 'NO NO'} ({cs_ai_recall*100:.2f}%)")
    print(f"\nGap Total: {gap_total:.4f} ({gap_total*100:.2f}%)")

    # Compare with baseline
    print("\n" + "="*70)
    print("COMPARISON WITH V3.7+TT BASELINE")
    print("="*70)

    baseline_acc = 0.5617
    baseline_cs_ai = 0.3622
    baseline_gap = 0.0383

    improvement_acc = test_acc - baseline_acc
    improvement_cs_ai = cs_ai_recall - baseline_cs_ai
    improvement_gap = baseline_gap - gap_total

    print(f"\n{'Metric':<20} {'Baseline':<12} {'V5.0':<12} {'Improvement'}")
    print("-"*60)
    print(f"{'Test Accuracy':<20} {baseline_acc*100:>6.2f}% {test_acc*100:>11.2f}% {improvement_acc*100:>11.2f}%")
    print(f"{'cs.AI Recall':<20} {baseline_cs_ai*100:>6.2f}% {cs_ai_recall*100:>11.2f}% {improvement_cs_ai*100:>11.2f}%")
    print(f"{'Gap Total':<20} {baseline_gap*100:>6.2f}% {gap_total*100:>11.2f}% {improvement_gap*100:>11.2f}%")

    if acc_target_met and cs_ai_target_met:
        print("\n" + ""*20)
        print("YES SUCCESS! BOTH OBJECTIVES MET!")
        print(""*20)
    elif test_acc > baseline_acc:
        print("\nYES Improvement over baseline!")
    else:
        print("\nWARNING:  No improvement over baseline")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
