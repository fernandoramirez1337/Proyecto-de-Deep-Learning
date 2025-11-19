"""
Training script for Hybrid CNN-LSTM model

Train the hybrid architecture according to project definition:
- CNN 1D for abstracts
- Bidirectional LSTM for titles
- Self-attention and global attention
- Weighted fusion
- Variational dropout and batch normalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from hybrid_cnn_lstm import HybridCNNLSTM
from preprocessing_hybrid import prepare_hybrid_data


# Configuration
VOCAB_SIZE = 50000
EMBED_DIM = 300
NUM_FILTERS = 256
KERNEL_SIZES = [3, 4, 5]
LSTM_HIDDEN = 256
NUM_CLASSES = 4
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
WEIGHT_DECAY = 1e-4
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI minority class
PATIENCE = 5
MAX_TITLE_LEN = 30
MAX_ABSTRACT_LEN = 200


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Train'):
        title_ids = batch['title_ids'].to(device)
        abstract_ids = batch['abstract_ids'].to(device)
        title_mask = batch['title_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(title_ids, abstract_ids, title_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Val'):
            title_ids = batch['title_ids'].to(device)
            abstract_ids = batch['abstract_ids'].to(device)
            title_mask = batch['title_mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(title_ids, abstract_ids, title_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    print("="*70)
    print("TRAINING HYBRID CNN-LSTM MODEL")
    print("="*70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Prepare data
    train_dataset, val_dataset, test_dataset, vocab, le = prepare_hybrid_data(
        max_vocab_size=VOCAB_SIZE,
        max_title_len=MAX_TITLE_LEN,
        max_abstract_len=MAX_ABSTRACT_LEN
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if device.type == "mps" else 2,
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 if device.type == "mps" else 2,
        pin_memory=device.type == "cuda"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 if device.type == "mps" else 2,
        pin_memory=device.type == "cuda"
    )

    # Model
    model = HybridCNNLSTM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        lstm_hidden=LSTM_HIDDEN,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Gap: {abs(train_acc - val_acc):.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"OK New best model! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nOK Best validation accuracy: {best_val_acc:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab),
        'class_names': le.classes_
    }, 'best_hybrid_model.pth')
    print("OK Model saved: best_hybrid_model.pth")

    # Test evaluation
    print(f"\n{'='*70}")
    print("TEST EVALUATION")
    print(f"{'='*70}\n")

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1: {test_f1:.4f}")

    # Classification report
    print(f"\n{classification_report(test_labels, test_preds, target_names=le.classes_, digits=4)}")

    # Check objectives
    cs_ai_idx = list(le.classes_).index('cs.AI')
    cs_ai_recall = classification_report(
        test_labels, test_preds,
        target_names=le.classes_,
        output_dict=True
    )['cs.AI']['recall']

    print(f"\nObjectives:")
    print(f"  Test Accuracy >= 60%: {test_acc >= 0.60} ({test_acc*100:.2f}%)")
    print(f"  cs.AI Recall > 30%: {cs_ai_recall > 0.30} ({cs_ai_recall*100:.2f}%)")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='o')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='o')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('hybrid_training_history.png', dpi=150, bbox_inches='tight')
    print("\nOK Training history saved: hybrid_training_history.png")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
