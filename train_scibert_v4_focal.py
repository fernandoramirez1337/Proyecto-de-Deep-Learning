"""
Training Script V4.0 - SciBERT con Focal Loss

MEJORAS IMPLEMENTADAS:
1. ✅ Focal Loss en lugar de CrossEntropyLoss
2. ✅ Class weighting optimizado (cs.AI x2.0)
3. ✅ Regularización balanceada
4. ✅ Early stopping mejorado

CONFIGURACIÓN V4.0:
- Base: V3.7 (mejor modelo anterior)
- Nueva loss: Focal Loss (gamma=2.0)
- Class weights: [2.0, 1.0, 1.0, 1.0]
- Dropout: 0.35
- Freeze layers: 3
- Learning rate: 5e-5
- Weight decay: 0.01

OBJETIVO:
- Test Accuracy: >= 60% (actual V3.7+TT: 56.17%)
- cs.AI Recall: > 30% (actual V3.7+TT: 36.22%)
- Gap total: < 3.83% (actual mejor)

ESPERANZA DE MEJORA:
- Focal Loss: +2-3% accuracy
- Total esperado: 58-59% accuracy manteniendo cs.AI recall
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import time

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from preprocessing_scibert import prepare_scibert_data
from focal_loss import FocalLoss, AdaptiveFocalLoss


class OptimizedSciBERTClassifier(nn.Module):
    """
    SciBERT optimizado (arquitectura V3.7)
    Compatible con Focal Loss
    """
    def __init__(self, num_classes=4, dropout=0.35, freeze_bert_layers=3):
        super().__init__()

        # Cargar SciBERT pre-entrenado
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Congelar primeras N capas de BERT
        if freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Dropout después de embeddings
        self.embedding_dropout = nn.Dropout(0.1)

        # Attention para pooling de secuencias
        self.title_attention = nn.Linear(self.hidden_size, 1)
        self.abstract_attention = nn.Linear(self.hidden_size, 1)

        # Fusion layer (misma arquitectura V3.7)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(128, num_classes)
        )

        # Inicialización
        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos para capas no-BERT"""
        for module in [self.title_attention, self.abstract_attention]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_layer, attention_mask):
        """Pooling con atención sobre secuencia"""
        scores = attention_layer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)

    def forward(self, title_input_ids, title_attention_mask,
                abstract_input_ids, abstract_attention_mask):
        """Forward pass"""
        # Procesar título con BERT
        title_outputs = self.bert(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask
        )
        title_hidden = self.embedding_dropout(title_outputs.last_hidden_state)

        # Procesar abstract con BERT
        abstract_outputs = self.bert(
            input_ids=abstract_input_ids,
            attention_mask=abstract_attention_mask
        )
        abstract_hidden = self.embedding_dropout(abstract_outputs.last_hidden_state)

        # Attention pooling
        title_pooled, title_attn_weights = self.attention_pool(
            title_hidden, self.title_attention, title_attention_mask
        )

        abstract_pooled, abstract_attn_weights = self.attention_pool(
            abstract_hidden, self.abstract_attention, abstract_attention_mask
        )

        # Concatenar y clasificar
        combined = torch.cat([title_pooled, abstract_pooled], dim=1)
        output = self.fusion(combined)

        return output, title_attn_weights, abstract_attn_weights


class FocalLossTrainer:
    """
    Trainer con Focal Loss
    Basado en OptimizedTrainer pero con Focal Loss
    """
    def __init__(self, model, device, lr=5e-5, weight_decay=0.01,
                 class_weights=None, focal_gamma=2.0, label_smoothing=0.1,
                 use_adaptive_focal=False):
        self.model = model.to(device)
        self.device = device

        # Loss: Focal Loss
        if use_adaptive_focal:
            print(f"Using Adaptive Focal Loss (gamma: 3.0→1.5)")
            self.criterion = AdaptiveFocalLoss(
                alpha=class_weights,
                gamma_start=3.0,
                gamma_end=1.5,
                label_smoothing=label_smoothing,
                total_epochs=10
            )
            self.adaptive_loss = True
        else:
            print(f"Using Focal Loss (gamma={focal_gamma})")
            if class_weights is not None:
                class_weights_tensor = torch.FloatTensor(class_weights).to(device)
            else:
                class_weights_tensor = None

            self.criterion = FocalLoss(
                alpha=class_weights_tensor,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
            self.adaptive_loss = False

        # Optimizer con learning rates diferenciales
        bert_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bert' in name and 'encoder' in name:
                    bert_params.append(param)
                else:
                    classifier_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': bert_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': lr * 5, 'weight_decay': weight_decay * 2}
        ])

        # Scheduler
        self.scheduler = None

        # Historial
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.best_epoch = 0

    def setup_scheduler(self, num_training_steps, num_warmup_steps=None):
        """Configurar scheduler con warmup"""
        if num_warmup_steps is None:
            num_warmup_steps = num_training_steps // 10

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(self, loader, epoch=0):
        """Entrenar una época"""
        self.model.train()

        # Actualizar gamma si es adaptivo
        if self.adaptive_loss:
            self.criterion.set_epoch(epoch)
            print(f"  Focal gamma: {self.criterion.focal_loss.gamma:.2f}")

        total_loss = 0
        all_preds = []
        all_labels = []

        progress = tqdm(loader, desc="Training")
        for batch in progress:
            title_ids = batch['title_input_ids'].to(self.device)
            title_mask = batch['title_attention_mask'].to(self.device)
            abstract_ids = batch['abstract_input_ids'].to(self.device)
            abstract_mask = batch['abstract_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            outputs, _, _ = self.model(title_ids, title_mask, abstract_ids, abstract_mask)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Métricas
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            current_lr = self.optimizer.param_groups[0]['lr']
            progress.set_postfix({
                'loss': f'{loss.item():.3f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def validate(self, loader):
        """Validar modelo"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                title_ids = batch['title_input_ids'].to(self.device)
                title_mask = batch['title_attention_mask'].to(self.device)
                abstract_ids = batch['abstract_input_ids'].to(self.device)
                abstract_mask = batch['abstract_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs, _, _ = self.model(title_ids, title_mask, abstract_ids, abstract_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1, all_labels, all_preds

    def train(self, train_loader, val_loader, epochs=10, patience=3):
        """Loop de entrenamiento con early stopping"""

        # Configurar scheduler
        num_training_steps = epochs * len(train_loader)
        self.setup_scheduler(num_training_steps)

        print(f"\n{'='*70}")
        print(f"TRAINING V4.0 - Focal Loss")
        print(f"{'='*70}")
        print(f"Total epochs: {epochs}")
        print(f"Patience: {patience}")
        print(f"Training steps: {num_training_steps}")
        print(f"{'='*70}\n")

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch=epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            epoch_time = time.time() - start_time
            overfit_gap = train_acc - val_acc

            # Imprimir métricas
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} ({val_acc*100:.2f}%), F1={val_f1:.4f}")
            print(f"  Gap:   {overfit_gap:+.4f} ({overfit_gap*100:+.2f}%)")

            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, 'best_scibert_v4_focal.pth')
                print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{patience}")

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered!")
                print(f"Best epoch: {self.best_epoch + 1}")
                print(f"Best Val Acc: {self.best_val_acc:.4f} ({self.best_val_acc*100:.2f}%)")
                print(f"{'='*70}")
                break

        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Best Val Acc: {self.best_val_acc:.4f} (epoch {self.best_epoch + 1})")
        print(f"{'='*70}\n")

    def plot_history(self, save_path='scibert_v4_focal_history.png'):
        """Graficar historial"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-o', label='Val Loss', alpha=0.8)
        ax1.axvline(x=self.best_epoch + 1, color='green', linestyle='--',
                   alpha=0.5, label=f'Best epoch: {self.best_epoch + 1}')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (Focal)')
        ax1.set_title('Training vs Validation Loss (Focal)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, self.train_accs, 'b-o', label='Train Acc', alpha=0.8)
        ax2.plot(epochs, self.val_accs, 'r-o', label='Val Acc', alpha=0.8)
        ax2.axhline(y=self.best_val_acc, color='green', linestyle='--',
                   label=f'Best Val: {self.best_val_acc:.3f}')
        ax2.axhline(y=0.60, color='orange', linestyle=':', label='Target: 60%')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ History plot saved to {save_path}")


def main():
    """Entrenamiento principal V4.0"""

    # ============================================================================
    # CONFIGURACIÓN V4.0 - FOCAL LOSS
    # ============================================================================
    VERSION = "V4.0"
    STRATEGY = "Focal Loss (gamma=2.0) + Class Weighting (cs.AI x2.0)"

    # Hiperparámetros (base: V3.7)
    FREEZE_BERT_LAYERS = 3          # Mantener V3.7
    DROPOUT = 0.35                  # Mantener V3.7
    BATCH_SIZE = 12                 # M2 optimizado
    EPOCHS = 10
    LR = 5e-5                       # Mantener V3.7
    WEIGHT_DECAY = 0.01             # Mantener V3.7
    PATIENCE = 3
    NUM_WORKERS = 0

    # Focal Loss específico
    FOCAL_GAMMA = 2.0               # Parámetro de enfoque
    LABEL_SMOOTHING = 0.1           # Suavizado
    CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]  # cs.AI x2.0 (como V3.7)
    USE_ADAPTIVE_FOCAL = False      # False: gamma fijo, True: gamma adaptativo

    print(f"{'='*70}")
    print(f"TRAINING SCIBERT - {VERSION}")
    print(f"{'='*70}")
    print(f"Strategy: {STRATEGY}")
    print(f"\nHyperparameters:")
    print(f"  Freeze layers: {FREEZE_BERT_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Focal gamma: {FOCAL_GAMMA}")
    print(f"  Class weights: {CLASS_WEIGHTS}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Adaptive focal: {USE_ADAPTIVE_FOCAL}")
    print(f"{'='*70}\n")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    # M2 optimization
    if device.type == "mps":
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print(f"Device: {device} (M2 optimized)")
    else:
        print(f"Device: {device}")

    # Preparar datos
    print("\nPreparing data...")
    train_dataset, val_dataset, test_dataset, tokenizer, le = prepare_scibert_data(
        use_light_model=False
    )

    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    print(f"✓ Classes: {list(le.classes_)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=False, persistent_workers=False)

    # Crear modelo
    print("\nCreating model...")
    model = OptimizedSciBERTClassifier(
        num_classes=4,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS
    )
    print(f"✓ Model created")

    # Entrenar con Focal Loss
    print("\nInitializing trainer...")
    trainer = FocalLossTrainer(
        model, device,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        class_weights=CLASS_WEIGHTS,
        focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        use_adaptive_focal=USE_ADAPTIVE_FOCAL
    )

    # Train
    trainer.train(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)

    # Graficar
    trainer.plot_history()

    # Evaluación final en test
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)

    checkpoint = torch.load('best_scibert_v4_focal.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_loss, test_acc, test_f1, test_labels, test_preds = trainer.validate(test_loader)

    # Resultados finales
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {VERSION}")
    print(f"{'='*70}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=le.classes_, digits=4))

    report = classification_report(test_labels, test_preds,
                                   target_names=le.classes_, digits=4,
                                   output_dict=True)

    # Matriz de confusión
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - SciBERT {VERSION}\nTest Acc: {test_acc:.3f} | Focal Loss',
              fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig('scibert_v4_focal_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved")

    # Evaluación de objetivos
    cs_ai_recall = report['cs.AI']['recall']
    cs_ai_target_met = cs_ai_recall > 0.30
    acc_target_met = test_acc >= 0.60

    gap_acc = abs(test_acc - 0.60)
    gap_cs_ai = abs(cs_ai_recall - 0.30) if not cs_ai_target_met else 0
    gap_total = gap_acc + gap_cs_ai

    print(f"\n{'='*70}")
    print(f"OBJECTIVES EVALUATION")
    print(f"{'='*70}")
    print(f"Test Accuracy >= 60%: {'✓ YES' if acc_target_met else '✗ NO'} ({test_acc*100:.2f}%, gap: {gap_acc*100:+.2f}%)")
    print(f"cs.AI Recall > 30%:   {'✓ YES' if cs_ai_target_met else '✗ NO'} ({cs_ai_recall*100:.2f}%, gap: {gap_cs_ai*100:+.2f}%)")
    print(f"Gap Total: {gap_total:.4f}")

    # Comparación con V3.7+TT
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH V3.7+TT (Best Previous)")
    print(f"{'='*70}")
    print(f"                    V3.7+TT    {VERSION}      Improvement")
    print(f"Test Accuracy:      56.17%     {test_acc*100:.2f}%     {(test_acc-0.5617)*100:+.2f}%")
    print(f"cs.AI Recall:       36.22%     {cs_ai_recall*100:.2f}%     {(cs_ai_recall-0.3622)*100:+.2f}%")
    print(f"Gap Total:          3.83%      {gap_total*100:.2f}%      {(gap_total-0.0383)*100:+.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
