# train_scibert_optimized.py
"""
SciBERT con hiperparmetros optimizados para prevenir overfitting
Ajustes basados en observaciones de entrenamiento anterior:
- Dropout ms agresivo
- Learning rate reducido para fine-tuning
- Weight decay aumentado
- Early stopping ms estricto
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

from transformers import AutoTokenizer, AutoModel
from preprocessing_scibert import prepare_scibert_data
from sklearn.utils.class_weight import compute_class_weight


class OptimizedSciBERTClassifier(nn.Module):
    """
    SciBERT optimizado con regularizacin agresiva
    """
    def __init__(self, num_classes=4, dropout=0.5, freeze_bert_layers=0):
        super().__init__()

        # Cargar SciBERT pre-entrenado
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Opcionalmente congelar primeras N capas de BERT
        if freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Dropout despus de embeddings (prevenir overfitting temprano)
        self.embedding_dropout = nn.Dropout(0.1)

        # Attention para pooling de secuencias
        self.title_attention = nn.Linear(self.hidden_size, 1)
        self.abstract_attention = nn.Linear(self.hidden_size, 1)

        # Fusion layer con MAYOR regularizacin
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),  # GELU mejor que ReLU para transformers
            nn.Dropout(dropout),  # 0.5 default

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # Capa extra para ms capacidad pero ms regularizacin
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),  # Menos dropout en capa final

            nn.Linear(128, num_classes)
        )

        # Inicializacin
        self._init_weights()

    def _init_weights(self):
        """Inicializacin de pesos para capas no-BERT"""
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
        """Pooling con atencin sobre secuencia"""
        scores = attention_layer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)

    def forward(self, title_input_ids, title_attention_mask,
                abstract_input_ids, abstract_attention_mask):
        """Forward pass"""
        # Procesar ttulo con BERT
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


class OptimizedTrainer:
    def __init__(self, model, device, lr=5e-5, weight_decay=0.01, class_weights=None):
        self.model = model.to(device)
        self.device = device

        # Loss con label smoothing Y class weights para balancear clases
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1,  # Suavizar fronteras para clases superpuestas
            weight=class_weights   # Balancear clases desbalanceadas
        )

        # Optimizer con learning rates diferenciales y MAYOR weight decay
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

        # Scheduler con warmup
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

        from transformers import get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(self, loader):
        """Entrenar una poca"""
        self.model.train()
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

            # Mtricas
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

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            epoch_time = time.time() - start_time
            overfit_gap = train_acc - val_acc

            # Imprimir métricas
            print(f"Época {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Overfit: {overfit_gap:+.4f}")

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
                }, 'best_scibert_optimized.pth')
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping! Mejor época: {self.best_epoch + 1} (Val Acc: {self.best_val_acc:.4f})")
                break

        print(f"\nEntrenamiento completo - Mejor Val Acc: {self.best_val_acc:.4f} (época {self.best_epoch + 1})")

    def plot_history(self, save_path='scibert_optimized_history.png'):
        """Graficar historial"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss
        ax1.plot(epochs, self.train_losses, 'b-o', label='Train Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-o', label='Val Loss', alpha=0.8)
        ax1.axvline(x=self.best_epoch + 1, color='green', linestyle='--',
                   alpha=0.5, label=f'Best epoch: {self.best_epoch + 1}')
        ax1.set_xlabel('poca')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, self.train_accs, 'b-o', label='Train Acc', alpha=0.8)
        ax2.plot(epochs, self.val_accs, 'r-o', label='Val Acc', alpha=0.8)
        ax2.axhline(y=self.best_val_acc, color='green', linestyle='--',
                   label=f'Best Val: {self.best_val_acc:.3f}')
        ax2.set_xlabel('poca')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def compute_class_weights_from_dataset(dataset, num_classes):
    """
    Calcula pesos de clase basados en la distribucin del dataset
    Devuelve pesos ms altos para clases minoritarias
    """
    # Extraer todas las etiquetas del dataset
    all_labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    all_labels = np.array(all_labels)

    # Calcular pesos usando sklearn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=all_labels
    )

    return class_weights


def main():
    """Entrenamiento principal"""
    # Configuración V3.5 - Punto medio entre V2 y V3
    FREEZE_BERT_LAYERS = 6      # Punto medio: entre 4 (V3) y 8 (V2)
    DROPOUT = 0.45              # Punto medio: entre 0.4 (V3) y 0.5 (V2)
    BATCH_SIZE = 12             # Optimizado para M2
    EPOCHS = 10
    LR = 4e-5                   # Punto medio: entre 3e-5 (V2) y 5e-5 (V3)
    WEIGHT_DECAY = 0.02         # Punto medio: entre 0.01 (V3) y 0.05 (V2)
    PATIENCE = 3
    USE_CLASS_WEIGHTS = True
    NUM_WORKERS = 0

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    # M2 optimization: Set MPS fallback
    if device.type == "mps":
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Preparar datos
    train_dataset, val_dataset, test_dataset, tokenizer, le = prepare_scibert_data(
        use_light_model=False
    )

    # DataLoaders - Optimized for M2 (num_workers=0, no pin_memory)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=False, persistent_workers=False)

    # Calcular class weights si est activado
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights_from_dataset(train_dataset, num_classes=4)

    # Crear modelo
    model = OptimizedSciBERTClassifier(
        num_classes=4,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS
    )

    # Entrenar con class weights
    trainer = OptimizedTrainer(
        model, device,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights
    )
    trainer.train(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)

    # Graficar
    trainer.plot_history()

    # Evaluacin final en test
    checkpoint = torch.load('best_scibert_optimized.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_loss, test_acc, test_f1, test_labels, test_preds = trainer.validate(test_loader)

    # Resultados finales
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=le.classes_, digits=4)}")

    report = classification_report(test_labels, test_preds,
                                   target_names=le.classes_, digits=4,
                                   output_dict=True)

    # Matriz de confusin
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusin - SciBERT Optimizado\nTest Acc: {test_acc:.3f}',
              fontsize=14, pad=20)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Prediccin', fontsize=12)
    plt.tight_layout()
    plt.savefig('scibert_optimized_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Evaluación de objetivos
    cs_ai_recall = report['cs.AI']['recall']
    success = test_acc >= 0.60 and cs_ai_recall > 0.30

    print(f"\nObjetivo alcanzado: {'✓ SI' if success else '✗ NO'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
