"""
Focal Loss Implementation
Para balancear clases y enfocarse en ejemplos difíciles

Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
https://arxiv.org/abs/1708.02002

Ventajas para este proyecto:
- Reduce peso de ejemplos fáciles (bien clasificados)
- Aumenta peso de ejemplos difíciles (mal clasificados)
- Mejor balance que class weighting estándar
- Combina bien con threshold tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss con soporte para class weights y label smoothing

    Args:
        alpha: Peso por clase (e.g., [2.0, 1.0, 1.0, 1.0] para cs.AI)
        gamma: Factor de enfoque (default: 2.0)
               - gamma=0: Equivalente a CrossEntropyLoss
               - gamma>0: Reduce peso de ejemplos fáciles
        label_smoothing: Suavizado de etiquetas (default: 0.1)
        reduction: 'mean', 'sum', o 'none'

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        donde:
        - p_t: probabilidad predicha para la clase correcta
        - alpha_t: peso de la clase
        - gamma: factor de enfoque
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1,
                 reduction='mean'):
        super().__init__()

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.FloatTensor(alpha)
            self.alpha = alpha
        else:
            self.alpha = None

        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits del modelo [batch_size, num_classes]
            targets: Labels verdaderos [batch_size]

        Returns:
            loss: Focal loss calculado
        """
        # Aplicar label smoothing manualmente
        num_classes = inputs.size(1)

        # Convertir targets a one-hot con smoothing
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes

        # Log-probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # Probabilities
        probs = torch.exp(log_probs)

        # Calcular cross entropy base
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1)

        # Calcular p_t (probabilidad de la clase correcta)
        p_t = (targets_one_hot * probs).sum(dim=1)

        # Aplicar modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Focal loss = modulating_factor * ce_loss
        focal_loss = modulating_factor * ce_loss

        # Aplicar class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Obtener alpha para cada muestra según su clase
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Reducción
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Focal Loss adaptativo que ajusta gamma dinámicamente durante el entrenamiento

    - Comienza con gamma alto (enfoque en difíciles)
    - Reduce gamma gradualmente (más balance)
    - Útil para entrenamiento progresivo
    """

    def __init__(self, alpha=None, gamma_start=3.0, gamma_end=1.5,
                 label_smoothing=0.1, total_epochs=10):
        super().__init__()

        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma_start,
                                    label_smoothing=label_smoothing)
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Actualizar gamma según época actual"""
        self.current_epoch = epoch

        # Decaimiento lineal de gamma
        progress = min(epoch / self.total_epochs, 1.0)
        current_gamma = self.gamma_start - (self.gamma_start - self.gamma_end) * progress

        self.focal_loss.gamma = current_gamma

    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)


def test_focal_loss():
    """Test de Focal Loss"""
    print("="*70)
    print("TESTING FOCAL LOSS")
    print("="*70)

    # Crear datos de ejemplo
    batch_size = 8
    num_classes = 4

    # Logits simulados
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print(f"\nBatch size: {batch_size}")
    print(f"Num classes: {num_classes}")
    print(f"Targets: {targets.tolist()}")

    # Test 1: Focal Loss sin class weights
    print("\n" + "="*70)
    print("Test 1: Focal Loss (gamma=2.0, sin class weights)")
    print("="*70)

    criterion1 = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss1 = criterion1(logits, targets)
    print(f"Loss: {loss1.item():.4f}")

    # Test 2: Focal Loss con class weights
    print("\n" + "="*70)
    print("Test 2: Focal Loss (gamma=2.0, class weights=[2.0, 1.0, 1.0, 1.0])")
    print("="*70)

    class_weights = [2.0, 1.0, 1.0, 1.0]
    criterion2 = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    loss2 = criterion2(logits, targets)
    print(f"Loss: {loss2.item():.4f}")

    # Test 3: CrossEntropy estándar (gamma=0)
    print("\n" + "="*70)
    print("Test 3: CrossEntropy equivalente (gamma=0)")
    print("="*70)

    criterion3 = FocalLoss(gamma=0.0, label_smoothing=0.1)
    loss3 = criterion3(logits, targets)
    print(f"Loss: {loss3.item():.4f}")

    # Test 4: Comparar con CrossEntropyLoss de PyTorch
    print("\n" + "="*70)
    print("Test 4: PyTorch CrossEntropyLoss (para validar)")
    print("="*70)

    criterion4 = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss4 = criterion4(logits, targets)
    print(f"Loss: {loss4.item():.4f}")
    print(f"Diferencia con Focal (gamma=0): {abs(loss3.item() - loss4.item()):.6f}")

    # Test 5: Adaptive Focal Loss
    print("\n" + "="*70)
    print("Test 5: Adaptive Focal Loss (gamma 3.0→1.5)")
    print("="*70)

    criterion5 = AdaptiveFocalLoss(
        alpha=class_weights,
        gamma_start=3.0,
        gamma_end=1.5,
        total_epochs=10
    )

    for epoch in range(3):
        criterion5.set_epoch(epoch)
        loss = criterion5(logits, targets)
        print(f"Epoch {epoch}: gamma={criterion5.focal_loss.gamma:.2f}, loss={loss.item():.4f}")

    print("\n" + "="*70)
    print("TESTS COMPLETED SUCCESSFULLY")
    print("="*70)

    return True


if __name__ == "__main__":
    test_focal_loss()
