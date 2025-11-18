"""
Multi-Class Threshold Optimizer
Encuentra los thresholds óptimos para cada clase

Mejora sobre threshold_tuning.py original:
- Original: Solo optimiza cs.AI (threshold único)
- Este: Optimiza todas las clases (4 thresholds)

Estrategias:
1. Grid Search: Búsqueda exhaustiva en espacio discretizado
2. Bayesian Optimization: Búsqueda inteligente (opcional)
3. Greedy Search: Optimización clase por clase
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import pickle
from tqdm import tqdm
import itertools


class ThresholdOptimizer:
    """
    Optimizador de thresholds multi-clase

    Args:
        model: Modelo entrenado
        val_loader: DataLoader de validación
        device: Device (cuda/mps/cpu)
        label_encoder: LabelEncoder para decodificar clases
    """

    def __init__(self, model, val_loader, device, label_encoder):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.le = label_encoder
        self.num_classes = len(label_encoder.classes_)

        # Extraer todas las predicciones y probabilidades
        print("Extracting predictions from validation set...")
        self.probs_all, self.labels_all = self._get_all_predictions()
        print(f"✓ Extracted {len(self.labels_all)} samples")

    def _get_all_predictions(self):
        """Obtener todas las predicciones y probabilidades del val set"""
        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Extracting predictions"):
                title_ids = batch['title_input_ids'].to(self.device)
                title_mask = batch['title_attention_mask'].to(self.device)
                abstract_ids = batch['abstract_input_ids'].to(self.device)
                abstract_mask = batch['abstract_attention_mask'].to(self.device)
                labels = batch['label'].cpu().numpy()

                # Forward pass
                outputs = self.model(title_ids, title_mask, abstract_ids, abstract_mask)

                # Handle tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Probabilidades
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_labels.extend(labels)

        all_probs = np.vstack(all_probs)  # [num_samples, num_classes]
        all_labels = np.array(all_labels)  # [num_samples]

        return all_probs, all_labels

    def apply_thresholds(self, probs, thresholds):
        """
        Aplicar thresholds multi-clase

        Estrategia:
        1. Para cada muestra, identificar clases que superan su threshold
        2. Si hay candidatos, elegir el de mayor probabilidad
        3. Si no hay candidatos, usar argmax estándar

        Args:
            probs: Probabilidades [num_samples, num_classes]
            thresholds: Thresholds por clase [num_classes]

        Returns:
            predictions: Predicciones [num_samples]
        """
        num_samples = probs.shape[0]
        predictions = np.zeros(num_samples, dtype=np.int64)

        for i in range(num_samples):
            sample_probs = probs[i]

            # Buscar candidatos que superen threshold
            candidates = []
            for class_idx, (prob, threshold) in enumerate(zip(sample_probs, thresholds)):
                if prob >= threshold:
                    candidates.append((class_idx, prob))

            # Elegir mejor candidato o argmax
            if candidates:
                predictions[i] = max(candidates, key=lambda x: x[1])[0]
            else:
                predictions[i] = sample_probs.argmax()

        return predictions

    def evaluate_thresholds(self, thresholds, verbose=False):
        """
        Evaluar un conjunto de thresholds

        Args:
            thresholds: Array de thresholds [num_classes]
            verbose: Si True, imprime métricas detalladas

        Returns:
            metrics: Dict con accuracy, f1, recall por clase, etc.
        """
        # Aplicar thresholds
        predictions = self.apply_thresholds(self.probs_all, thresholds)

        # Calcular métricas
        accuracy = accuracy_score(self.labels_all, predictions)
        f1_weighted = f1_score(self.labels_all, predictions, average='weighted')
        f1_macro = f1_score(self.labels_all, predictions, average='macro')

        # Recall por clase
        recall_per_class = recall_score(self.labels_all, predictions, average=None)

        # Precision por clase
        precision_per_class = precision_score(self.labels_all, predictions,
                                               average=None, zero_division=0)

        # Métrica especial: Gap total
        # Gap = |accuracy - target_acc| + |cs_ai_recall - target_cs_ai_recall|
        cs_ai_idx = list(self.le.classes_).index('cs.AI')
        cs_ai_recall = recall_per_class[cs_ai_idx]

        gap_acc = abs(accuracy - 0.60)  # Target: 60%
        gap_cs_ai = abs(cs_ai_recall - 0.30)  # Target: 30%
        gap_total = gap_acc + gap_cs_ai

        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'recall_per_class': recall_per_class,
            'precision_per_class': precision_per_class,
            'cs_ai_recall': cs_ai_recall,
            'gap_total': gap_total,
            'gap_acc': gap_acc,
            'gap_cs_ai': gap_cs_ai,
            'thresholds': thresholds
        }

        if verbose:
            print(f"\nThresholds: {thresholds}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"F1 (weighted): {f1_weighted:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
            print(f"\nPer-class metrics:")
            for i, cls in enumerate(self.le.classes_):
                print(f"  {cls}: P={precision_per_class[i]:.4f}, "
                      f"R={recall_per_class[i]:.4f}")
            print(f"\nGap Total: {gap_total:.4f}")
            print(f"  Gap Acc: {gap_acc:.4f} (target: 0.60)")
            print(f"  Gap cs.AI: {gap_cs_ai:.4f} (target: 0.30)")

        return metrics

    def grid_search(self, threshold_range=(0.30, 0.60), step=0.05,
                   optimize_metric='gap_total', minimize=True):
        """
        Grid search exhaustivo sobre espacio de thresholds

        Args:
            threshold_range: Rango de thresholds a explorar (min, max)
            step: Paso del grid
            optimize_metric: Métrica a optimizar ('gap_total', 'accuracy', 'f1_weighted')
            minimize: Si True, minimiza métrica; si False, maximiza

        Returns:
            best_thresholds, best_metrics, all_results
        """
        # Generar grid
        threshold_values = np.arange(threshold_range[0], threshold_range[1] + step, step)

        print(f"\n{'='*70}")
        print(f"GRID SEARCH - Multi-Class Threshold Optimization")
        print(f"{'='*70}")
        print(f"Threshold range: {threshold_range}")
        print(f"Step: {step}")
        print(f"Grid size per class: {len(threshold_values)}")
        print(f"Total combinations: {len(threshold_values)**self.num_classes:,}")
        print(f"Optimize metric: {optimize_metric} ({'minimize' if minimize else 'maximize'})")

        # ADVERTENCIA: Grid search completo es muy costoso para 4 clases
        # Con step=0.05 y range 0.30-0.60: (7)^4 = 2,401 combinaciones
        # Con step=0.10 y range 0.30-0.60: (4)^4 = 256 combinaciones

        total_combinations = len(threshold_values)**self.num_classes

        if total_combinations > 10000:
            print(f"\n⚠️  WARNING: {total_combinations:,} combinations is very expensive!")
            print(f"Consider using greedy_search() or reducing step size.")
            return None, None, None

        # Grid search
        all_results = []
        best_metric_value = float('inf') if minimize else float('-inf')
        best_thresholds = None
        best_metrics = None

        # Generar todas las combinaciones
        all_combinations = list(itertools.product(threshold_values, repeat=self.num_classes))

        print(f"\nSearching {len(all_combinations):,} combinations...")

        for thresholds in tqdm(all_combinations):
            thresholds = np.array(thresholds)

            # Evaluar
            metrics = self.evaluate_thresholds(thresholds, verbose=False)
            all_results.append(metrics)

            # Actualizar mejor
            metric_value = metrics[optimize_metric]

            if minimize:
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_thresholds = thresholds
                    best_metrics = metrics
            else:
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_thresholds = thresholds
                    best_metrics = metrics

        # Resultados
        print(f"\n{'='*70}")
        print("BEST THRESHOLDS FOUND")
        print(f"{'='*70}")

        self.evaluate_thresholds(best_thresholds, verbose=True)

        return best_thresholds, best_metrics, all_results

    def greedy_search(self, initial_thresholds=None, threshold_range=(0.30, 0.60),
                     step=0.05, optimize_metric='gap_total', minimize=True):
        """
        Greedy search: Optimiza una clase a la vez

        Mucho más rápido que grid search:
        - Grid: O(n^k) donde n=valores, k=clases
        - Greedy: O(k*n) = lineal

        Args:
            initial_thresholds: Thresholds iniciales (default: [0.5, 0.5, 0.5, 0.5])
            threshold_range: Rango a explorar
            step: Paso
            optimize_metric: Métrica a optimizar
            minimize: Si True, minimiza; si False, maximiza

        Returns:
            best_thresholds, best_metrics, history
        """
        print(f"\n{'='*70}")
        print(f"GREEDY SEARCH - Multi-Class Threshold Optimization")
        print(f"{'='*70}")
        print(f"Threshold range: {threshold_range}")
        print(f"Step: {step}")
        print(f"Optimize metric: {optimize_metric} ({'minimize' if minimize else 'maximize'})")

        # Inicializar
        if initial_thresholds is None:
            current_thresholds = np.array([0.5] * self.num_classes)
        else:
            current_thresholds = np.array(initial_thresholds)

        threshold_values = np.arange(threshold_range[0], threshold_range[1] + step, step)

        # Evaluar inicial
        current_metrics = self.evaluate_thresholds(current_thresholds, verbose=False)
        best_metric_value = current_metrics[optimize_metric]

        history = [current_metrics]

        print(f"\nInitial thresholds: {current_thresholds}")
        print(f"Initial {optimize_metric}: {best_metric_value:.4f}")

        # Optimizar cada clase secuencialmente
        for class_idx in range(self.num_classes):
            class_name = self.le.classes_[class_idx]
            print(f"\n--- Optimizing {class_name} (class {class_idx}) ---")

            best_threshold_for_class = current_thresholds[class_idx]

            for threshold in threshold_values:
                # Probar este threshold para la clase actual
                test_thresholds = current_thresholds.copy()
                test_thresholds[class_idx] = threshold

                # Evaluar
                metrics = self.evaluate_thresholds(test_thresholds, verbose=False)
                metric_value = metrics[optimize_metric]

                # Actualizar si es mejor
                improved = (minimize and metric_value < best_metric_value) or \
                          (not minimize and metric_value > best_metric_value)

                if improved:
                    best_metric_value = metric_value
                    best_threshold_for_class = threshold
                    current_metrics = metrics

            # Actualizar threshold de esta clase
            current_thresholds[class_idx] = best_threshold_for_class
            history.append(current_metrics)

            print(f"  Best threshold for {class_name}: {best_threshold_for_class:.2f}")
            print(f"  {optimize_metric}: {best_metric_value:.4f}")

        # Resultados finales
        print(f"\n{'='*70}")
        print("BEST THRESHOLDS FOUND (Greedy)")
        print(f"{'='*70}")

        self.evaluate_thresholds(current_thresholds, verbose=True)

        return current_thresholds, current_metrics, history

    def class_priority_search(self, priority_order=None, threshold_range=(0.30, 0.60),
                             step=0.05, optimize_metric='gap_total', minimize=True):
        """
        Similar a greedy pero con orden de prioridad específico

        Args:
            priority_order: Orden de clases a optimizar (default: cs.AI primero)
            threshold_range: Rango
            step: Paso
            optimize_metric: Métrica
            minimize: Minimizar o maximizar

        Returns:
            best_thresholds, best_metrics, history
        """
        # Default: Optimizar cs.AI primero (más importante)
        if priority_order is None:
            cs_ai_idx = list(self.le.classes_).index('cs.AI')
            other_indices = [i for i in range(self.num_classes) if i != cs_ai_idx]
            priority_order = [cs_ai_idx] + other_indices

        print(f"\nPriority order: {[self.le.classes_[i] for i in priority_order]}")

        # Usar greedy search con orden personalizado
        initial_thresholds = np.array([0.5] * self.num_classes)

        threshold_values = np.arange(threshold_range[0], threshold_range[1] + step, step)

        current_thresholds = initial_thresholds.copy()
        current_metrics = self.evaluate_thresholds(current_thresholds, verbose=False)
        best_metric_value = current_metrics[optimize_metric]

        history = [current_metrics]

        # Optimizar en orden de prioridad
        for class_idx in priority_order:
            class_name = self.le.classes_[class_idx]
            print(f"\n--- Optimizing {class_name} (priority) ---")

            best_threshold_for_class = current_thresholds[class_idx]

            for threshold in threshold_values:
                test_thresholds = current_thresholds.copy()
                test_thresholds[class_idx] = threshold

                metrics = self.evaluate_thresholds(test_thresholds, verbose=False)
                metric_value = metrics[optimize_metric]

                improved = (minimize and metric_value < best_metric_value) or \
                          (not minimize and metric_value > best_metric_value)

                if improved:
                    best_metric_value = metric_value
                    best_threshold_for_class = threshold
                    current_metrics = metrics

            current_thresholds[class_idx] = best_threshold_for_class
            history.append(current_metrics)

            print(f"  Best threshold: {best_threshold_for_class:.2f}")
            print(f"  {optimize_metric}: {best_metric_value:.4f}")

        return current_thresholds, current_metrics, history


def main():
    """Ejemplo de uso del Threshold Optimizer"""
    print("="*70)
    print("THRESHOLD OPTIMIZER - Demo")
    print("="*70)

    print("\nThis optimizer finds the best thresholds for each class.")
    print("\nStrategies:")
    print("  1. Grid Search: Exhaustive (slow for many classes)")
    print("  2. Greedy Search: Sequential optimization (fast)")
    print("  3. Priority Search: Greedy with custom order")

    print("\n" + "="*70)
    print("Example usage:")
    print("="*70)

    code = """
# Crear optimizer
optimizer = ThresholdOptimizer(
    model=trained_model,
    val_loader=val_loader,
    device=device,
    label_encoder=label_encoder
)

# Opción 1: Greedy search (recomendado)
best_thresholds, metrics, history = optimizer.greedy_search(
    threshold_range=(0.30, 0.60),
    step=0.05,
    optimize_metric='gap_total',
    minimize=True
)

# Opción 2: Grid search (solo para step grande)
best_thresholds, metrics, results = optimizer.grid_search(
    threshold_range=(0.30, 0.60),
    step=0.10,  # Step grande para reducir combinaciones
    optimize_metric='gap_total',
    minimize=True
)

# Usar thresholds optimizados en predictor
predictor = OptimizedPredictor(thresholds=best_thresholds)
"""

    print(code)


if __name__ == "__main__":
    main()
