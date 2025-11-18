"""
Evaluación Completa de Todas las Mejoras Implementadas

Este script evalúa y compara:
1. Baseline: V3.7 con threshold tuning (mejor anterior)
2. V4.0: V3.7 + Focal Loss
3. V4.0 + Multi-class threshold tuning
4. Ensemble: V2 + V3.7
5. Ensemble + Multi-class threshold tuning

Genera reportes comparativos y visualizaciones.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import pandas as pd
from torch.utils.data import DataLoader

from preprocessing_scibert import prepare_scibert_data
from train_scibert_optimized import OptimizedSciBERTClassifier
from threshold_optimizer import ThresholdOptimizer
from ensemble_predictor import create_ensemble_v2_v37


class ComprehensiveEvaluator:
    """
    Evaluador comprehensivo de todas las mejoras
    """

    def __init__(self, test_loader, label_encoder, device):
        self.test_loader = test_loader
        self.le = label_encoder
        self.device = device
        self.results = {}

    def evaluate_model(self, model, name, thresholds=None):
        """
        Evaluar un modelo individual

        Args:
            model: Modelo a evaluar
            name: Nombre del modelo
            thresholds: Thresholds por clase (None = argmax estándar)

        Returns:
            dict con métricas
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")

        if thresholds is not None:
            print(f"Thresholds: {thresholds}")

        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                title_ids = batch['title_input_ids'].to(self.device)
                title_mask = batch['title_attention_mask'].to(self.device)
                abstract_ids = batch['abstract_input_ids'].to(self.device)
                abstract_mask = batch['abstract_attention_mask'].to(self.device)
                labels = batch['label'].cpu().numpy()

                outputs = model(title_ids, title_mask, abstract_ids, abstract_mask)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(labels)

        all_probs = np.vstack(all_probs)
        all_labels = np.array(all_labels)

        # Aplicar thresholds si están especificados
        if thresholds is not None:
            predictions = self._apply_thresholds(all_probs, thresholds)
        else:
            predictions = all_probs.argmax(axis=1)

        # Calcular métricas
        metrics = self._compute_metrics(all_labels, predictions, all_probs)
        metrics['name'] = name
        metrics['thresholds'] = thresholds

        self.results[name] = metrics

        # Imprimir resultados
        self._print_metrics(metrics)

        return metrics

    def _apply_thresholds(self, probs, thresholds):
        """Aplicar thresholds multi-clase"""
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

    def _compute_metrics(self, labels, predictions, probs):
        """Calcular todas las métricas"""
        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average='weighted')
        f1_macro = f1_score(labels, predictions, average='macro')

        recall_per_class = recall_score(labels, predictions, average=None)
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)

        cs_ai_idx = list(self.le.classes_).index('cs.AI')
        cs_ai_recall = recall_per_class[cs_ai_idx]
        cs_ai_precision = precision_per_class[cs_ai_idx]

        # Objetivos
        gap_acc = abs(accuracy - 0.60)
        gap_cs_ai = max(0, 0.30 - cs_ai_recall)  # 0 si ya cumplió
        gap_total = gap_acc + gap_cs_ai

        objectives_met = {
            'accuracy_60': accuracy >= 0.60,
            'cs_ai_recall_30': cs_ai_recall > 0.30
        }

        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'recall_per_class': recall_per_class,
            'precision_per_class': precision_per_class,
            'cs_ai_recall': cs_ai_recall,
            'cs_ai_precision': cs_ai_precision,
            'gap_acc': gap_acc,
            'gap_cs_ai': gap_cs_ai,
            'gap_total': gap_total,
            'objectives_met': objectives_met,
            'predictions': predictions,
            'labels': labels
        }

    def _print_metrics(self, metrics):
        """Imprimir métricas"""
        print(f"\nAccuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")

        print(f"\nPer-class metrics:")
        for i, cls in enumerate(self.le.classes_):
            marker = " ★" if cls == 'cs.AI' else ""
            print(f"  {cls}: P={metrics['precision_per_class'][i]:.4f}, "
                  f"R={metrics['recall_per_class'][i]:.4f}{marker}")

        print(f"\nObjectives:")
        obj = metrics['objectives_met']
        print(f"  Accuracy >= 60%: {'✓' if obj['accuracy_60'] else '✗'} "
              f"({metrics['accuracy']*100:.2f}%, gap: {metrics['gap_acc']*100:+.2f}%)")
        print(f"  cs.AI Recall > 30%: {'✓' if obj['cs_ai_recall_30'] else '✗'} "
              f"({metrics['cs_ai_recall']*100:.2f}%, gap: {metrics['gap_cs_ai']*100:+.2f}%)")
        print(f"  Gap Total: {metrics['gap_total']:.4f}")

    def compare_all(self, baseline_name='V3.7+TT'):
        """
        Comparar todos los modelos evaluados

        Args:
            baseline_name: Nombre del modelo baseline
        """
        print(f"\n{'='*70}")
        print("COMPREHENSIVE COMPARISON")
        print(f"{'='*70}")

        if baseline_name not in self.results:
            print(f"Baseline '{baseline_name}' not found!")
            return

        baseline = self.results[baseline_name]

        # Crear tabla comparativa
        data = []
        for name, metrics in self.results.items():
            improvement_acc = (metrics['accuracy'] - baseline['accuracy']) * 100
            improvement_cs_ai = (metrics['cs_ai_recall'] - baseline['cs_ai_recall']) * 100
            improvement_gap = (baseline['gap_total'] - metrics['gap_total']) * 100

            data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Δ Acc': f"{improvement_acc:+.2f}%",
                'cs.AI Recall': f"{metrics['cs_ai_recall']*100:.2f}%",
                'Δ cs.AI': f"{improvement_cs_ai:+.2f}%",
                'Gap Total': f"{metrics['gap_total']*100:.2f}%",
                'Δ Gap': f"{improvement_gap:+.2f}%",
                'Acc ≥60%': '✓' if metrics['objectives_met']['accuracy_60'] else '✗',
                'cs.AI >30%': '✓' if metrics['objectives_met']['cs_ai_recall_30'] else '✗'
            })

        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))

        # Guardar a CSV
        df.to_csv('model_comparison.csv', index=False)
        print(f"\n✓ Comparison saved to model_comparison.csv")

        # Visualización
        self._plot_comparison()

        return df

    def _plot_comparison(self):
        """Crear visualizaciones comparativas"""
        if len(self.results) < 2:
            print("Need at least 2 models to compare")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        cs_ai_recalls = [self.results[m]['cs_ai_recall'] for m in models]
        f1_scores = [self.results[m]['f1_weighted'] for m in models]
        gap_totals = [self.results[m]['gap_total'] for m in models]

        # 1. Accuracy comparison
        ax = axes[0, 0]
        bars = ax.bar(models, accuracies, color='steelblue', alpha=0.7)
        ax.axhline(y=0.60, color='red', linestyle='--', label='Target: 60%', linewidth=2)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=10)

        # 2. cs.AI Recall comparison
        ax = axes[0, 1]
        bars = ax.bar(models, cs_ai_recalls, color='forestgreen', alpha=0.7)
        ax.axhline(y=0.30, color='red', linestyle='--', label='Target: 30%', linewidth=2)
        ax.set_ylabel('cs.AI Recall', fontsize=12)
        ax.set_title('cs.AI Recall Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, recall in zip(bars, cs_ai_recalls):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{recall*100:.2f}%', ha='center', va='bottom', fontsize=10)

        # 3. F1 Score comparison
        ax = axes[1, 0]
        bars = ax.bar(models, f1_scores, color='coral', alpha=0.7)
        ax.set_ylabel('F1 Score (Weighted)', fontsize=12)
        ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{f1:.4f}', ha='center', va='bottom', fontsize=10)

        # 4. Gap Total comparison (lower is better)
        ax = axes[1, 1]
        bars = ax.bar(models, gap_totals, color='purple', alpha=0.7)
        ax.set_ylabel('Gap Total (lower is better)', fontsize=12)
        ax.set_title('Gap Total Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, gap in zip(bars, gap_totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{gap:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Comparison plot saved to model_comparison.png")


def main():
    """
    Evaluación completa de todas las mejoras

    Ejecutar después de entrenar los modelos necesarios.
    """
    print("="*70)
    print("COMPREHENSIVE EVALUATION - All Improvements")
    print("="*70)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Preparar datos
    print("\nLoading data...")
    _, _, test_dataset, tokenizer, le = prepare_scibert_data(use_light_model=False)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"✓ Test set: {len(test_dataset)} samples")

    # Crear evaluador
    evaluator = ComprehensiveEvaluator(test_loader, le, device)

    # =========================================================================
    # EVALUACIÓN 1: Baseline V3.7 + Threshold Tuning
    # =========================================================================
    print("\n" + "="*70)
    print("1. BASELINE: V3.7 + Threshold Tuning (Best Previous)")
    print("="*70)

    try:
        # Cargar modelo V3.7
        model_v37 = OptimizedSciBERTClassifier(
            num_classes=4,
            dropout=0.35,
            freeze_bert_layers=3
        )
        checkpoint = torch.load('best_scibert_v3.7_final.pth', map_location=device)
        model_v37.load_state_dict(checkpoint['model_state_dict'])
        model_v37.to(device)

        # Evaluar con threshold tuning (0.40 para cs.AI)
        thresholds_v37 = np.array([0.40, 0.5, 0.5, 0.5])
        evaluator.evaluate_model(model_v37, 'V3.7+TT', thresholds=thresholds_v37)

    except FileNotFoundError:
        print("⚠️  Model V3.7 not found. Train it first with train_scibert_optimized.py")

    # =========================================================================
    # EVALUACIÓN 2: V4.0 con Focal Loss
    # =========================================================================
    print("\n" + "="*70)
    print("2. V4.0: Focal Loss")
    print("="*70)

    try:
        model_v4 = OptimizedSciBERTClassifier(
            num_classes=4,
            dropout=0.35,
            freeze_bert_layers=3
        )
        checkpoint = torch.load('best_scibert_v4_focal.pth', map_location=device)
        model_v4.load_state_dict(checkpoint['model_state_dict'])
        model_v4.to(device)

        # Evaluar sin threshold tuning
        evaluator.evaluate_model(model_v4, 'V4.0 (Focal)', thresholds=None)

        # Evaluar con threshold tuning básico
        evaluator.evaluate_model(model_v4, 'V4.0+TT (basic)', thresholds=thresholds_v37)

    except FileNotFoundError:
        print("⚠️  Model V4.0 not found. Train it first with train_scibert_v4_focal.py")

    # =========================================================================
    # EVALUACIÓN 3: V4.0 + Multi-class Threshold Optimization
    # =========================================================================
    print("\n" + "="*70)
    print("3. V4.0 + Multi-class Threshold Optimization")
    print("="*70)

    try:
        # Optimizar thresholds para V4.0
        val_dataset = prepare_scibert_data(use_light_model=False)[1]
        val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False,
                               num_workers=0, pin_memory=False)

        optimizer = ThresholdOptimizer(model_v4, val_loader, device, le)

        print("\nOptimizing thresholds with Greedy Search...")
        best_thresholds, _, _ = optimizer.greedy_search(
            threshold_range=(0.30, 0.60),
            step=0.05,
            optimize_metric='gap_total',
            minimize=True
        )

        # Evaluar con thresholds optimizados
        evaluator.evaluate_model(model_v4, 'V4.0+TT (optimized)',
                                thresholds=best_thresholds)

    except Exception as e:
        print(f"⚠️  Error optimizing thresholds: {e}")

    # =========================================================================
    # COMPARACIÓN FINAL
    # =========================================================================
    if len(evaluator.results) > 0:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)

        comparison_df = evaluator.compare_all(baseline_name='V3.7+TT')

        # Encontrar mejor modelo
        best_model = min(evaluator.results.items(),
                        key=lambda x: x[1]['gap_total'])

        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model[0]}")
        print(f"{'='*70}")
        print(f"Accuracy: {best_model[1]['accuracy']*100:.2f}%")
        print(f"cs.AI Recall: {best_model[1]['cs_ai_recall']*100:.2f}%")
        print(f"Gap Total: {best_model[1]['gap_total']:.4f}")
        print(f"{'='*70}\n")

    else:
        print("\n⚠️  No models were evaluated. Train models first.")


if __name__ == "__main__":
    main()
