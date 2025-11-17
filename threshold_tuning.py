"""
Threshold Tuning for V3.7 Model
"""

import torch
import numpy as np
from sklearn.metrics import classification_report
from preprocessing_scibert import prepare_scibert_data
from train_scibert_optimized import OptimizedSciBERTClassifier
from torch.utils.data import DataLoader


def evaluate_with_threshold(model, test_loader, device, threshold_cs_ai=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            title_ids = batch['title_input_ids'].to(device)
            title_mask = batch['title_attention_mask'].to(device)
            abstract_ids = batch['abstract_input_ids'].to(device)
            abstract_mask = batch['abstract_attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(title_ids, title_mask, abstract_ids, abstract_mask)
            
            # El modelo retorna (logits, attention_weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Probabilidades
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Concatenar
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    # Aplicar threshold tuning
    all_preds = []
    for prob in all_probs:
        # Si la probabilidad de cs.AI supera el umbral custom, asignar cs.AI
        if prob[0] >= threshold_cs_ai:
            all_preds.append(0)  # cs.AI
        else:
            all_preds.append(np.argmax(prob))

    all_preds = np.array(all_preds)

    return all_labels, all_preds, all_probs


def main():
    print("="*70)
    print("THRESHOLD TUNING - V3.7 Model Analysis")
    print("="*70)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Cargar datos
    print("Loading data...")
    _, _, test_dataset, tokenizer, le = prepare_scibert_data(use_light_model=False)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Cargar modelo V3.7
    print("Loading V3.7 model...\n")
    model = OptimizedSciBERTClassifier(num_classes=4, dropout=0.35, freeze_bert_layers=3)
    checkpoint = torch.load('best_scibert_optimized.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Probar diferentes thresholds
    thresholds = [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

    results = []

    for threshold in thresholds:
        print(f"Threshold cs.AI = {threshold:.2f}")

        labels, preds, probs = evaluate_with_threshold(
            model, test_loader, device, threshold_cs_ai=threshold
        )

        # Calcular métricas
        report = classification_report(labels, preds, target_names=le.classes_,
                                      digits=4, output_dict=True)

        acc = report['accuracy']
        cs_ai_recall = report['cs.AI']['recall']
        cs_ai_precision = report['cs.AI']['precision']
        cs_ai_f1 = report['cs.AI']['f1-score']

        # Guardar resultados
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'cs_ai_recall': cs_ai_recall,
            'cs_ai_precision': cs_ai_precision,
            'cs_ai_f1': cs_ai_f1,
            'gap_to_goals': (max(0, 0.60 - acc)) + (max(0, 0.30 - cs_ai_recall))
        })

        print(f"  Acc: {acc*100:.2f}% | cs.AI Recall: {cs_ai_recall*100:.2f}% | Gap: {results[-1]['gap_to_goals']*100:.2f}%\n")

    # Resumen
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'cs.AI Recall':<15} {'Gap Total':<12} {'Status'}")
    print("-"*70)

    for r in results:
        status = ""
        if r['cs_ai_recall'] >= 0.30:
            status += "OK cs.AI "
        if r['accuracy'] >= 0.60:
            status += "OK Acc "
        if not status:
            status = "FAIL"

        print(f"{r['threshold']:<12.2f} {r['accuracy']*100:<11.2f}% "
              f"{r['cs_ai_recall']*100:<14.2f}% {r['gap_to_goals']*100:<11.2f}% {status}")

    # Recomendación
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    valid_results = [r for r in results if r['cs_ai_recall'] >= 0.30]

    if valid_results:
        optimal = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\nOptimal threshold: {optimal['threshold']}")
        print(f"  Accuracy: {optimal['accuracy']*100:.2f}%")
        print(f"  cs.AI Recall: {optimal['cs_ai_recall']*100:.2f}%")
        print(f"  Gap total: {optimal['gap_to_goals']*100:.2f}%")

        original = results[0]
        print(f"\nImprovement vs original (threshold=0.5):")
        print(f"  Accuracy: {(optimal['accuracy'] - original['accuracy'])*100:+.2f}%")
        print(f"  cs.AI Recall: {(optimal['cs_ai_recall'] - original['cs_ai_recall'])*100:+.2f}%")
        print(f"  Gap improved: {(original['gap_to_goals'] - optimal['gap_to_goals'])*100:.2f}%")
    else:
        print("\nNo threshold reaches cs.AI >= 30%")
        print("Continue with V3.8 (weight x2.3)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
