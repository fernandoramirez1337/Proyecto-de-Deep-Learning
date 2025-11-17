"""
Prediccion Optimizada con Threshold Tuning
Modelo final: V3.7 + threshold=0.40

Metricas finales:
- Test Accuracy: 56.17%
- cs.AI Recall: 36.22% (objetivo: >30%)
- Gap total: 3.83%
"""

import torch
import pickle
from transformers import AutoTokenizer
from train_scibert_optimized import OptimizedSciBERTClassifier


class OptimizedPredictor:
    """Predictor con threshold tuning para cs.AI"""

    def __init__(self, model_path='best_scibert_v3.7_final.pth',
                 threshold_cs_ai=0.40):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.threshold_cs_ai = threshold_cs_ai

        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

        # Cargar label encoder
        with open('scibert_label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)

        # Cargar modelo
        self.model = OptimizedSciBERTClassifier(
            num_classes=4,
            dropout=0.35,
            freeze_bert_layers=3
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")
        print(f"Threshold cs.AI: {self.threshold_cs_ai}")
        print(f"Classes: {list(self.le.classes_)}")

    def predict(self, title, abstract, return_probs=False):
        # Tokenizar
        title_enc = self.tokenizer(
            title,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        abstract_enc = self.tokenizer(
            abstract,
            max_length=384,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Mover a device
        title_ids = title_enc['input_ids'].to(self.device)
        title_mask = title_enc['attention_mask'].to(self.device)
        abstract_ids = abstract_enc['input_ids'].to(self.device)
        abstract_mask = abstract_enc['attention_mask'].to(self.device)

        # Predicción
        with torch.no_grad():
            outputs = self.model(title_ids, title_mask, abstract_ids, abstract_mask)

            # El modelo retorna (logits, attention_weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Probabilidades
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # Aplicar threshold tuning para cs.AI
        cs_ai_idx = list(self.le.classes_).index('cs.AI')

        if probs[cs_ai_idx] >= self.threshold_cs_ai:
            pred_idx = cs_ai_idx
        else:
            pred_idx = probs.argmax()

        prediction = self.le.classes_[pred_idx]

        if return_probs:
            probs_dict = {cls: float(prob) for cls, prob in zip(self.le.classes_, probs)}
            return prediction, probs_dict

        return prediction

    def predict_batch(self, papers):
        predictions = []
        for paper in papers:
            pred = self.predict(paper['title'], paper['abstract'])
            predictions.append(pred)
        return predictions


def main():
    print("="*70)
    print("OPTIMIZED PREDICTOR - V3.7 + Threshold Tuning")
    print("="*70)

    # Crear predictor
    predictor = OptimizedPredictor(threshold_cs_ai=0.40)

    # Ejemplos de prueba
    ejemplos = [
        {
            'title': 'Deep Learning for Image Classification',
            'abstract': 'We propose a novel convolutional neural network architecture for image classification tasks. Our model achieves state-of-the-art results on ImageNet.'
        },
        {
            'title': 'Natural Language Processing with Transformers',
            'abstract': 'This paper introduces a new approach to language understanding using transformer-based models. We evaluate on several NLP benchmarks.'
        },
        {
            'title': 'Reinforcement Learning for Game Playing',
            'abstract': 'We develop an AI agent that learns to play games through reinforcement learning. The agent uses deep Q-learning to optimize its strategy.'
        },
        {
            'title': 'Machine Learning for Predictive Analytics',
            'abstract': 'We apply various machine learning algorithms to predict customer churn. Random forests and gradient boosting show the best performance.'
        }
    ]

    print(f"\n{'='*70}")
    print("PREDICTIONS")
    print(f"{'='*70}\n")

    for i, ejemplo in enumerate(ejemplos, 1):
        pred, probs = predictor.predict(
            ejemplo['title'],
            ejemplo['abstract'],
            return_probs=True
        )

        print(f"Paper {i}:")
        print(f"  Title: {ejemplo['title']}")
        print(f"  Prediction: {pred}")
        print(f"  Probabilities:")
        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            marker = " *" if cls == pred else ""
            print(f"    {cls}: {prob:.4f}{marker}")
        print()

    print("="*70)
    print("Predictions completed")
    print("="*70)


if __name__ == "__main__":
    main()
