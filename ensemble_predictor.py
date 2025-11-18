"""
Ensemble Predictor - Combinación de Múltiples Modelos
Combina V2 (alta accuracy) + V3.7 (alta cs.AI recall)

Estrategias de ensemble:
1. Weighted Voting: Promedio ponderado de probabilidades
2. Stacking: Usar predicciones como features para meta-modelo
3. Boosting: Modelos secuenciales que corrigen errores
4. Majority Voting: Votación por mayoría

Para este proyecto usamos Weighted Voting por su simplicidad y efectividad.
"""

import torch
import pickle
import numpy as np
from transformers import AutoTokenizer
from train_scibert_optimized import OptimizedSciBERTClassifier


class EnsemblePredictor:
    """
    Ensemble de múltiples modelos SciBERT

    Combina predicciones de varios modelos usando weighted average.
    Aplica threshold tuning por clase para optimizar métricas.

    Args:
        model_configs: Lista de configuraciones de modelos
        weights: Pesos para cada modelo (default: uniforme)
        thresholds: Thresholds por clase (default: 0.5 para todas)
    """

    def __init__(self, model_configs, weights=None, thresholds=None):
        """
        model_configs: Lista de dicts con formato:
        [
            {
                'path': 'best_scibert_v2.pth',
                'dropout': 0.5,
                'freeze_layers': 8,
                'name': 'V2'
            },
            {
                'path': 'best_scibert_v3.7_final.pth',
                'dropout': 0.35,
                'freeze_layers': 3,
                'name': 'V3.7'
            }
        ]
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Configuración de ensemble
        self.model_configs = model_configs
        self.num_models = len(model_configs)

        # Pesos de modelos (default: uniforme)
        if weights is None:
            self.weights = np.ones(self.num_models) / self.num_models
        else:
            self.weights = np.array(weights)
            # Normalizar para que sumen 1
            self.weights = self.weights / self.weights.sum()

        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

        # Cargar label encoder
        with open('scibert_label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)

        # Thresholds por clase (default: 0.5)
        if thresholds is None:
            self.thresholds = np.array([0.5, 0.5, 0.5, 0.5])
        else:
            self.thresholds = np.array(thresholds)

        # Cargar modelos
        self.models = []
        for i, config in enumerate(model_configs):
            print(f"\nLoading model {i+1}/{self.num_models}: {config['name']}")
            model = self._load_model(config)
            self.models.append(model)

        print(f"\nEnsemble ready with {self.num_models} models")
        print(f"Model weights: {self.weights}")
        print(f"Class thresholds: {self.thresholds}")
        print(f"Device: {self.device}")

    def _load_model(self, config):
        """Cargar un modelo individual"""
        model = OptimizedSciBERTClassifier(
            num_classes=4,
            dropout=config['dropout'],
            freeze_bert_layers=config['freeze_layers']
        )

        checkpoint = torch.load(config['path'], map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"  ✓ {config['name']}: dropout={config['dropout']}, freeze={config['freeze_layers']}")
        return model

    def predict_proba_single_model(self, model, title_ids, title_mask,
                                   abstract_ids, abstract_mask):
        """Obtener probabilidades de un modelo individual"""
        with torch.no_grad():
            outputs = model(title_ids, title_mask, abstract_ids, abstract_mask)

            # El modelo retorna (logits, attention_weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Probabilidades
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        return probs

    def predict_proba(self, title, abstract):
        """
        Obtener probabilidades ensemble

        Args:
            title: Título del paper
            abstract: Abstract del paper

        Returns:
            probs: Probabilidades por clase [num_classes]
        """
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

        # Obtener probabilidades de cada modelo
        all_probs = []
        for model in self.models:
            probs = self.predict_proba_single_model(
                model, title_ids, title_mask, abstract_ids, abstract_mask
            )
            all_probs.append(probs)

        all_probs = np.array(all_probs)  # [num_models, num_classes]

        # Weighted average
        ensemble_probs = (all_probs * self.weights[:, np.newaxis]).sum(axis=0)

        return ensemble_probs

    def predict(self, title, abstract, return_probs=False, return_all_probs=False):
        """
        Predecir clase con threshold tuning

        Args:
            title: Título del paper
            abstract: Abstract del paper
            return_probs: Si True, retorna también probabilidades ensemble
            return_all_probs: Si True, retorna probabilidades de cada modelo

        Returns:
            prediction: Clase predicha
            [ensemble_probs]: Probabilidades ensemble (opcional)
            [all_probs]: Probabilidades de cada modelo (opcional)
        """
        # Obtener probabilidades ensemble
        ensemble_probs = self.predict_proba(title, abstract)

        # Aplicar threshold tuning multi-clase
        # Para cada clase, si prob >= threshold, es candidata
        candidates = []
        for i, (prob, threshold) in enumerate(zip(ensemble_probs, self.thresholds)):
            if prob >= threshold:
                candidates.append((i, prob))

        # Si hay candidatos, elegir el de mayor probabilidad
        if candidates:
            pred_idx = max(candidates, key=lambda x: x[1])[0]
        else:
            # Si ninguno supera threshold, usar argmax estándar
            pred_idx = ensemble_probs.argmax()

        prediction = self.le.classes_[pred_idx]

        # Preparar retorno
        result = [prediction]

        if return_probs:
            probs_dict = {cls: float(prob)
                         for cls, prob in zip(self.le.classes_, ensemble_probs)}
            result.append(probs_dict)

        if return_all_probs:
            # Obtener probabilidades individuales de cada modelo
            title_enc = self.tokenizer(title, max_length=128, padding='max_length',
                                       truncation=True, return_tensors='pt')
            abstract_enc = self.tokenizer(abstract, max_length=384, padding='max_length',
                                          truncation=True, return_tensors='pt')

            title_ids = title_enc['input_ids'].to(self.device)
            title_mask = title_enc['attention_mask'].to(self.device)
            abstract_ids = abstract_enc['input_ids'].to(self.device)
            abstract_mask = abstract_enc['attention_mask'].to(self.device)

            all_probs = {}
            for i, (model, config) in enumerate(zip(self.models, self.model_configs)):
                probs = self.predict_proba_single_model(
                    model, title_ids, title_mask, abstract_ids, abstract_mask
                )
                all_probs[config['name']] = {
                    cls: float(prob) for cls, prob in zip(self.le.classes_, probs)
                }

            result.append(all_probs)

        return result if len(result) > 1 else result[0]

    def predict_batch(self, papers, show_progress=True):
        """
        Predecir en batch

        Args:
            papers: Lista de dicts con 'title' y 'abstract'
            show_progress: Mostrar barra de progreso

        Returns:
            predictions: Lista de predicciones
        """
        predictions = []

        iterator = papers
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(papers, desc="Predicting")
            except ImportError:
                pass

        for paper in iterator:
            pred = self.predict(paper['title'], paper['abstract'])
            predictions.append(pred)

        return predictions

    def evaluate(self, test_loader):
        """
        Evaluar ensemble en test set

        Args:
            test_loader: DataLoader de test

        Returns:
            accuracy, predictions, true_labels
        """
        all_preds = []
        all_labels = []

        for batch in test_loader:
            title_ids = batch['title_input_ids'].to(self.device)
            title_mask = batch['title_attention_mask'].to(self.device)
            abstract_ids = batch['abstract_input_ids'].to(self.device)
            abstract_mask = batch['abstract_attention_mask'].to(self.device)
            labels = batch['label'].cpu().numpy()

            # Obtener probabilidades de cada modelo
            batch_size = title_ids.size(0)

            for i in range(batch_size):
                # Extraer muestra individual
                t_ids = title_ids[i:i+1]
                t_mask = title_mask[i:i+1]
                a_ids = abstract_ids[i:i+1]
                a_mask = abstract_mask[i:i+1]

                # Ensemble probs
                all_probs = []
                for model in self.models:
                    probs = self.predict_proba_single_model(
                        model, t_ids, t_mask, a_ids, a_mask
                    )
                    all_probs.append(probs)

                all_probs = np.array(all_probs)
                ensemble_probs = (all_probs * self.weights[:, np.newaxis]).sum(axis=0)

                # Threshold tuning
                candidates = []
                for idx, (prob, threshold) in enumerate(zip(ensemble_probs, self.thresholds)):
                    if prob >= threshold:
                        candidates.append((idx, prob))

                if candidates:
                    pred_idx = max(candidates, key=lambda x: x[1])[0]
                else:
                    pred_idx = ensemble_probs.argmax()

                all_preds.append(pred_idx)

            all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        return accuracy, all_preds, all_labels


def create_ensemble_v2_v37(weights=None, thresholds=None):
    """
    Crear ensemble de V2 (alta accuracy) + V3.7 (alta cs.AI recall)

    Args:
        weights: Pesos para [V2, V3.7] (default: [0.4, 0.6])
        thresholds: Thresholds por clase (default: [0.40, 0.5, 0.5, 0.5])

    Returns:
        EnsemblePredictor configurado
    """
    model_configs = [
        {
            'path': 'best_scibert_v2.pth',
            'dropout': 0.5,
            'freeze_layers': 8,
            'name': 'V2'
        },
        {
            'path': 'best_scibert_v3.7_final.pth',
            'dropout': 0.35,
            'freeze_layers': 3,
            'name': 'V3.7'
        }
    ]

    # Default weights: V3.7 tiene más peso (mejor balance)
    if weights is None:
        weights = [0.4, 0.6]

    # Default thresholds: cs.AI = 0.40, resto = 0.5
    if thresholds is None:
        thresholds = [0.40, 0.5, 0.5, 0.5]

    return EnsemblePredictor(model_configs, weights=weights, thresholds=thresholds)


def main():
    """Ejemplo de uso del Ensemble Predictor"""
    print("="*70)
    print("ENSEMBLE PREDICTOR DEMO")
    print("="*70)

    # Nota: Requiere tener modelos V2 y V3.7 entrenados
    print("\nNOTE: This demo requires trained models:")
    print("  - best_scibert_v2.pth")
    print("  - best_scibert_v3.7_final.pth")
    print("  - scibert_label_encoder.pkl")
    print("\nTrain models first using training scripts.")

    print("\n" + "="*70)
    print("Example code:")
    print("="*70)

    code = """
# Crear ensemble
ensemble = create_ensemble_v2_v37(
    weights=[0.4, 0.6],           # V2: 40%, V3.7: 60%
    thresholds=[0.40, 0.5, 0.5, 0.5]  # cs.AI threshold = 0.40
)

# Predecir
prediction = ensemble.predict(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture..."
)

# Predecir con probabilidades
pred, probs, all_probs = ensemble.predict(
    title="Deep Learning for Computer Vision",
    abstract="We propose a novel CNN architecture...",
    return_probs=True,
    return_all_probs=True
)

print(f"Prediction: {pred}")
print(f"Ensemble probs: {probs}")
print(f"Individual model probs: {all_probs}")
"""
    print(code)


if __name__ == "__main__":
    main()
