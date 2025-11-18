# Advanced Improvements Analysis - Exploración Exhaustiva

**Fecha:** 2025-11-18
**Contexto:** Análisis profundo mientras entrena V2 para ensemble
**Objetivo:** 60% accuracy (actual: 56.17%, gap: -3.83%)

---

## 📊 Estado Actual - Análisis Profundo

### Modelo Actual: V3.7+TT
```
Arquitectura: Dual-Encoder SciBERT
- Title encoder: SciBERT (32 tokens)
- Abstract encoder: SciBERT (128 tokens, mismo weights)
- Fusion: 3-layer MLP (1536→512→256→128→4)
- Pooling: Attention pooling (learnable)
- Regularización: Dropout 0.35, Weight Decay 0.01, Label Smoothing 0.1
- Freeze: 3 primeras capas de 12

Resultados:
- Test Accuracy: 56.17%
- cs.AI Recall: 36.22% ✓ (objetivo >30%)
- Gap: 3.83%
```

### Técnicas YA Probadas (10 versiones)
- ✅ Layer freezing (3, 5-6, 8 capas)
- ✅ Dropout variations (0.35, 0.4, 0.5)
- ✅ Learning rate tuning (3e-5, 5e-5)
- ✅ Weight decay variations (0.01, 0.015, 0.05)
- ✅ Class weighting (x1.5, x2.0, x2.3, x3.0)
- ✅ Threshold tuning (single-class, multi-class)
- ✅ Focal Loss (gamma=2.0) ❌ FAILED
- ✅ Label smoothing (0.1)
- ✅ Gradient clipping (1.0)
- ✅ Early stopping (patience=3)

---

## 🚀 CATEGORÍA A: Técnicas Arquitecturales (NO probadas)

### A1. **Cross-Attention Entre Title y Abstract** ⭐⭐⭐

**Problema actual:**
```python
# Actual: Concatenación simple
title_pooled = attention_pool(title_hidden)     # [batch, 768]
abstract_pooled = attention_pool(abstract_hidden) # [batch, 768]
combined = torch.cat([title_pooled, abstract_pooled], dim=1)  # [batch, 1536]
```

**Mejora propuesta:**
```python
# Cross-attention: Title y Abstract interactúan
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8):
        super().__init__()
        self.title_to_abstract = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.abstract_to_title = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, title_seq, abstract_seq):
        # Title attends to abstract
        title_enhanced, _ = self.title_to_abstract(
            query=title_seq,      # [batch, 32, 768]
            key=abstract_seq,     # [batch, 128, 768]
            value=abstract_seq
        )

        # Abstract attends to title
        abstract_enhanced, _ = self.abstract_to_title(
            query=abstract_seq,
            key=title_seq,
            value=title_seq
        )

        # Pool enhanced representations
        title_pooled = title_enhanced.mean(dim=1)      # [batch, 768]
        abstract_pooled = abstract_enhanced.mean(dim=1) # [batch, 768]

        return torch.cat([title_pooled, abstract_pooled], dim=1)
```

**Ventajas:**
- Title keywords influencian interpretación de abstract
- Abstract contexto refina comprensión de title
- Interacción bidireccional (no solo concatenación)

**Mejora esperada:** +1-2% accuracy

**Implementación:** 30 min modificación + 60-80 min entrenamiento

---

### A2. **Hierarchical Attention** ⭐⭐⭐

**Concepto:**
Atención a múltiples niveles: palabra → sentencia → documento

```python
class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        # Word-level attention
        self.word_attention = nn.Linear(hidden_size, 1)

        # Sentence-level attention (cada N tokens = sentencia)
        self.sentence_attention = nn.Linear(hidden_size, 1)

    def forward(self, sequence, mask):
        # [batch, seq_len, 768]

        # 1. Word-level attention (actual implementation)
        word_scores = self.word_attention(sequence).squeeze(-1)
        word_scores = word_scores.masked_fill(mask == 0, -1e9)
        word_weights = torch.softmax(word_scores, dim=1)

        # 2. Sentence-level grouping
        # Agrupar cada 10 tokens como "sentencia"
        batch_size, seq_len, hidden = sequence.shape
        num_sentences = seq_len // 10

        sentences = sequence[:, :num_sentences*10, :].reshape(
            batch_size, num_sentences, 10, hidden
        )  # [batch, num_sent, 10, 768]

        # Pool dentro de cada sentencia
        sentence_repr = (sentences * word_weights[:, :num_sentences*10].reshape(
            batch_size, num_sentences, 10, 1
        )).sum(dim=2)  # [batch, num_sent, 768]

        # 3. Sentence-level attention
        sent_scores = self.sentence_attention(sentence_repr).squeeze(-1)
        sent_weights = torch.softmax(sent_scores, dim=1)

        # Final pooled
        pooled = (sentence_repr * sent_weights.unsqueeze(-1)).sum(dim=1)

        return pooled
```

**Ventajas:**
- Captura estructura jerárquica de texto científico
- Diferentes niveles de abstracción
- Más expresivo que single-level attention

**Mejora esperada:** +1-1.5% accuracy

**Complejidad:** Media (1h implementación)

---

### A3. **Separate Encoders con Especialización** ⭐⭐

**Concepto actual:**
Title y Abstract usan MISMO SciBERT encoder (pesos compartidos)

**Mejora propuesta:**
Encoders separados con fine-tuning diferenciado

```python
class SpecializedDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder para titles (keywords, corto)
        self.title_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        # Encoder para abstracts (contextual, largo)
        self.abstract_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        # Fine-tune diferente
        # Title: Solo últimas 6 capas
        for layer in self.title_encoder.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        # Abstract: Solo últimas 3 capas (más adaptable)
        for layer in self.abstract_encoder.encoder.layer[:9]:
            for param in layer.parameters():
                param.requires_grad = False
```

**Ventajas:**
- Title encoder se especializa en keywords
- Abstract encoder se especializa en contexto
- Más parámetros (mejor capacidad)

**Desventajas:**
- 2x modelo size (~2GB en lugar de 1GB)
- 2x training time
- No viable en M2 (requiere GPU)

**Mejora esperada:** +2-3% accuracy (en GPU)

---

### A4. **Graph Neural Network sobre Keywords** ⭐⭐

**Concepto:**
Extraer keywords del title/abstract y crear grafo de relaciones

```python
class KeywordGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        # GNN para procesar relaciones entre keywords
        self.gnn = GCNConv(768, 768)

    def forward(self, text):
        # 1. Extract keywords (usando TF-IDF o YAKE)
        keywords = extract_keywords(text, top_k=10)

        # 2. Encode keywords con BERT
        keyword_embeds = self.bert(keywords)  # [10, 768]

        # 3. Build graph (co-occurrence edges)
        edge_index = build_cooccurrence_graph(keywords)

        # 4. GNN processing
        keyword_graph_embeds = self.gnn(keyword_embeds, edge_index)

        # 5. Pool graph
        graph_pooled = keyword_graph_embeds.mean(dim=0)  # [768]

        return graph_pooled
```

**Ventajas:**
- Captura relaciones semánticas entre conceptos
- Útil para papers científicos (keywords importantes)
- Explainable (puedes ver qué keywords influyen)

**Desventajas:**
- Complejidad alta
- Requiere librerías adicionales (PyTorch Geometric)
- Preprocesamiento costoso

**Mejora esperada:** +1-2% accuracy

**Complejidad:** Alta (4-6h implementación)

---

## 🔬 CATEGORÍA B: Técnicas de Loss Function

### B1. **Dice Loss** ⭐⭐⭐

**Por qué Dice en lugar de Focal:**
Focal Loss falló porque penaliza demasiado ejemplos fáciles (gamma=2.0)

Dice Loss es más suave y funciona bien para imbalance:

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # Convertir a probabilidades
        probs = F.softmax(logits, dim=1)  # [batch, 4]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=4).float()  # [batch, 4]

        # Dice coefficient por clase
        intersection = (probs * targets_one_hot).sum(dim=0)  # [4]
        cardinality = (probs + targets_one_hot).sum(dim=0)   # [4]

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # Aplicar class weights
        if self.class_weights is not None:
            dice = dice * self.class_weights

        # Dice loss = 1 - dice
        dice_loss = 1 - dice.mean()

        return dice_loss
```

**Ventajas sobre Focal Loss:**
- No tiene parámetro gamma (menos sensible)
- Optimiza directamente F1-score (no cross-entropy)
- Más suave con ejemplos fáciles

**Mejora esperada:** +1-2% accuracy

**Riesgo:** Bajo (más estable que Focal)

---

### B2. **Label Distribution Learning** ⭐⭐

**Concepto:**
En lugar de one-hot (cs.AI = [1, 0, 0, 0]), usar distribución suave basada en similaridad semántica

```python
# Matriz de similaridad entre clases (pre-calculada)
CLASS_SIMILARITY = torch.FloatTensor([
    #    AI    CL    CV    LG
    [1.00, 0.30, 0.20, 0.40],  # cs.AI similar a LG (machine learning)
    [0.30, 1.00, 0.15, 0.25],  # cs.CL
    [0.20, 0.15, 1.00, 0.30],  # cs.CV similar a LG (deep learning)
    [0.40, 0.25, 0.30, 1.00]   # cs.LG
])

def soft_labels(target_class):
    # En lugar de [1, 0, 0, 0] para cs.AI
    # Usar [0.70, 0.15, 0.05, 0.10] (basado en similaridad)
    soft = CLASS_SIMILARITY[target_class]
    soft = soft / soft.sum()  # Normalizar
    return soft
```

**Ventajas:**
- Refleja relaciones reales entre clases
- Reduce overconfidence
- Papers científicos tienen overlap (AI/ML, CV/ML)

**Mejora esperada:** +0.5-1% accuracy

---

### B3. **Curriculum Learning Loss** ⭐⭐

**Concepto:**
Entrenar primero en ejemplos fáciles, gradualmente incorporar difíciles

```python
class CurriculumTrainer:
    def __init__(self, model, difficulty_scores):
        self.model = model
        self.difficulty_scores = difficulty_scores  # [num_samples]

    def train_epoch(self, epoch, max_epochs):
        # Curriculum pace: % de datos a usar
        curriculum_pace = min(1.0, 0.5 + 0.5 * (epoch / max_epochs))

        # Seleccionar muestras basado en dificultad
        num_samples = int(len(self.difficulty_scores) * curriculum_pace)
        sorted_indices = self.difficulty_scores.argsort()
        selected_indices = sorted_indices[:num_samples]

        # Entrenar solo en muestras seleccionadas
        for idx in selected_indices:
            # Training step...
            pass
```

**Cómo calcular difficulty:**
```python
# Basado en:
# 1. Longitud de abstract (más largo = más difícil)
# 2. Número de palabras técnicas
# 3. Entropía de predicción de modelo base

def compute_difficulty(sample):
    abstract_len = len(sample['abstract'].split())
    num_technical_words = count_technical_terms(sample)

    # Normalizar
    difficulty = 0.5 * (abstract_len / 500) + 0.5 * (num_technical_words / 20)
    return difficulty
```

**Ventajas:**
- Entrenamiento más estable
- Menos overfitting en ejemplos difíciles
- Convergencia más rápida

**Mejora esperada:** +0.5-1.5% accuracy

---

## 📚 CATEGORÍA C: Técnicas de Data

### C1. **Back-Translation Data Augmentation** ⭐⭐⭐

**Implementación detallada:**

```python
from transformers import MarianMTModel, MarianTokenizer

class BackTranslationAugmenter:
    def __init__(self):
        # English → Spanish
        self.model_en_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
        self.tokenizer_en_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

        # Spanish → English
        self.model_es_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
        self.tokenizer_es_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')

    def augment(self, text):
        # EN → ES
        inputs = self.tokenizer_en_es(text, return_tensors="pt", padding=True)
        translated = self.model_en_es.generate(**inputs)
        spanish = self.tokenizer_en_es.decode(translated[0], skip_special_tokens=True)

        # ES → EN
        inputs = self.tokenizer_es_en(spanish, return_tensors="pt", padding=True)
        back_translated = self.model_es_en.generate(**inputs)
        english = self.tokenizer_es_en.decode(back_translated[0], skip_special_tokens=True)

        return english

# Augmentar cs.AI samples
augmenter = BackTranslationAugmenter()

cs_ai_samples = df[df['category'] == 'cs.AI']
augmented = []

for idx, row in cs_ai_samples.iterrows():
    # Original
    augmented.append(row)

    # Back-translated abstract
    aug_abstract = augmenter.augment(row['abstract'])
    augmented.append({
        'title': row['title'],
        'abstract': aug_abstract,
        'category': 'cs.AI'
    })

# Nuevo dataset: 12,000 → 12,450 samples
# cs.AI: 450 → 900 samples (2x)
```

**Mejora esperada:** +1.5-2.5% accuracy

**Tiempo:** 2-3h implementación + augmentation

---

### C2. **Mixup en Feature Space** ⭐⭐

**Concepto:**
Interpolación de embeddings entre muestras

```python
def mixup_data(embeddings, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = embeddings.size(0)
    index = torch.randperm(batch_size)

    # Interpolar embeddings
    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]

    # Labels mixtas
    labels_a, labels_b = labels, labels[index]

    return mixed_embeddings, labels_a, labels_b, lam

# Durante training
title_pooled, abstract_pooled = encode(title, abstract)
combined = torch.cat([title_pooled, abstract_pooled], dim=1)

# Aplicar mixup
combined_mixed, labels_a, labels_b, lam = mixup_data(combined, labels)

# Loss mixto
outputs = classifier(combined_mixed)
loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
```

**Ventajas:**
- Regularización implícita
- Crea "samples virtuales"
- Reduce overfitting

**Mejora esperada:** +0.5-1% accuracy

---

### C3. **Pseudo-Labeling (Semi-Supervised)** ⭐⭐

**Concepto:**
Usar predicciones confiadas del modelo para aumentar dataset

```python
class PseudoLabeler:
    def __init__(self, model, confidence_threshold=0.95):
        self.model = model
        self.threshold = confidence_threshold

    def generate_pseudo_labels(self, unlabeled_data):
        self.model.eval()
        pseudo_labeled = []

        for sample in unlabeled_data:
            with torch.no_grad():
                logits = self.model(sample)
                probs = F.softmax(logits, dim=1)
                max_prob, pred_class = probs.max(dim=1)

                # Solo usar predicciones muy confiadas
                if max_prob > self.threshold:
                    pseudo_labeled.append({
                        'text': sample,
                        'label': pred_class,
                        'confidence': max_prob,
                        'pseudo': True
                    })

        return pseudo_labeled

# Iterar:
# 1. Entrenar en labeled data
# 2. Generar pseudo-labels
# 3. Re-entrenar con labeled + pseudo-labeled
# 4. Repeat
```

**Fuente de unlabeled data:**
- Más papers de ArXiv (sin etiquetar)
- Usar modelo actual para etiquetar

**Mejora esperada:** +1-2% accuracy

**Complejidad:** Media-Alta

---

## 🧠 CATEGORÍA D: Técnicas de Ensemble Avanzadas

### D1. **Stacking Ensemble** ⭐⭐⭐

**Más sofisticado que weighted voting:**

```python
class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models  # [V2, V3.7, V4, ...]
        self.meta_model = meta_model    # Modelo simple (LogisticRegression)

    def train_meta_model(self, val_loader):
        # 1. Obtener predicciones de base models en VAL
        meta_features = []
        meta_labels = []

        for batch in val_loader:
            base_preds = []
            for model in self.base_models:
                probs = model.predict_proba(batch)
                base_preds.append(probs)

            # Concatenar predicciones como features
            meta_feat = np.concatenate(base_preds, axis=1)  # [batch, 4*num_models]
            meta_features.append(meta_feat)
            meta_labels.append(batch['labels'])

        meta_features = np.vstack(meta_features)
        meta_labels = np.concatenate(meta_labels)

        # 2. Entrenar meta-modelo
        self.meta_model.fit(meta_features, meta_labels)

    def predict(self, sample):
        # Predicciones de base models
        base_preds = [model.predict_proba(sample) for model in self.base_models]
        meta_feat = np.concatenate(base_preds)

        # Meta-modelo decide
        final_pred = self.meta_model.predict(meta_feat)
        return final_pred
```

**Ventajas:**
- Aprende a combinar modelos de forma óptima
- Más flexible que pesos fijos
- Puede capturar interacciones entre modelos

**Mejora esperada:** +1-2% sobre ensemble simple

---

### D2. **Boosting-Style Ensemble** ⭐⭐

**Concepto:**
Entrenar modelos secuenciales que corrigen errores de anteriores

```python
class BoostingEnsemble:
    def __init__(self):
        self.models = []
        self.weights = []

    def train(self, train_data):
        # Inicializar pesos de samples (uniforme)
        sample_weights = np.ones(len(train_data)) / len(train_data)

        for iteration in range(num_iterations):
            # 1. Entrenar modelo con sample_weights
            model = train_model(train_data, sample_weights)

            # 2. Evaluar
            predictions = model.predict(train_data)
            errors = (predictions != train_data.labels).astype(float)

            # 3. Calcular error ponderado
            weighted_error = (errors * sample_weights).sum()

            # 4. Peso del modelo
            model_weight = 0.5 * np.log((1 - weighted_error) / weighted_error)

            # 5. Actualizar sample_weights (más peso a errores)
            sample_weights *= np.exp(model_weight * errors)
            sample_weights /= sample_weights.sum()

            self.models.append(model)
            self.weights.append(model_weight)
```

**Ventajas:**
- Foco en ejemplos difíciles
- Mejora iterativa
- Teóricamente garantizado reducir error

**Desventajas:**
- Requiere entrenar múltiples modelos
- Costoso en tiempo (~3-4h por modelo x N modelos)

**Mejora esperada:** +2-3% accuracy

---

### D3. **Multi-Task Ensemble** ⭐⭐

**Concepto:**
Entrenar modelos con tareas auxiliares

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

        # Task 1: Clasificación principal (4 clases)
        self.classifier_main = nn.Linear(768, 4)

        # Task 2: Predecir si paper tiene "learning" en abstract
        self.classifier_aux1 = nn.Linear(768, 2)

        # Task 3: Predecir longitud de abstract (regresión)
        self.regressor_aux2 = nn.Linear(768, 1)

    def forward(self, x):
        hidden = self.bert(x).pooler_output

        main_logits = self.classifier_main(hidden)
        aux1_logits = self.classifier_aux1(hidden)
        aux2_pred = self.regressor_aux2(hidden)

        return main_logits, aux1_logits, aux2_pred

# Loss combinado
loss = loss_main + 0.3 * loss_aux1 + 0.2 * loss_aux2
```

**Ventajas:**
- Tareas auxiliares mejoran representaciones
- Regularización implícita
- Mejor generalización

**Mejora esperada:** +0.5-1.5% accuracy

---

## 🎯 CATEGORÍA E: Optimizaciones Específicas para cs.AI

### E1. **Focal Loss Corregido** ⭐⭐⭐

**Por qué V4.0 falló:** gamma=2.0 demasiado alto

**Configuración conservadora:**

```python
# V4.0 (FALLÓ)
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0]
LR = 5e-5
DROPOUT = 0.35

# V4.1 (CORREGIDO)
FOCAL_GAMMA = 1.0              # Reducido 50%
CLASS_WEIGHTS = None           # Dejar que Focal maneje
ALPHA_FOCAL = [1.5, 1.0, 1.0, 1.0]  # Class weights EN Focal
LR = 3e-5                      # Reducido 40%
DROPOUT = 0.40                 # Aumentado regularización
WEIGHT_DECAY = 0.015           # Aumentado
```

**Mejora esperada:** +1-2% accuracy (si se reintenta)

---

### E2. **Cost-Sensitive Learning Avanzado** ⭐⭐

**Concepto:**
Penalizar más errores de cs.AI

```python
class CostSensitiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Matriz de costos [true_class, pred_class]
        self.cost_matrix = torch.FloatTensor([
            # Predicted:  AI    CL    CV    LG
            [0.0,  2.0,  2.0,  1.5],  # True: AI (error de AI costly)
            [1.0,  0.0,  1.0,  1.0],  # True: CL
            [1.0,  1.0,  0.0,  1.0],  # True: CV
            [1.0,  1.0,  1.0,  0.0]   # True: LG
        ])

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        # Cost para cada predicción
        batch_size = targets.size(0)
        costs = torch.zeros(batch_size)

        for i in range(batch_size):
            true_class = targets[i]
            predicted_probs = probs[i]

            # Expected cost
            cost = (predicted_probs * self.cost_matrix[true_class]).sum()
            costs[i] = cost

        return costs.mean()
```

**Ventajas:**
- Penaliza específicamente errores de cs.AI
- Más flexible que class weights

**Mejora esperada:** +0.5-1% accuracy

---

### E3. **SMOTE para cs.AI (Oversampling Inteligente)** ⭐⭐

**Concepto:**
Crear samples sintéticos de cs.AI interpolando en embedding space

```python
from imblearn.over_sampling import SMOTE

class SMOTEForText:
    def __init__(self, model):
        self.model = model

    def oversample_cs_ai(self, dataset):
        # 1. Encode todos los samples
        embeddings = []
        labels = []

        for sample in dataset:
            with torch.no_grad():
                emb = self.model.encode(sample)  # [768]
                embeddings.append(emb)
                labels.append(sample['label'])

        embeddings = np.array(embeddings)
        labels = np.array(labels)

        # 2. SMOTE
        smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
        embeddings_resampled, labels_resampled = smote.fit_resample(embeddings, labels)

        # 3. Decodificar nuevos samples (usar nearest neighbor en espacio original)
        new_samples = []
        for emb in embeddings_resampled[len(dataset):]:  # Solo nuevos
            # Buscar sample original más cercano
            distances = np.linalg.norm(embeddings - emb, axis=1)
            nearest_idx = distances.argmin()

            # Modificar ligeramente el texto original
            original_sample = dataset[nearest_idx]
            new_sample = perturb_text(original_sample)  # Synonym replacement
            new_samples.append(new_sample)

        return new_samples
```

**Ventajas:**
- Oversampling inteligente (no solo duplicar)
- Balancea clases en embedding space
- Reduce overfitting vs simple duplication

**Mejora esperada:** +1-1.5% accuracy

---

## 📈 CATEGORÍA F: Modelos Más Grandes (Requiere GPU)

### F1. **RoBERTa-base Fine-Tuning** ⭐⭐⭐

```python
from transformers import RobertaModel

model = RobertaModel.from_pretrained('roberta-base')
# 125M parámetros (vs SciBERT 110M)
# Mejor en benchmarks generales
```

**Mejora esperada:** +2-3% accuracy

**Problema:** No viable en M2 (requiere GPU con >8GB VRAM)

---

### F2. **DeBERTa-v3-base** ⭐⭐⭐

```python
from transformers import DebertaV2Model

model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')
# 184M parámetros
# State-of-the-art en muchos benchmarks
```

**Mejora esperada:** +3-4% accuracy

**Problema:** No viable en M2

---

### F3. **SciBERT-large** ⭐⭐⭐

Si existiera (no existe oficialmente), pero alternativas:

```python
# BioLinkBERT-large (similar domain)
model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')

# PubMedBERT-large
model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-large')
```

**Mejora esperada:** +2-3% accuracy

---

## 🔥 TOP 5 RECOMENDACIONES (Ordenadas por Viabilidad)

### **1. Back-Translation Data Augmentation** ⭐⭐⭐
- **Tiempo:** 2-3h
- **Mejora:** +1.5-2.5%
- **Riesgo:** Bajo
- **Viable en M2:** ✅ SÍ
- **Próxima prioridad:** #1

### **2. Cross-Attention Fusion** ⭐⭐⭐
- **Tiempo:** 30min implementación + 1-2h entrenamiento
- **Mejora:** +1-2%
- **Riesgo:** Bajo
- **Viable en M2:** ✅ SÍ
- **Próxima prioridad:** #2

### **3. Dice Loss** ⭐⭐⭐
- **Tiempo:** 1h implementación + 1-2h entrenamiento
- **Mejora:** +1-2%
- **Riesgo:** Bajo (más estable que Focal)
- **Viable en M2:** ✅ SÍ
- **Próxima prioridad:** #3

### **4. Stacking Ensemble** ⭐⭐⭐
- **Tiempo:** 1h (si ya tienes V2, V3.7 entrenados)
- **Mejora:** +1-2%
- **Riesgo:** Bajo
- **Viable en M2:** ✅ SÍ
- **Próxima prioridad:** #4 (después de entrenar V2)

### **5. SMOTE + Mixup** ⭐⭐
- **Tiempo:** 2h implementación + 1-2h entrenamiento
- **Mejora:** +1-1.5%
- **Riesgo:** Medio
- **Viable en M2:** ✅ SÍ
- **Próxima prioridad:** #5

---

## 🎯 Hoja de Ruta Sugerida

### **Si quieres alcanzar 60%:**

**Fase 1: Quick Wins (1 semana)**
1. Back-Translation Augmentation (2-3h)
2. Cross-Attention Architecture (2-3h)
3. Ensemble V2+V3.7 con stacking (1h)

**Resultado esperado:** 56.17% + 1.5% + 1% + 1% = **~59-60%** ✅

**Fase 2: Advanced (si no alcanza)**
4. Dice Loss reentrenamiento (2h)
5. SMOTE oversampling (2h)

**Resultado esperado:** +1-2% adicional = **60-61%**

---

### **Si quieres optimizar tiempo:**

**Fast Track (4-5 horas total):**
1. Ensemble V2+V3.7 (1h) → +1%
2. Cross-Attention (2h) → +1-2%
3. Back-Translation light (1h, solo 50% de cs.AI) → +1%

**Resultado esperado:** ~58-59% accuracy

---

## 📊 Tabla Comparativa de Todas las Técnicas

| Técnica | Tiempo | Mejora | Riesgo | M2 Viable | Prioridad |
|---------|--------|--------|--------|-----------|-----------|
| **Back-Translation** | 2-3h | +1.5-2.5% | Bajo | ✅ | ⭐⭐⭐ |
| **Cross-Attention** | 2h | +1-2% | Bajo | ✅ | ⭐⭐⭐ |
| **Dice Loss** | 2h | +1-2% | Bajo | ✅ | ⭐⭐⭐ |
| **Stacking Ensemble** | 1h | +1-2% | Bajo | ✅ | ⭐⭐⭐ |
| **Hierarchical Attention** | 2-3h | +1-1.5% | Medio | ✅ | ⭐⭐ |
| **SMOTE + Mixup** | 2-3h | +1-1.5% | Medio | ✅ | ⭐⭐ |
| **Label Distribution** | 1h | +0.5-1% | Bajo | ✅ | ⭐⭐ |
| **Focal Loss v2** | 2h | +1-2% | Medio | ✅ | ⭐⭐ |
| **Cost-Sensitive** | 2h | +0.5-1% | Medio | ✅ | ⭐⭐ |
| **Curriculum Learning** | 2-3h | +0.5-1.5% | Medio | ✅ | ⭐ |
| **Pseudo-Labeling** | 4-6h | +1-2% | Alto | ✅ | ⭐ |
| **Graph NN** | 6-8h | +1-2% | Alto | ⚠️ | ⭐ |
| **Separate Encoders** | 3-4h | +2-3% | Bajo | ❌ GPU | - |
| **RoBERTa/DeBERTa** | 4-6h | +2-4% | Bajo | ❌ GPU | - |

---

## 🚀 Quick Start Scripts

He creado código base para las top 3 técnicas. Archivos listos:

1. `advanced_cross_attention.py` - Cross-Attention implementation
2. `advanced_data_augmentation.py` - Back-Translation pipeline
3. `advanced_dice_loss.py` - Dice Loss implementation
4. `advanced_stacking_ensemble.py` - Stacking ensemble

**Para empezar después del ensemble V2+V3.7:**
```bash
# Opción más prometedora
python advanced_data_augmentation.py --augment_cs_ai --output data/arxiv_augmented.csv
python train_scibert_optimized.py --data data/arxiv_augmented.csv
```

---

**¿Quieres que implemente alguna de estas técnicas ahora mientras esperas?**
