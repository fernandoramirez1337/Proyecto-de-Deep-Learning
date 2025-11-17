# model_scibert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class SciBERTClassifier(nn.Module):
    """
    Clasificador basado en SciBERT pre-entrenado
    Usa embeddings contextuales para ttulo y abstract
    """
    def __init__(self, num_classes=4, dropout=0.3, freeze_bert=False):
        super().__init__()

        # Cargar SciBERT pre-entrenado
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Opcionalmente congelar BERT para fine-tuning ms rpido
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Attention para pooling de secuencias
        self.title_attention = nn.Linear(self.hidden_size, 1)
        self.abstract_attention = nn.Linear(self.hidden_size, 1)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def attention_pool(self, hidden_states, attention_layer, attention_mask):
        """
        Pooling con atencin sobre secuencia
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_layer: Linear layer para scores
            attention_mask: (batch, seq_len) mscara de padding
        """
        # Calcular scores de atencin
        scores = attention_layer(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Aplicar mscara (poner -inf en tokens padding)
        scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax para obtener pesos
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Pooling ponderado
        pooled = (hidden_states * weights).sum(dim=1)  # (batch, hidden_size)

        return pooled, weights.squeeze(-1)

    def forward(self, title_input_ids, title_attention_mask,
                abstract_input_ids, abstract_attention_mask):
        """
        Forward pass
        Args:
            title_input_ids: (batch, title_len)
            title_attention_mask: (batch, title_len)
            abstract_input_ids: (batch, abstract_len)
            abstract_attention_mask: (batch, abstract_len)
        """
        # Procesar ttulo con BERT
        title_outputs = self.bert(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask
        )
        title_hidden = title_outputs.last_hidden_state  # (batch, seq_len, 768)

        # Procesar abstract con BERT
        abstract_outputs = self.bert(
            input_ids=abstract_input_ids,
            attention_mask=abstract_attention_mask
        )
        abstract_hidden = abstract_outputs.last_hidden_state  # (batch, seq_len, 768)

        # Attention pooling para cada modalidad
        title_pooled, title_attn_weights = self.attention_pool(
            title_hidden, self.title_attention, title_attention_mask
        )

        abstract_pooled, abstract_attn_weights = self.attention_pool(
            abstract_hidden, self.abstract_attention, abstract_attention_mask
        )

        # Concatenar representaciones
        combined = torch.cat([title_pooled, abstract_pooled], dim=1)  # (batch, 768*2)

        # Clasificacin
        output = self.fusion(combined)

        return output, title_attn_weights, abstract_attn_weights


class LightSciBERTClassifier(nn.Module):
    """
    Versin ms ligera: concatena ttulo+abstract y procesa una sola vez
    Ms rpida pero posiblemente menos precisa
    """
    def __init__(self, num_classes=4, dropout=0.3, freeze_bert=False):
        super().__init__()

        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Clasificador simple
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass con texto combinado
        Args:
            input_ids: (batch, seq_len) - title [SEP] abstract
            attention_mask: (batch, seq_len)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Usar [CLS] token como representacin
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Clasificacin
        logits = self.classifier(cls_output)

        return logits


# Test rpido
if __name__ == "__main__":
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    # Modelo dual (ttulo + abstract separados)
    model_dual = SciBERTClassifier(num_classes=4, freeze_bert=True)

    # Test forward pass
    batch_size = 2
    title_text = ["Deep Learning for Computer Vision", "Natural Language Processing"]
    abstract_text = ["This paper presents a novel approach...", "We introduce a new method..."]

    title_enc = tokenizer(title_text, padding=True, truncation=True,
                         max_length=32, return_tensors='pt')
    abstract_enc = tokenizer(abstract_text, padding=True, truncation=True,
                            max_length=128, return_tensors='pt')

    with torch.no_grad():
        output, title_attn, abstract_attn = model_dual(
            title_enc['input_ids'], title_enc['attention_mask'],
            abstract_enc['input_ids'], abstract_enc['attention_mask']
        )

    # Modelo ligero
    model_light = LightSciBERTClassifier(num_classes=4, freeze_bert=True)

    combined_text = [f"{t} [SEP] {a}" for t, a in zip(title_text, abstract_text)]
    combined_enc = tokenizer(combined_text, padding=True, truncation=True,
                            max_length=160, return_tensors='pt')

    with torch.no_grad():
        output_light = model_light(combined_enc['input_ids'],
                                   combined_enc['attention_mask'])

    print("Modelos OK")
