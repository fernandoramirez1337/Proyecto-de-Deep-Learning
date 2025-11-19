"""
Cross-Attention Fusion Architecture
Mejora sobre concatenación simple de title + abstract

Mejora esperada: +1-2% accuracy
Tiempo: 30min implementación + 60-80min entrenamiento
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class CrossAttentionSciBERT(nn.Module):
    """
    SciBERT con Cross-Attention entre Title y Abstract

    Mejora sobre V3.7: En lugar de concatenar title_pooled + abstract_pooled,
    permite que title y abstract interactúen mediante cross-attention.
    """

    def __init__(self, num_classes=4, dropout=0.35, freeze_bert_layers=3):
        super().__init__()

        # SciBERT encoder (compartido entre title y abstract)
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Freeze primeras capas
        if freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Embedding dropout
        self.embedding_dropout = nn.Dropout(0.1)

        # Cross-Attention: Title -> Abstract
        self.cross_attn_title_to_abstract = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Cross-Attention: Abstract -> Title
        self.cross_attn_abstract_to_title = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization después de cross-attention
        self.norm_title = nn.LayerNorm(self.hidden_size)
        self.norm_abstract = nn.LayerNorm(self.hidden_size)

        # Self-attention pooling para cada modalidad (opcional)
        self.title_pool_attn = nn.Linear(self.hidden_size, 1)
        self.abstract_pool_attn = nn.Linear(self.hidden_size, 1)

        # Fusion network (igual que V3.7)
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

        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos personalizados"""
        for module in [self.title_pool_attn, self.abstract_pool_attn]:
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
        return pooled

    def forward(self, title_input_ids, title_attention_mask,
                abstract_input_ids, abstract_attention_mask):
        """
        Forward pass con cross-attention

        Flow:
        1. Encode title y abstract con SciBERT
        2. Apply cross-attention (bidireccional)
        3. Pool enhanced representations
        4. Classify
        """

        # 1. Encode con SciBERT
        title_outputs = self.bert(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask
        )
        title_hidden = self.embedding_dropout(title_outputs.last_hidden_state)
        # [batch, title_seq_len, 768]

        abstract_outputs = self.bert(
            input_ids=abstract_input_ids,
            attention_mask=abstract_attention_mask
        )
        abstract_hidden = self.embedding_dropout(abstract_outputs.last_hidden_state)
        # [batch, abstract_seq_len, 768]

        # 2. Cross-Attention

        # Title attends to Abstract (keywords del title buscan contexto en abstract)
        title_enhanced, _ = self.cross_attn_title_to_abstract(
            query=title_hidden,
            key=abstract_hidden,
            value=abstract_hidden,
            key_padding_mask=(abstract_attention_mask == 0)
        )
        # [batch, title_seq_len, 768]

        # Residual connection + normalization
        title_enhanced = self.norm_title(title_hidden + title_enhanced)

        # Abstract attends to Title (contexto del abstract influenciado por keywords)
        abstract_enhanced, _ = self.cross_attn_abstract_to_title(
            query=abstract_hidden,
            key=title_hidden,
            value=title_hidden,
            key_padding_mask=(title_attention_mask == 0)
        )
        # [batch, abstract_seq_len, 768]

        # Residual connection + normalization
        abstract_enhanced = self.norm_abstract(abstract_hidden + abstract_enhanced)

        # 3. Pool enhanced representations
        title_pooled = self.attention_pool(
            title_enhanced,
            self.title_pool_attn,
            title_attention_mask
        )  # [batch, 768]

        abstract_pooled = self.attention_pool(
            abstract_enhanced,
            self.abstract_pool_attn,
            abstract_attention_mask
        )  # [batch, 768]

        # 4. Concatenate and classify
        combined = torch.cat([title_pooled, abstract_pooled], dim=1)  # [batch, 1536]
        output = self.fusion(combined)  # [batch, 4]

        return output


def compare_architectures():
    """
    Comparar arquitectura original vs cross-attention
    """
    print("="*70)
    print("ARCHITECTURE COMPARISON")
    print("="*70)

    print("\nV3.7 Original (Simple Concatenation):")
    print("  title_hidden -> attention_pool -> title_pooled [768]")
    print("  abstract_hidden -> attention_pool -> abstract_pooled [768]")
    print("  concat([title_pooled, abstract_pooled]) -> [1536]")
    print("  fusion_network -> [4 classes]")
    print("  ")
    print("  Limitation: No interaction between title and abstract")

    print("\nCross-Attention (This Implementation):")
    print("  title_hidden -> cross_attn(query=title, key/value=abstract) -> title_enhanced")
    print("  abstract_hidden -> cross_attn(query=abstract, key/value=title) -> abstract_enhanced")
    print("  title_enhanced -> attention_pool -> title_pooled [768]")
    print("  abstract_enhanced -> attention_pool -> abstract_pooled [768]")
    print("  concat([title_pooled, abstract_pooled]) -> [1536]")
    print("  fusion_network -> [4 classes]")
    print("  ")
    print("  Advantage: Title and abstract inform each other!")

    print("\n" + "="*70)
    print("PARAMETER COUNT COMPARISON")
    print("="*70)

    # V3.7 original
    v37_params = 110e6  # SciBERT
    v37_params += 768 + 768  # Attention pooling
    v37_params += (1536*512 + 512*256 + 256*128 + 128*4)  # Fusion

    # Cross-attention
    cross_params = 110e6  # SciBERT
    cross_params += 2 * (768*768*8 + 768)  # 2 cross-attention layers
    cross_params += 768 + 768  # Attention pooling
    cross_params += (1536*512 + 512*256 + 256*128 + 128*4)  # Fusion

    print(f"\nV3.7 Original: ~{v37_params/1e6:.1f}M parameters")
    print(f"Cross-Attention: ~{cross_params/1e6:.1f}M parameters")
    print(f"Increase: ~{(cross_params-v37_params)/1e6:.1f}M parameters (+{(cross_params/v37_params-1)*100:.1f}%)")

    print("\nNote: Aumento marginal en parámetros (~10%)")
    print("      pero mejora significativa en capacidad de interacción")

    print("\n" + "="*70)


if __name__ == "__main__":
    compare_architectures()

    print("\n" + "="*70)
    print("USAGE EXAMPLE")
    print("="*70)

    print("""
# Replace model in train_scibert_optimized.py:

from advanced_cross_attention import CrossAttentionSciBERT

# Instead of:
# model = OptimizedSciBERTClassifier(...)

# Use:
model = CrossAttentionSciBERT(
    num_classes=4,
    dropout=0.35,
    freeze_bert_layers=3
)

# Rest of training code remains the same
trainer = OptimizedTrainer(model, device, ...)
trainer.train(...)

# Expected improvement: +1-2% accuracy
# Training time: Same as V3.7 (~60-80 min on M2)
""")

    print("="*70)
