"""
Hybrid CNN-LSTM Model for Academic Paper Classification

Architecture according to project definition:
- CNN 1D for abstract feature extraction
- Bidirectional LSTM for title sequential processing
- Self-attention over LSTM outputs
- Global attention over CNN features
- Weighted attention fusion layer
- Variational dropout and batch normalization

PyTorch implementation from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism for LSTM outputs"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] - 1 for valid positions, 0 for padding
        Returns:
            context: [batch, hidden_dim]
            attention_weights: [batch, seq_len]
        """
        # Calculate attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax over sequence length
        attention_weights = F.softmax(scores, dim=1)  # [batch, seq_len]

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, seq_len]
            lstm_output  # [batch, seq_len, hidden_dim]
        ).squeeze(1)  # [batch, hidden_dim]

        return context, attention_weights


class GlobalAttention(nn.Module):
    """Global attention over CNN feature maps"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, cnn_features):
        """
        Args:
            cnn_features: [batch, num_filters, feature_len]
        Returns:
            context: [batch, num_filters]
            attention_weights: [batch, feature_len]
        """
        # Transpose to [batch, feature_len, num_filters]
        features_t = cnn_features.transpose(1, 2)

        # Calculate attention scores
        scores = self.attention(features_t).squeeze(-1)  # [batch, feature_len]

        # Softmax
        attention_weights = F.softmax(scores, dim=1)  # [batch, feature_len]

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, feature_len]
            features_t  # [batch, feature_len, num_filters]
        ).squeeze(1)  # [batch, num_filters]

        return context, attention_weights


class WeightedAttentionFusion(nn.Module):
    """Learns relative importance of title and abstract representations"""
    def __init__(self, title_dim, abstract_dim):
        super().__init__()
        self.title_weight = nn.Linear(title_dim, 1)
        self.abstract_weight = nn.Linear(abstract_dim, 1)

    def forward(self, title_repr, abstract_repr):
        """
        Args:
            title_repr: [batch, title_dim]
            abstract_repr: [batch, abstract_dim]
        Returns:
            fused: [batch, title_dim + abstract_dim]
            fusion_weights: [batch, 2] - [title_weight, abstract_weight]
        """
        # Calculate importance weights
        w_title = self.title_weight(title_repr)  # [batch, 1]
        w_abstract = self.abstract_weight(abstract_repr)  # [batch, 1]

        # Concatenate and softmax
        weights = torch.cat([w_title, w_abstract], dim=1)  # [batch, 2]
        fusion_weights = F.softmax(weights, dim=1)  # [batch, 2]

        # Apply weights
        weighted_title = title_repr * fusion_weights[:, 0:1]
        weighted_abstract = abstract_repr * fusion_weights[:, 1:2]

        # Concatenate weighted representations
        fused = torch.cat([weighted_title, weighted_abstract], dim=1)

        return fused, fusion_weights


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for academic paper classification

    Components:
    - Embedding layer (shared for title and abstract)
    - CNN 1D for abstract processing
    - Bidirectional LSTM for title processing
    - Self-attention over LSTM outputs
    - Global attention over CNN features
    - Weighted attention fusion
    - Classification head with variational dropout
    """
    def __init__(self,
                 vocab_size,
                 embed_dim=300,
                 num_filters=256,
                 kernel_sizes=[3, 4, 5],
                 lstm_hidden=256,
                 num_classes=4,
                 dropout=0.5,
                 pretrained_embeddings=None):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout * 0.5)

        # CNN 1D for abstracts (multiple kernel sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        # Global attention over CNN features
        total_filters = num_filters * len(kernel_sizes)
        self.cnn_attention = GlobalAttention(total_filters)

        # Bidirectional LSTM for titles
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Self-attention over LSTM outputs
        self.lstm_attention = SelfAttention(lstm_hidden * 2)  # *2 for bidirectional

        # Weighted attention fusion
        self.fusion = WeightedAttentionFusion(
            title_dim=lstm_hidden * 2,
            abstract_dim=total_filters
        )

        # Classification head
        fused_dim = lstm_hidden * 2 + total_filters
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, title_ids, abstract_ids, title_mask=None):
        """
        Args:
            title_ids: [batch, title_len]
            abstract_ids: [batch, abstract_len]
            title_mask: [batch, title_len] - optional mask for padding

        Returns:
            logits: [batch, num_classes]
            attention_maps: dict with attention weights for visualization
        """
        batch_size = title_ids.size(0)

        # === TITLE PROCESSING (LSTM + Self-Attention) ===
        title_embed = self.embedding(title_ids)  # [batch, title_len, embed_dim]
        title_embed = self.embed_dropout(title_embed)

        # Bidirectional LSTM
        lstm_out, _ = self.lstm(title_embed)  # [batch, title_len, lstm_hidden*2]

        # Self-attention over LSTM outputs
        title_repr, title_attn_weights = self.lstm_attention(lstm_out, title_mask)
        # title_repr: [batch, lstm_hidden*2]

        # === ABSTRACT PROCESSING (CNN + Global Attention) ===
        abstract_embed = self.embedding(abstract_ids)  # [batch, abstract_len, embed_dim]
        abstract_embed = self.embed_dropout(abstract_embed)

        # Transpose for Conv1d: [batch, embed_dim, abstract_len]
        abstract_embed = abstract_embed.transpose(1, 2)

        # Apply multiple CNN filters
        conv_outputs = []
        for conv, bn in zip(self.convs, self.conv_bn):
            x = conv(abstract_embed)  # [batch, num_filters, abstract_len]
            x = bn(x)
            x = F.relu(x)
            conv_outputs.append(x)

        # Concatenate filter outputs
        cnn_features = torch.cat(conv_outputs, dim=1)  # [batch, total_filters, abstract_len]

        # Global attention over CNN features
        abstract_repr, abstract_attn_weights = self.cnn_attention(cnn_features)
        # abstract_repr: [batch, total_filters]

        # === WEIGHTED ATTENTION FUSION ===
        fused_repr, fusion_weights = self.fusion(title_repr, abstract_repr)
        # fused_repr: [batch, lstm_hidden*2 + total_filters]

        # === CLASSIFICATION ===
        logits = self.classifier(fused_repr)  # [batch, num_classes]

        # Store attention maps for visualization
        attention_maps = {
            'title_attention': title_attn_weights,  # [batch, title_len]
            'abstract_attention': abstract_attn_weights,  # [batch, abstract_len]
            'fusion_weights': fusion_weights  # [batch, 2]
        }

        return logits, attention_maps


if __name__ == "__main__":
    # Test model
    print("Testing Hybrid CNN-LSTM model...")

    model = HybridCNNLSTM(
        vocab_size=10000,
        embed_dim=300,
        num_filters=256,
        kernel_sizes=[3, 4, 5],
        lstm_hidden=256,
        num_classes=4,
        dropout=0.5
    )

    # Dummy inputs
    batch_size = 16
    title_ids = torch.randint(1, 1000, (batch_size, 20))
    abstract_ids = torch.randint(1, 1000, (batch_size, 150))
    title_mask = torch.ones(batch_size, 20)

    # Forward pass
    logits, attention_maps = model(title_ids, abstract_ids, title_mask)

    print(f"Input shapes:")
    print(f"  Title: {title_ids.shape}")
    print(f"  Abstract: {abstract_ids.shape}")
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Title attention: {attention_maps['title_attention'].shape}")
    print(f"  Abstract attention: {attention_maps['abstract_attention'].shape}")
    print(f"  Fusion weights: {attention_maps['fusion_weights'].shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nOK Model test passed!")
