"""
Attention Maps Visualization for Hybrid CNN-LSTM

Visualize:
- Self-attention weights over LSTM (title)
- Global attention weights over CNN features (abstract)
- Weighted fusion weights (title vs abstract importance)
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from hybrid_cnn_lstm import HybridCNNLSTM
from preprocessing_hybrid import Vocabulary
import pickle


class AttentionVisualizer:
    """Visualize attention maps from Hybrid CNN-LSTM model"""

    def __init__(self, model_path, vocab_path):
        """
        Args:
            model_path: Path to saved model
            vocab_path: Path to vocabulary
        """
        # Load vocabulary
        self.vocab = Vocabulary.load(vocab_path)

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab_size = checkpoint['vocab_size']

        self.model = HybridCNNLSTM(
            vocab_size=vocab_size,
            embed_dim=300,
            num_filters=256,
            kernel_sizes=[3, 4, 5],
            lstm_hidden=256,
            num_classes=4,
            dropout=0.5
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.class_names = checkpoint['class_names']

    def predict_with_attention(self, title, abstract, max_title_len=30, max_abstract_len=200):
        """
        Predict class and return attention maps

        Returns:
            prediction, attention_maps, probabilities
        """
        # Encode title
        title_ids = self.vocab.encode(title, max_title_len)
        title_len = len(title_ids)
        title_words = self.vocab.tokenize(title)[:max_title_len]
        title_ids += [0] * (max_title_len - title_len)

        # Encode abstract
        abstract_ids = self.vocab.encode(abstract, max_abstract_len)
        abstract_len = len(abstract_ids)
        abstract_words = self.vocab.tokenize(abstract)[:max_abstract_len]
        abstract_ids += [0] * (max_abstract_len - abstract_len)

        # Create mask
        title_mask = [1] * title_len + [0] * (max_title_len - title_len)

        # Convert to tensors
        title_tensor = torch.tensor(title_ids, dtype=torch.long).unsqueeze(0)
        abstract_tensor = torch.tensor(abstract_ids, dtype=torch.long).unsqueeze(0)
        mask_tensor = torch.tensor(title_mask, dtype=torch.float).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            logits, attention_maps = self.model(title_tensor, abstract_tensor, mask_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
            prediction = torch.argmax(logits, dim=1).item()

        # Extract attention weights
        title_attn = attention_maps['title_attention'].squeeze(0).numpy()[:title_len]
        abstract_attn = attention_maps['abstract_attention'].squeeze(0).numpy()[:abstract_len]
        fusion_weights = attention_maps['fusion_weights'].squeeze(0).numpy()

        return {
            'prediction': self.class_names[prediction],
            'probabilities': probabilities,
            'title_words': title_words,
            'abstract_words': abstract_words,
            'title_attention': title_attn,
            'abstract_attention': abstract_attn,
            'fusion_weights': fusion_weights
        }

    def plot_attention_map(self, words, attention_weights, title, max_words=50):
        """Plot attention heatmap for a sequence"""
        # Limit words for visualization
        if len(words) > max_words:
            words = words[:max_words]
            attention_weights = attention_weights[:max_words]

        # Normalize attention weights
        attention_weights = attention_weights / attention_weights.max()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 2))

        # Create color map
        colors = ['white', 'lightblue', 'blue', 'darkblue']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)

        # Plot heatmap
        attention_2d = attention_weights.reshape(1, -1)
        sns.heatmap(
            attention_2d,
            xticklabels=words,
            yticklabels=False,
            cmap=cmap,
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Words', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()

        return fig

    def plot_fusion_weights(self, fusion_weights):
        """Plot fusion weights (title vs abstract importance)"""
        fig, ax = plt.subplots(figsize=(6, 4))

        modalities = ['Title', 'Abstract']
        weights = fusion_weights

        colors = ['skyblue', 'lightcoral']
        bars = ax.bar(modalities, weights, color=colors, alpha=0.7, edgecolor='black')

        # Add percentage labels
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{weight*100:.1f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        ax.set_ylabel('Fusion Weight', fontsize=11)
        ax.set_title('Weighted Attention Fusion\n(Learned Importance)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_sample(self, title, abstract, save_path=None):
        """
        Complete visualization for a sample

        Shows:
        - Title attention map
        - Abstract attention map (first 100 words)
        - Fusion weights
        - Prediction probabilities
        """
        # Get prediction and attention
        result = self.predict_with_attention(title, abstract)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

        # 1. Title attention
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_attention_inline(
            ax1,
            result['title_words'],
            result['title_attention'],
            'Title Self-Attention (LSTM)',
            max_words=30
        )

        # 2. Abstract attention (first 100 words)
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_attention_inline(
            ax2,
            result['abstract_words'],
            result['abstract_attention'],
            'Abstract Global Attention (CNN)',
            max_words=100
        )

        # 3. Fusion weights
        ax3 = fig.add_subplot(gs[2, 0])
        modalities = ['Title', 'Abstract']
        weights = result['fusion_weights']
        colors = ['skyblue', 'lightcoral']
        bars = ax3.bar(modalities, weights, color=colors, alpha=0.7, edgecolor='black')
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{weight*100:.1f}%',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        ax3.set_ylabel('Fusion Weight', fontsize=10)
        ax3.set_title('Weighted Fusion', fontsize=11, fontweight='bold')
        ax3.set_ylim(0, 1.0)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Prediction probabilities
        ax4 = fig.add_subplot(gs[2, 1])
        probs = result['probabilities']
        colors_prob = ['lightgreen' if i == np.argmax(probs) else 'lightgray'
                       for i in range(len(probs))]
        bars = ax4.barh(self.class_names, probs, color=colors_prob, alpha=0.7, edgecolor='black')
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax4.text(
                width,
                bar.get_y() + bar.get_height() / 2.,
                f'{prob*100:.1f}%',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        ax4.set_xlabel('Probability', fontsize=10)
        ax4.set_title(f'Prediction: {result["prediction"]}', fontsize=11, fontweight='bold')
        ax4.set_xlim(0, 1.0)
        ax4.grid(axis='x', alpha=0.3)

        # 5. Sample text
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')
        text_content = f"Title:\n{title}\n\nAbstract (first 200 chars):\n{abstract[:200]}..."
        ax5.text(0, 0.5, text_content, fontsize=9, verticalalignment='center',
                fontfamily='monospace', wrap=True)

        plt.suptitle('Hybrid CNN-LSTM Attention Visualization', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"OK Visualization saved: {save_path}")

        plt.show()
        return fig

    def _plot_attention_inline(self, ax, words, attention_weights, title, max_words=50):
        """Helper to plot attention in existing axis"""
        # Limit words
        if len(words) > max_words:
            words = words[:max_words]
            attention_weights = attention_weights[:max_words]

        # Normalize
        attention_weights = attention_weights / (attention_weights.max() + 1e-8)

        # Create color map
        colors = ['white', 'lightblue', 'blue', 'darkblue']
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=100)

        # Plot heatmap
        attention_2d = attention_weights.reshape(1, -1)
        sns.heatmap(
            attention_2d,
            xticklabels=words,
            yticklabels=False,
            cmap=cmap,
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Words', fontsize=9)
        ax.tick_params(axis='x', labelsize=7, rotation=45)


if __name__ == "__main__":
    import sys

    # Example usage
    print("Testing attention visualization...")

    # Check if model exists
    import os
    if not os.path.exists('best_hybrid_model.pth'):
        print("ERROR: Model not found. Train the model first with: python train_hybrid.py")
        sys.exit(1)

    if not os.path.exists('vocab_hybrid.pkl'):
        print("ERROR: Vocabulary not found.")
        sys.exit(1)

    # Load visualizer
    viz = AttentionVisualizer('best_hybrid_model.pth', 'vocab_hybrid.pkl')

    # Example paper
    title = "Deep Learning for Computer Vision"
    abstract = "We propose a novel convolutional neural network architecture for image classification tasks. " \
               "Our method achieves state-of-the-art results on ImageNet dataset."

    # Visualize
    viz.visualize_sample(title, abstract, save_path='attention_example.png')

    print("\nOK Visualization test passed!")
