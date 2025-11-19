"""
Data preprocessing for Hybrid CNN-LSTM model

- Build vocabulary from titles and abstracts
- Tokenize text
- Create datasets with proper padding
- No external NLP libraries (pure PyTorch)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Vocabulary:
    """Simple vocabulary builder"""
    def __init__(self, max_vocab_size=50000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()

    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        print("Building vocabulary...")

        # Count words
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)

        # Filter by min_freq and max_vocab_size
        filtered_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.min_freq
        ][:self.max_vocab_size - 2]  # -2 for PAD and UNK

        # Build word2idx and idx2word
        for idx, word in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {filtered_words[:10]}")

    @staticmethod
    def tokenize(text):
        """Simple tokenization"""
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        # Split and filter empty
        words = [w.strip() for w in text.split() if w.strip()]
        return words

    def encode(self, text, max_len=None):
        """Convert text to token indices"""
        words = self.tokenize(text)
        if max_len:
            words = words[:max_len]
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 = UNK
        return indices

    def decode(self, indices):
        """Convert indices back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices if idx != 0]
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        """Save vocabulary"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load vocabulary"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class HybridDataset(Dataset):
    """Dataset for Hybrid CNN-LSTM model"""
    def __init__(self, titles, abstracts, labels, vocab, max_title_len=30, max_abstract_len=200):
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels
        self.vocab = vocab
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Encode and pad title
        title_ids = self.vocab.encode(self.titles[idx], self.max_title_len)
        title_len = len(title_ids)
        title_ids += [0] * (self.max_title_len - title_len)  # Pad with 0

        # Encode and pad abstract
        abstract_ids = self.vocab.encode(self.abstracts[idx], self.max_abstract_len)
        abstract_len = len(abstract_ids)
        abstract_ids += [0] * (self.max_abstract_len - abstract_len)

        # Create mask for title (1 for real tokens, 0 for padding)
        title_mask = [1] * title_len + [0] * (self.max_title_len - title_len)

        return {
            'title_ids': torch.tensor(title_ids, dtype=torch.long),
            'abstract_ids': torch.tensor(abstract_ids, dtype=torch.long),
            'title_mask': torch.tensor(title_mask, dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def prepare_hybrid_data(data_path='data/arxiv_papers_raw.csv',
                       vocab_path='vocab_hybrid.pkl',
                       test_size=0.15,
                       val_size=0.15,
                       max_vocab_size=50000,
                       min_freq=2,
                       max_title_len=30,
                       max_abstract_len=200):
    """
    Prepare data for Hybrid CNN-LSTM model

    Returns:
        train_dataset, val_dataset, test_dataset, vocab, label_encoder
    """
    print("="*70)
    print("PREPARING DATA FOR HYBRID CNN-LSTM MODEL")
    print("="*70)

    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} samples")
    print(f"Class distribution:\n{df['category'].value_counts()}\n")

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df['category'])

    # Build or load vocabulary
    try:
        vocab = Vocabulary.load(vocab_path)
        print(f"Loaded vocabulary from {vocab_path}")
        print(f"Vocabulary size: {len(vocab)}")
    except FileNotFoundError:
        print(f"Building vocabulary...")
        vocab = Vocabulary(max_vocab_size=max_vocab_size, min_freq=min_freq)

        # Build vocab from all texts
        all_texts = df['title'].tolist() + df['abstract'].tolist()
        vocab.build_vocab(all_texts)

        # Save vocabulary
        vocab.save(vocab_path)
        print(f"Vocabulary saved to {vocab_path}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[['title', 'abstract']].values,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=y_temp
    )

    # Create datasets
    train_dataset = HybridDataset(
        titles=X_train[:, 0],
        abstracts=X_train[:, 1],
        labels=y_train,
        vocab=vocab,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len
    )

    val_dataset = HybridDataset(
        titles=X_val[:, 0],
        abstracts=X_val[:, 1],
        labels=y_val,
        vocab=vocab,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len
    )

    test_dataset = HybridDataset(
        titles=X_test[:, 0],
        abstracts=X_test[:, 1],
        labels=y_test,
        vocab=vocab,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Show example
    print(f"\nExample (first training sample):")
    sample = train_dataset[0]
    print(f"  Title length: {sample['title_mask'].sum().int().item()} tokens")
    print(f"  Abstract length: {(sample['abstract_ids'] != 0).sum().item()} tokens")
    print(f"  Label: {le.classes_[sample['label']]}")

    return train_dataset, val_dataset, test_dataset, vocab, le


if __name__ == "__main__":
    # Test preprocessing
    import os

    if not os.path.exists('data/arxiv_papers_raw.csv'):
        print("ERROR: data/arxiv_papers_raw.csv not found")
        exit(1)

    train_ds, val_ds, test_ds, vocab, le = prepare_hybrid_data()

    # Test sample
    sample = train_ds[0]
    print(f"\nSample tensors:")
    print(f"  title_ids shape: {sample['title_ids'].shape}")
    print(f"  abstract_ids shape: {sample['abstract_ids'].shape}")
    print(f"  title_mask shape: {sample['title_mask'].shape}")
    print(f"  label: {sample['label']}")

    # Decode sample
    print(f"\nDecoded title: {vocab.decode(sample['title_ids'].tolist()[:10])}")
    print(f"Decoded abstract: {vocab.decode(sample['abstract_ids'].tolist()[:20])}")

    print("\nOK Preprocessing test passed!")
