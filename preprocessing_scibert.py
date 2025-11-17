# preprocessing_scibert.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle

class SciBERTDataset(Dataset):
    """Dataset para modelo dual (ttulo + abstract separados)"""
    def __init__(self, titles, abstracts, labels, tokenizer,
                 max_title_len=32, max_abstract_len=128):
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_title_len = max_title_len
        self.max_abstract_len = max_abstract_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenizar ttulo
        title_encoding = self.tokenizer(
            self.titles[idx],
            max_length=self.max_title_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenizar abstract
        abstract_encoding = self.tokenizer(
            self.abstracts[idx],
            max_length=self.max_abstract_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'title_input_ids': title_encoding['input_ids'].squeeze(0),
            'title_attention_mask': title_encoding['attention_mask'].squeeze(0),
            'abstract_input_ids': abstract_encoding['input_ids'].squeeze(0),
            'abstract_attention_mask': abstract_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class SciBERTLightDataset(Dataset):
    """Dataset para modelo ligero (ttulo + abstract combinados)"""
    def __init__(self, titles, abstracts, labels, tokenizer, max_len=160):
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Combinar ttulo y abstract con [SEP]
        combined_text = f"{self.titles[idx]} [SEP] {self.abstracts[idx]}"

        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def prepare_scibert_data(use_light_model=False, test_size=0.15, val_size=0.15):
    """
    Prepara datos para modelos SciBERT

    Args:
        use_light_model: Si True, usa modelo ligero (texto combinado)
                        Si False, usa modelo dual (ttulo + abstract separados)
        test_size: Fraccin para test set
        val_size: Fraccin para validation set

    Returns:
        train_dataset, val_dataset, test_dataset, tokenizer, label_encoder
    """
    df = pd.read_csv('data/arxiv_papers_raw.csv')

    # Codificar labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    # Cargar tokenizer de SciBERT
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    # Divisin de datos
    X = df[['title', 'abstract']]
    y = df['label']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Calcular val_size ajustado
    adjusted_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=42, stratify=y_temp
    )

    # Crear datasets

    if use_light_model:
        train_dataset = SciBERTLightDataset(
            X_train['title'].tolist(),
            X_train['abstract'].tolist(),
            y_train.tolist(),
            tokenizer,
            max_len=160
        )

        val_dataset = SciBERTLightDataset(
            X_val['title'].tolist(),
            X_val['abstract'].tolist(),
            y_val.tolist(),
            tokenizer,
            max_len=160
        )

        test_dataset = SciBERTLightDataset(
            X_test['title'].tolist(),
            X_test['abstract'].tolist(),
            y_test.tolist(),
            tokenizer,
            max_len=160
        )
    else:
        train_dataset = SciBERTDataset(
            X_train['title'].tolist(),
            X_train['abstract'].tolist(),
            y_train.tolist(),
            tokenizer,
            max_title_len=32,
            max_abstract_len=128
        )

        val_dataset = SciBERTDataset(
            X_val['title'].tolist(),
            X_val['abstract'].tolist(),
            y_val.tolist(),
            tokenizer,
            max_title_len=32,
            max_abstract_len=128
        )

        test_dataset = SciBERTDataset(
            X_test['title'].tolist(),
            X_test['abstract'].tolist(),
            y_test.tolist(),
            tokenizer,
            max_title_len=32,
            max_abstract_len=128
        )

    # Guardar label encoder
    print("\n Guardando label encoder...")
    with open('scibert_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(" Preparacin completa!")

    return train_dataset, val_dataset, test_dataset, tokenizer, le


def test_dataset():
    """Test rpido del dataset"""
    print(" Testing SciBERT dataset preparation...")

    # Test modelo dual
    print("\n--- Modelo Dual ---")
    train_ds, val_ds, test_ds, tokenizer, le = prepare_scibert_data(use_light_model=False)

    # Verificar un batch
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(train_loader))

    print(f"\n Batch del modelo dual:")
    print(f"   title_input_ids: {batch['title_input_ids'].shape}")
    print(f"   title_attention_mask: {batch['title_attention_mask'].shape}")
    print(f"   abstract_input_ids: {batch['abstract_input_ids'].shape}")
    print(f"   abstract_attention_mask: {batch['abstract_attention_mask'].shape}")
    print(f"   label: {batch['label'].shape}")

    # Decodificar ejemplo
    print(f"\n Ejemplo de ttulo tokenizado:")
    print(f"   Tokens: {batch['title_input_ids'][0][:20]}")
    print(f"   Texto: {tokenizer.decode(batch['title_input_ids'][0][:20])}")

    # Test modelo ligero
    print("\n\n--- Modelo Ligero ---")
    train_ds_light, _, _, _, _ = prepare_scibert_data(use_light_model=True)

    train_loader_light = DataLoader(train_ds_light, batch_size=4, shuffle=True)
    batch_light = next(iter(train_loader_light))

    print(f"\n Batch del modelo ligero:")
    print(f"   input_ids: {batch_light['input_ids'].shape}")
    print(f"   attention_mask: {batch_light['attention_mask'].shape}")
    print(f"   label: {batch_light['label'].shape}")

    print(f"\n Ejemplo de texto combinado:")
    print(f"   Texto: {tokenizer.decode(batch_light['input_ids'][0][:50])}...")

    print("\n Dataset test completado!")


if __name__ == "__main__":
    test_dataset()
