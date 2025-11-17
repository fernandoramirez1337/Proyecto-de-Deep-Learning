# test_pipeline.py
import torch
from preprocessing import prepare_data
from model import MultimodalAttentionClassifier
from torch.utils.data import DataLoader

# Preparar datos
train_dataset, val_dataset, test_dataset, preprocessor, le = prepare_data()

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Crear modelo
vocab_size = len(preprocessor.vocab)
num_classes = len(le.classes_)
model = MultimodalAttentionClassifier(vocab_size, num_classes)

# Verificar dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Device: {device}")

# Test forward pass
batch = next(iter(train_loader))
titles = batch['title'].to(device)
abstracts = batch['abstract'].to(device)
labels = batch['label'].to(device)

output, attention_weights = model(titles, abstracts)

# Calcular loss
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, labels)
print(f"Forward pass exitoso! Output: {output.shape}, Loss inicial: {loss.item():.4f}")