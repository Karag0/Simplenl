import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import numpy as np
import random
import math
import warnings

# Подавляем предупреждение о nested tensors
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

# Загрузка и подготовка данных
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

# Параметры
VOCAB_SIZE = 10000
BATCH_SIZE = 16  # batch size
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# Построение словаря
def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    for idx, (word, count) in enumerate(counter.most_common(vocab_size - 4)):
        vocab[word] = idx + 4
    return vocab

vocab = build_vocab(train_data['text'], VOCAB_SIZE)

# Векторизация текста
def text_to_vector(text, vocab):
    tokens = text.lower().split()
    vector = [vocab['<sos>']] + [vocab.get(token, vocab['<unk>']) for token in tokens] + [vocab['<eos>']]
    return vector

# Позиционное кодирование
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Добавляем позиционное кодирование для всех позиций в последовательности
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Упрощенная версия трансформера
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Создаем свои слои внимания
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.norm_layers1 = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_layers)
        ])
        
        self.norm_layers2 = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, 1)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
    def forward(self, x, padding_mask=None):
        # Эмбеддинг
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        
        # Пропускаем через кастомные слои внимания
        for i in range(self.num_layers):
            # Self-attention
            attn_output, _ = self.self_attentions[i](
                x, x, x, 
                key_padding_mask=padding_mask,
                need_weights=False
            )
            x = self.norm_layers1[i](x + self.dropout(attn_output))
            
            # Feed-forward
            ff_output = self.linear_layers[i](x)
            x = self.norm_layers2[i](x + self.dropout(ff_output))
        
        # Берем выход только для первого токена (<sos>)
        x = x[:, 0, :]
        
        # Классификация
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()

# Dataset класс
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.texts = [text_to_vector(text, vocab) for text in data['text']]
        self.labels = data['label']
        self.raw_texts = data['text']
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx]), self.raw_texts[idx]

# Collate function для обработки последовательностей разной длины
def collate_fn(batch):
    texts, labels, raw_texts = zip(*batch)
    lengths = [len(t) for t in texts]
    
    # Паддинг до максимальной длины в батче
    padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    
    # Создаем маску для паддинга
    padding_mask = (padded == 0)
    
    return padded, torch.tensor(labels), padding_mask, raw_texts

# Инициализация всего
train_dataset = ReviewDataset(train_data)
test_dataset = ReviewDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleTransformerClassifier(
    len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT
).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for texts, labels, padding_mask, _ in train_loader:
        texts, labels = texts.to(device), labels.float().to(device)
        padding_mask = padding_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts, padding_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# Тестирование
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []
all_texts = []

with torch.no_grad():
    for texts, labels, padding_mask, raw_texts in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        padding_mask = padding_mask.to(device)
        outputs = model(texts, padding_mask)
        predicted = (outputs > 0.5).long()
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Сохраняем предсказания для анализа
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_texts.extend(raw_texts)

print(f'Accuracy: {100 * correct / total:.2f}%')

# Функция для предсказания тональности
def predict_sentiment(text, model, vocab):
    model.eval()
    vector = text_to_vector(text, vocab)
    tensor = torch.tensor(vector).unsqueeze(0).to(device)
    
    # Создаем маску для паддинга (в данном случае не нужна, так как одна последовательность)
    padding_mask = torch.zeros(1, len(vector), dtype=torch.bool).to(device)
    
    output = model(tensor, padding_mask)
    pred = (output > 0.5).item()
    confidence = output.item() if pred else 1 - output.item()
    return "Positive" if pred == 1 else "Negative", confidence

# Вывод примеров предсказаний
print("\nПримеры предсказаний:")
print("-" * 80)

# Выбираем случайные примеры из тестового набора
indices = random.sample(range(len(all_texts)), 10)

for i, idx in enumerate(indices):
    text = all_texts[idx]
    # Обрезаем текст для удобства отображения
    short_text = text[:150] + "..." if len(text) > 150 else text
    true_label = "Positive" if all_labels[idx] == 1 else "Negative"
    pred_label = "Positive" if all_predictions[idx] == 1 else "Negative"
    
    # Получаем уверенность модели для этого примера
    _, confidence = predict_sentiment(text, model, vocab)
    
    print(f"Пример {i+1}:")
    print(f"Текст: {short_text}")
    print(f"Истинная метка: {true_label}")
    print(f"Предсказание: {pred_label}")
    print(f"Уверенность: {confidence:.2f}")
    print("-" * 80)

# Тест на пользовательских примерах
print("\nТест на пользовательских примерах:") # Вы можете сюда вставить свои отзывы. 
test_examples = [
    "This movie was absolutely wonderful! I loved every minute of it.",
    "Terrible film. Waste of time and money.",
    "It was okay, nothing special.",
    "The acting was great but the plot was weak.",
    "One of the best movies I've seen this year!"
]

for i, example in enumerate(test_examples):
    prediction, confidence = predict_sentiment(example, model, vocab)
    print(f"Пример {i+1}: '{example}'")
    print(f"Предсказание: {prediction} (уверенность: {confidence:.2f})")
    print()

# Анализ ошибок
print("\nАнализ ошибок:")
error_indices = [i for i, (p, l) in enumerate(zip(all_predictions, all_labels)) if p != l]

if error_indices:
    # Выбираем несколько случайных ошибок
    error_samples = random.sample(error_indices, min(5, len(error_indices)))
    
    for i, idx in enumerate(error_samples):
        text = all_texts[idx]
        short_text = text[:150] + "..." if len(text) > 150 else text
        true_label = "Positive" if all_labels[idx] == 1 else "Negative"
        pred_label = "Positive" if all_predictions[idx] == 1 else "Negative"
        
        print(f"Ошибка {i+1}:")
        print(f"Текст: {short_text}")
        print(f"Истинная метка: {true_label}")
        print(f"Предсказание: {pred_label}")
        print("-" * 80)
else:
    print("Нет ошибок в предсказаниях!")

# Сохранение модели
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'model_params': {
        'vocab_size': len(vocab),
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
}, 'transformer_sentiment_model.pth')

print("Модель сохранена как 'transformer_sentiment_model.pth'")
