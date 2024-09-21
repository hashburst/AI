from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Carica il tokenizer e il modello pre-addestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Prepara i dati di input
texts = ["Hello, how are you?", "I am fine, thank you!"]
labels = [0, 1]

# Tokenizza i testi
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Esegui la predizione
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

print(predictions)
