#Для быстрого обучения лучше запустить в Colab с графическим процессором Т4
#https://colab.research.google.com/drive/1dbbTGjTkn5Vp5WcdzlOLN9-MK9NRaPty?usp=sharing


import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from tqdm import tqdm

DATA_PATH = "dataset.json"
MODEL_NAME = "cointegrated/rubert-tiny2"
EPOCHS = 3
BATCH_SIZE = 8 

label2id = {
    "O": 0, 
    "B-TERM": 1, "I-TERM": 2, 
    "B-FORMULA": 3, "I-FORMULA": 4, 
    "B-NAME": 5, "I-NAME": 6
}
id2label = {v: k for k, v in label2id.items()}

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item['entities']

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        labels = torch.ones(self.max_len, dtype=torch.long) * -100
        offsets = encoding.pop('offset_mapping')[0]
        
        for i, (start_char, end_char) in enumerate(offsets):
            if start_char == 0 and end_char == 0:
                continue
                
            labels[i] = label2id["O"]
            
            for ent in entities:
                ent_label = ent.get('label', 'TERM') # Читаем метку из нашего умного датасета
                
                if start_char == ent['start']:
                    labels[i] = label2id[f"B-{ent_label}"]
                elif ent['start'] < start_char < ent['end']:
                    labels[i] = label2id[f"I-{ent_label}"]
                
        item_tensors = {key: val[0] for key, val in encoding.items()}
        item_tensors['labels'] = labels
        return item_tensors

def train_model():
    print("Читаю датасет")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset_json = json.load(f)

    print("Загружаю токенизатор")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Загружаю саму нейросеть RuBERT-tiny2")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=7, 
        id2label=id2label, 
        label2id=label2id
    )

    dataset = NERDataset(dataset_json, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Обучение будет проходить на: {device}")
    model.to(device)

    model.train()
    for epoch in range(EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{EPOCHS}")
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Обучение")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Средняя ошибка (loss) за эпоху: {total_loss / len(dataloader):.4f}")

    print("Обучение завершено! Сохраняю модель")
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Модель успешно сохранена в папку ./saved_model")

if __name__ == "__main__":
    train_model()