import torch
from transformers import AutoTokenizer

class dataset_processor:
    def __init__(self, model_name="cointegrated/rubert-tiny2", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        # Маппинг тегов в индексы для модели
        self.label_map = {"O": 0, "B-TERM": 1, "I-TERM": 2}

    def tokenize_and_align_labels(self, text, entities):
        """
        text: сырой текст статьи
        entities: список словарей [{'start': 0, 'end': 10, 'label': 'TERM'}]
        """
        tokenized_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = tokenized_input.pop("offset_mapping")[0]
        input_ids = tokenized_input["input_ids"][0]
        
        # Заполняем массив меток значением -100 (игнорируется при расчете Loss)
        labels = torch.full((self.max_length,), -100, dtype=torch.long)

        for i, offset in enumerate(offsets):
            start_char, end_char = offset
            
            # Пропускаем паддинги и служебные токены
            if start_char == 0 and end_char == 0:
                continue

            # Определяем, попадает ли токен в границы сущности
            current_label = "O"
            for ent in entities:
                if start_char >= ent['start'] and end_char <= ent['end']:
                    # Если это начало сущности (первый токен слова)
                    if start_char == ent['start']:
                        current_label = f"B-{ent['label']}"
                    else:
                        current_label = f"I-{ent['label']}"
                    break
            
            labels[i] = self.label_map.get(current_label, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized_input["attention_mask"][0],
            "labels": labels
        }

if __name__ == "__main__":
    processor = dataset_processor()
    
    example_text = "Абелев дифференциал определяется на римановой поверхности."
    example_entities = [
        {"start": 0, "end": 19, "label": "TERM"},  # Абелев дифференциал
        {"start": 36, "end": 57, "label": "TERM"} # римановой поверхности
    ]
    
    result = processor.tokenize_and_align_labels(example_text, example_entities)
    
    # Декодируем обратно для проверки корректности разметки
    tokens = processor.tokenizer.convert_ids_to_tokens(result["input_ids"])
    for token, label_idx in zip(tokens, result["labels"]):
        if token == "[PAD]":
            break
        label_name = [k for k, v in processor.label_map.items() if v == label_idx]
        label_name = label_name[0] if label_name else "IGNORE"
        print(f"{token:15} | {label_name}")