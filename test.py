import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset
import evaluate
import numpy as np

MODEL_PATH = "./math_ner_weighted"
DATA_PATH = "dataset.json"
MAX_LENGTH = 512
BATCH_SIZE = 8 

print(f"Инициализация валидатора. Модель: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model.to("cuda")

id2label = model.config.id2label
label2id = model.config.label2id

print("Загрузка данных")
raw_ds = Dataset.from_json(DATA_PATH)
ds_split = raw_ds.train_test_split(test_size=0.3, seed=42)
test_raw = ds_split["test"]

def tokenize_and_align(examples):

    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True
    )

    labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        doc_labels = [label2id["O"]] * len(offsets)
        entities = examples["entities"][i]

        for idx, (start, end) in enumerate(offsets):

            if start == end == 0:
                doc_labels[idx] = -100
                continue

            for ent in entities:
                if start >= ent['start'] and end <= ent['end']:
                    label = f"B-{ent['label']}" if start == ent['start'] else f"I-{ent['label']}"
                    doc_labels[idx] = label2id.get(label, 0)
                    break
        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Перезапуск токенизации с выравниванием")
test_dataset = test_raw.map(
    tokenize_and_align,
    batched=True,
    remove_columns=raw_ds.column_names,
    load_from_cache_file=False
)
test_dataset.set_format("torch")

print("Токенизация")
test_dataset = test_raw.map(tokenize_and_align, batched=True, remove_columns=raw_ds.column_names)
test_dataset.set_format("torch")

metric = evaluate.load("seqeval")
data_collator = DataCollatorForTokenClassification(tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

all_preds = []
all_labels = []

print("Прогон через нейросеть")
for batch in tqdm(test_loader):
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()
    labels = batch["labels"].cpu().numpy()

    for pred, label in zip(predictions, labels):
        true_p = [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        true_l = [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        all_preds.append(true_p)
        all_labels.append(true_l)

results = metric.compute(predictions=all_preds, references=all_labels)

print("\n" + "="*50)
print("Результаты")
print(f"Общая точность (Precision): {results['overall_precision']:.4f}")
print(f"Полнота (Recall):           {results['overall_recall']:.4f}")
print(f"F1-мера (F1-Score):        {results['overall_f1']:.4f}")
print(f"Accuracy:                  {results['overall_accuracy']:.4f}")
print("="*50)

for key, value in results.items():
    if isinstance(value, dict):
        print(f"Класс {key:10}: F1={value['f1']:.4f}, Precision={value['precision']:.4f}")