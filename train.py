# для быстрого обучения советуется использовать Google Colab


import json
import torch
from torch import nn
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed
)


MODEL_NAME = "cointegrated/rubert-tiny2"
DATA_PATH = "dataset.json"              
OUTPUT_DIR = "./math_ner_weighted"       
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 4                              
LEARNING_RATE = 2e-5

set_seed(42)


class WeightedNERTrainer(Trainer):

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight_tensor = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)

        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss = loss_fct(active_logits, active_labels)
        return (loss, outputs) if return_outputs else loss

def extract_labels(data):
    unique_labels = {"O"}
    for item in data:
        for ent in item["entities"]:
            unique_labels.add(f"B-{ent['label']}")
            unique_labels.add(f"I-{ent['label']}")
    return sorted(list(unique_labels))

def main():
    print("Чтение данных")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    label_list = extract_labels(raw_data)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    print(f"Найдено классов: {len(label_list)}. {label_list}")

    print(f"Загрузка модели {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    raw_ds = Dataset.from_list(raw_data)
    ds_split = raw_ds.train_test_split(test_size=0.3, seed=42)

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

    print("Компиляция графа данных (выравнивание и маскирование)")
    tokenized_ds = ds_split.map(
        tokenize_and_align,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Токенизация"
    )

    weights = torch.ones(len(label_list))

    if "O" in label2id:
        weights[label2id["O"]] = 0.2

    if "B-NAME" in label2id: weights[label2id["B-NAME"]] = 15.0
    if "I-NAME" in label2id: weights[label2id["I-NAME"]] = 15.0

    if "B-THEOREM" in label2id: weights[label2id["B-THEOREM"]] = 3.0
    if "I-THEOREM" in label2id: weights[label2id["I-THEOREM"]] = 3.0

    print(f"Инженерные веса лосса сконфигурированы")


    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none" 
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = WeightedNERTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\nЗапуск обучения.")
    trainer.train()

    print(f"\n[Обучение завершено. Сохранение в {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Готово!")

if __name__ == "__main__":
    main()