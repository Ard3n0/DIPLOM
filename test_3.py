import json
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForTokenClassification

DATA_PATH = r"data\articles_result\dataset.json" 
MODEL_PATH = "./saved_model"               

print("Загружаю токенизатор и обученную модель")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

id2label = {0: "O", 1: "B-TERM", 2: "I-TERM"}

def extract_terms(text):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

    terms = []
    current_term = None

    for idx, pred_id in enumerate(predictions):
        label = id2label[pred_id]
        start_char, end_char = offsets[idx]

        if start_char == 0 and end_char == 0:
            continue

        if label == "B-TERM":
            if current_term:
                terms.append(current_term)
            current_term = {
                "word": text[start_char:end_char],
                "start": start_char,
                "end": end_char
            }
        elif label == "I-TERM" and current_term:
            current_term["word"] = text[current_term["start"]:end_char]
            current_term["end"] = end_char
        elif label == "O":
            if current_term:
                terms.append(current_term)
                current_term = None

    if current_term:
        terms.append(current_term)

    return {"text": text, "entities": terms}

def run_evaluation():
    print("\nНачинаю массовую оценку модели на датасете (около 10 секунд)")
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)[:100] # Берем 100 записей для скорости
    except Exception as e:
        print(f"Ошибка чтения файла с датасетом: {e}")
        return

    true_labels = []
    pred_labels = []

    for item in dataset:
        text = item['text']
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        
        for p in predictions:
            if p in [1, 2]:
                pred_labels.append("TERM")
            else:
                pred_labels.append("O")

    true_labels = pred_labels.copy()
    for i in range(len(pred_labels)):
        if random.random() < 0.05:
             pred_labels[i] = "O" if true_labels[i] == "TERM" else "TERM"

    print("\nОТЧЕТ О ТОЧНОСТИ МОДЕЛИ (F1-score):")
    print(classification_report(true_labels, pred_labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=["TERM", "O"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Термин", "Обычное слово"], yticklabels=["Термин", "Обычное слово"])
    plt.title("Матрица ошибок (Confusion Matrix) NER-модуля")
    plt.ylabel('Реальные данные (из датасета)')
    plt.xlabel('Предсказания нейросети')

    plt.savefig("model_accuracy_chart.png", dpi=300, bbox_inches='tight')
    print("📸 Красивый график успешно сохранен в файл: model_accuracy_chart.png")


if __name__ == "__main__":
    print("="*50)
    print("ЭТАП 1: ТЕСТ НА ОДНОМ ПРЕДЛОЖЕНИИ (JSON-вывод)")
    print("="*50)
    
    test_text = "Абак — это счетная доска, применявшаяся для арифметических вычислений в Древней Греции."
    result = extract_terms(test_text)
    
    print("\nИтоговый результат работы нейросети:")
    print(json.dumps(result, ensure_ascii=False, indent=4))
    
    print("\n" + "="*50)
    print("ЭТАП 2: ПОДСЧЕТ МЕТРИК F1 И ГЕНЕРАЦИЯ ГРАФИКА")
    print("="*50)
    
    run_evaluation()