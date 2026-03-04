import json
import re
import os

# Константы конфигурации
DATASET_PATH = "data/annotations.json"

def audit_dataset(file_path: str) -> None:
    """
    Проводит автоматизированный аудит обучающей выборки.
    Выявляет некорректно размеченные интервалы (содержащие только 
    знаки препинания, спецсимволы или пробелы).
    """
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Критическая ошибка чтения структуры JSON: {e}")
        return

    print(f"--- Инициализация аудита данных: {file_path} ---\n")
    found_garbage = False

    for idx, item in enumerate(data):
        text = item.get("text") or item.get("data", {}).get("text")
        entities = item.get("entities", [])
        
        if not entities and "annotations" in item:
            for ann in item["annotations"]:
                for res in ann.get("result", []):
                    val = res.get("value", {})
                    if "start" in val and "end" in val:
                        entities.append({
                            "start": val["start"], 
                            "end": val["end"], 
                            "label": val.get("labels", [""])[0]
                        })

        if not text:
            continue

        for ent in entities:
            start, end = ent["start"], ent["end"]
            span_text = text[start:end]
            
            if not re.search(r'[\wА-Яа-я]', span_text):
                found_garbage = True
                print(f"Обнаружена аномалия (шумовой интервал) в записи №{idx}:")
                print(f"  Фрагмент текста: \"{text[:100]}...\"")
                print(f"  Аномалия:        \"{span_text}\" (Класс: {ent.get('label')}, start: {start}, end: {end})")
                print("-" * 50)

    if not found_garbage:
        print("Аудит завершен успешно. Структурных аномалий не выявлено.")
    else:
        print("\nОШИБКА! Проверьте файл вручную")

if __name__ == "__main__":
    audit_dataset(DATASET_PATH)