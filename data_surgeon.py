import json
import re
import os
import shutil
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ANNOTATIONS_FILE: str = "data/annotations.json"
BACKUP_PATH: str = "data/generated_300_backup.json"

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """Рекурсивно разворачивает вложенные списки в плоскую структуру."""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def clean_dataset(file_path: str, backup_path: str) -> None:
    """
    Нормализует структуру датасета, фильтрует шумовые сущности 
    и сохраняет результат с предварительным бэкапом.
    """
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return

    # Резервное копирование
    try:
        shutil.copy(file_path, backup_path)
    except IOError as e:
        logger.error(f"Ошибка создания резервной копии: {e}")
        return

    # Загрузка данных
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON: {e}")
        return

    # Нормализация вложенности
    data = flatten_list(raw_data) if isinstance(raw_data, list) else [raw_data]
    logger.info(f"Начало очистки. Исходное количество объектов: {len(data)}")
    
    cleaned_data = []
    total_removed_ents = 0

    for item in data:
        text = item.get("text") or item.get("data", {}).get("text")
        if not text:
            continue

        entities = item.get("entities") or []

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

        initial_count = len(entities)
        valid_entities = []
        
        # Эвристическая фильтрация
        for ent in entities:
            span_text = text[ent['start']:ent['end']]
            if re.search(r'[\wА-Яа-я]', span_text):
                valid_entities.append(ent)
        
        total_removed_ents += (initial_count - len(valid_entities))

        if valid_entities:
            cleaned_data.append({
                "text": text,
                "entities": valid_entities
            })

    # Сохранение очищенного датасета
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    logger.info("Очистка завершена успешно.")
    logger.info(f"Удалено шумовых сущностей: {total_removed_ents}")
    logger.info(f"Итоговое количество чистых объектов: {len(cleaned_data)}")
    logger.info(f"Оригинал сохранен в {backup_path}")

if __name__ == "__main__":
    clean_dataset(ANNOTATIONS_FILE, BACKUP_PATH)