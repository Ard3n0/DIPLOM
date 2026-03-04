import json
import logging
import re
from typing import List, Tuple, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def clean_span(text: str, start: int, end: int) -> Tuple[int, int]:

    span_text = text[start:end]
    
    # Поиск первого значимого символа
    match_start = re.search(r'\w', span_text)
    if not match_start:
        return start, end
    
    new_start = start + match_start.start()
    
    # Поиск последнего значимого символа
    trimmed_from_start = span_text[match_start.start():]
    match_end = list(re.finditer(r'\w', trimmed_from_start))
    
    if not match_end:
        return new_start, end
    
    new_end = new_start + match_end[-1].end()
    
    return new_start, new_end

def load_label_studio_data(file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка ввода-вывода при чтении {file_path}: {e}")
        return []

    clean_data = []
    
    for item in raw_data:
        text = item.get("text") or item.get("data", {}).get("text")
        raw_entities = item.get("entities", [])
        
        if not raw_entities and "annotations" in item:
            for ann in item["annotations"]:
                for res in ann.get("result", []):
                    val = res.get("value", {})
                    if "start" in val and "end" in val:
                        raw_entities.append({
                            "start": val["start"], 
                            "end": val["end"], 
                            "label": val.get("labels", [""])[0]
                        })

        if text and raw_entities:
            final_entities = []
            
            for ent in raw_entities:
                # Техническая чистка границ интервала
                clean_s, clean_e = clean_span(text, ent["start"], ent["end"])
                
                # Валидация: начало должно быть строго меньше конца
                if clean_s < clean_e:
                    final_entities.append((clean_s, clean_e, ent["label"]))
            
            clean_data.append((text, {"entities": final_entities}))

    logger.info(f"Успешно загружено и нормализовано примеров: {len(clean_data)}")
    return clean_data