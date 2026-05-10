import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

if Path("data/articles").exists():
    INPUT_DIR = Path("data/articles")
elif Path("articles").exists():
    INPUT_DIR = Path("articles")
else:
    print("папки нет")
    exit()

print(f"Найдена папка с данными")
OUTPUT_FILE = Path("dataset_restored.json")

def build_ner_dataset():
    dataset = []
    uri_pattern = re.compile(r'URI\[\[(.*?)\]\]/URI')
    
    xml_files = list(INPUT_DIR.glob("*.xml"))
    print(f"Файлы найдены")
    
    if len(xml_files) == 0:
        print("Нету xml")
        return

    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            uri_map = {}
            for f_main in root.findall(".//formulas_main/formula"):
                uri = f_main.get("uri")
                if uri: uri_map[uri] = {"text": f_main.text or "", "label": "FORMULA"}
                
            for f_aux in root.findall(".//formulas_aux/formula"):
                uri = f_aux.get("uri")
                if uri: uri_map[uri] = {"text": f_aux.text or "", "label": "FORMULA"}
                
            for rel in root.findall(".//relations/relation"):
                uri = rel.get("uri")
                rel_text = rel.findtext("rel_text", default="")
                if uri: uri_map[uri] = {"text": rel_text, "label": "TERM"}

            raw_text = root.findtext("text", default="")
            
            clean_text = ""
            entities = []
            current_pos = 0
            
            for match in uri_pattern.finditer(raw_text):
                uri = match.group(1)
                clean_text += raw_text[current_pos:match.start()]
                
                mapped = uri_map.get(uri)
                replacement = mapped["text"] if mapped else ""
                label = mapped["label"] if mapped else None
                
                start_entity = len(clean_text)
                clean_text += replacement
                end_entity = len(clean_text)
                
                if label:
                    entities.append({
                        "label": label,
                        "start": start_entity,
                        "end": end_entity
                    })
                    
                current_pos = match.end()
                
            clean_text += raw_text[current_pos:]
            
            dataset.append({
                "text": clean_text,
                "entities": entities
            })
            
        except Exception as e:
            print(f"Сбой в{file_path.name}: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    print(f"Создан файл {OUTPUT_FILE.name}")

if __name__ == "__main__":
    build_ner_dataset()