import os
import re
import json
import glob
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XML_FOLDER_PATH = os.path.join(BASE_DIR, "data", "articles")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "data", "articles_result", "dataset.json")
# =======================

def process_xml_dataset():
    dataset = []
    uri_pattern = re.compile(r'URI\[\[(.*?)\]\]/URI')

    print(f"👀 Ищу файлы вот в этой папке:\n{XML_FOLDER_PATH}")
    
    search_path = os.path.join(XML_FOLDER_PATH, "*.xml")
    xml_files = glob.glob(search_path)
    
    if not xml_files:
        print("❌ Ошибка: В этой папке нет XML файлов! Проверь, правильно ли названа папка.")
        return

    print(f"✅ Найдено файлов: {len(xml_files)}. Начинаю обработку...")

    for filepath in xml_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            continue

        content = re.sub(r'<\?xml.*?\?>', '', content)
        xml_content = f"<root>{content}</root>"

        try:
            root = ET.fromstring(xml_content)
        except Exception as e:
            continue

        for article in root.findall('.//article'):
            uri_dict = {}
            
            for rel in article.findall('.//relations/relation'):
                uri = rel.get('uri')
                rel_text = rel.find('rel_text').text if rel.find('rel_text') is not None else ""
                uri_dict[uri] = (rel_text, "TERM")
                
            formulas = article.findall('.//formulas_main/formula') + article.findall('.//formulas_aux/formula')
            for form in formulas:
                uri = form.get('uri')
                form_text = form.text if form.text is not None else ""
                uri_dict[uri] = (form_text, "FORMULA")

            text_element = article.find('text')
            if text_element is None or not text_element.text:
                continue
                
            raw_text = text_element.text
            clean_text = ""
            entities = []
            last_pos = 0

            for match in uri_pattern.finditer(raw_text):
                clean_text += raw_text[last_pos:match.start()]
                
                uri = match.group(1)
                item_text, item_type = uri_dict.get(uri, ("", "UNKNOWN"))
                
                start_coord = len(clean_text)
                clean_text += item_text
                end_coord = len(clean_text)
                
                if item_type == "TERM" and item_text:
                    if any(char in item_text for char in ['$', '\\', '=', '^', '_', '{', '}']):
                        final_label = "FORMULA"
                    elif item_text[0].isupper() and item_text.lower() != item_text:
                        final_label = "NAME"
                    else:
                        final_label = "TERM"

                    entities.append({
                        "word": item_text,
                        "label": final_label,
                        "start": start_coord,
                        "end": end_coord
                    })
                
                last_pos = match.end()

            clean_text += raw_text[last_pos:]
            
            if entities:
                dataset.append({
                    "text": clean_text,
                    "entities": entities
                })
            
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Готово! Создан умный файл: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    process_xml_dataset()