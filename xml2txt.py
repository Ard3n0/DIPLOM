import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_DIR = Path("./articles")
OUTPUT_DIR = Path("./batches_v1.2")
BATCH_SIZE = 100

def smart_defragment(text):
    if not text: return ""

    formulas = re.findall(r'\$.*?\$', text)
    for i, f in enumerate(formulas):
        text = text.replace(f, f"[[FORMULA_{i}]]")

    def sub_func(match):
        return match.group(0).replace(" ", "")

    text = re.sub(r'(?:(?<=\s)|(?<=^))(\w\s){2,}\w', sub_func, text)

    for i, f in enumerate(formulas):
        text = text.replace(f"[[FORMULA_{i}]]", f)

    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def process_corpus():
    if not SOURCE_DIR.exists():
        print(f"Ошибка: Папка {SOURCE_DIR} не найдена.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    xml_files = sorted(list(SOURCE_DIR.glob("*.xml")))
    
    print(f"Обработка {len(xml_files)} файлов")

    current_batch = []
    batch_idx = 1

    for i, file_path in enumerate(xml_files, 1):
        try:
            with open(file_path, 'rb') as f:
                raw_xml = f.read()
            
            root = ET.fromstring(raw_xml)
            text_orig = root.findtext('text_orig', default='')
            title = root.findtext('title', default='Untitled')

            if text_orig:
                cleaned = smart_defragment(text_orig)
                entry = f"--- ID: {file_path.stem} | TITLE: {title} ---\n{cleaned}\n"
                current_batch.append(entry)

            if i % BATCH_SIZE == 0 or i == len(xml_files):
                out_path = OUTPUT_DIR / f"batch_{batch_idx:03d}.txt"
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(current_batch))
                print(f"БАтч {batch_idx:03d} готов")
                current_batch = []
                batch_idx += 1

        except Exception as e:
            print(f"Сбой в файле {file_path.name}: {e}")

if __name__ == "__main__":
    process_corpus()
    print("Готово")