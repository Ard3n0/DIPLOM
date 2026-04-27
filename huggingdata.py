import json
import os

# Пути к файлам
INPUT_FILE = "raw_dataset.jsonl"  # Твой скачанный файл
OUTPUT_FILE = "Site/autonomous_dataset.jsonl" # Файл, который "подхватит" сервер

# Настройка маппинга: что во что превращаем
# Слева - то что в датасете, справа - твоя метка
MAPPING = {
    "PERSON": "NAME",
    "PRODUCT": "NAME",
    "WORK_OF_ART": "NAME",
    "LAW": "NAME",
    "DISEASE": "TERM",
    "SCIENCE": "TERM",
    "EVENT": "TERM",
    "AWARD": "TERM"
}

def process_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Ошибка: Файл {INPUT_FILE} не найден!")
        return

    cleaned_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as f_out: # Добавляем в конец нашего файла
        
        for line in f_in:
            try:
                data = json.loads(line)
                tokens = data.get("tokens", [])
                ner_tags = data.get("ner_tags", [])
                
                # Если в датасете метки текстовые (напр. "B-PERSON"), 
                # нам нужно превратить их в ID твоей модели
                # Для этого используем твой label2id из модели
                
                new_tags = []
                has_useful_info = False

                for tag in ner_tags:
                    # Убираем B- или I- для проверки в маппинге
                    base_tag = tag.replace("B-", "").replace("I-", "")
                    
                    if base_tag in MAPPING:
                        # Если нашли совпадение, меняем метку
                        prefix = "B-" if "B-" in tag else "I-"
                        new_label = prefix + MAPPING[base_tag]
                        # ТУТ ВАЖНО: превращаем текст в ID (цифру)
                        # Замени цифры ниже на реальные ID из твоего model.config.label2id
                        if "NAME" in new_label:
                            new_tags.append(1) # Например, 1 - это NAME
                        else:
                            new_tags.append(2) # Например, 2 - это TERM
                        has_useful_info = True
                    else:
                        # Всех городов, организаций и т.д. помечаем как "O" (ноль)
                        new_tags.append(0)

                # Сохраняем только если в предложении нашлось что-то полезное
                if has_useful_info:
                    clean_entry = {
                        "tokens": tokens,
                        "ner_tags": new_tags
                    }
                    f_out.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")
                    cleaned_count += 1
            except Exception as e:
                print(f"Пропущена строка из-за ошибки: {e}")
                continue

    print(f"Готово! Обработано и добавлено в обучение {cleaned_count} предложений.")

if __name__ == "__main__":
    process_dataset()