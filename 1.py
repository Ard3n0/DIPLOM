import json

# Списки для генерации
scientists = ["Гаусс", "Эйлер", "Риман", "Коши", "Лобачевский", "Пифагор", "Лаплас"]
concepts = ["интеграл", "дифференциал", "вектор", "матрица", "ряд", "определитель"]
math_expr = ["E=mc^2", "a^2+b^2=c^2", "f(x)=y", "sin(x)", "lim(n->inf)", "P=NP"]

data_rows = []

# Шаблон 1: Имя + Термин + Формула
for s in scientists:
    for c in concepts:
        for f in math_expr:
            entry = {
                "tokens": ["Ученый", s, "изучал", c, "и", "вывел", f],
                "ner_tags": [0, 3, 0, 1, 0, 0, 5] # 3=B-NAME, 1=B-TERM, 5=B-FORMULA
            }
            data_rows.append(entry)

# Записываем в файл
with open("autonomous_dataset.jsonl", "w", encoding="utf-8") as f:
    for row in data_rows[:300]: # Возьмем первые 300 для баланса
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Синтетический датасет готов. Теперь модель знает, кто такие Гаусс и что такое sin(x).")