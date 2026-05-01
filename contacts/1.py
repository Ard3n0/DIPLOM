import json

scientists = ["Гаусс", "Эйлер", "Риман", "Коши", "Вейерштрасс", "Лобачевский", "Фурье"]
terms = ["интеграл", "дифференциал", "производная", "матрица", "вектор", "функция"]
formulas = ["E=mc^2", "a^2+b^2=c^2", "f(x)=y", "sin(x)", "df/dx", "lim(n->0)"]

dataset = []

for s in scientists:
    for t in terms:
        for f in formulas:
            if len(dataset) >= 300: break
            entry = {
                "tokens": ["Ученый", s, "изучал", t, "и", "использовал", f],
                "ner_tags": [0, 3, 0, 1, 0, 0, 5]
            }
            dataset.append(entry)

with open("autonomous_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")