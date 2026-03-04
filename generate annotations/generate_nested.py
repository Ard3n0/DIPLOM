import json
import random
import re

# Словари терминов
PERSONS = [
    "Канторович", "Серр", "Галуа", "Гротендик", "Нётер", "Гильберт", "Риман", "Вейль", "Картан", "Пуанкаре",
    "Лебег", "Банах", "Хаусдорф", "Колмогоров", "Артин", "Борель", "Фреше", "Александров", "Урыссон",
    "Цорн", "Тихонов", "Стоун", "Чех", "Эйленберг", "Маклейн", "Ходж", "Лере", "Делинь", "Милнор",
    "Морс", "Уитни", "Атья", "Зингер", "Ли", "Кронекер", "Дедекинд", "Шварц", "Мальгранж", "Хирцебрух",
    "Серф", "Том", "Ланг", "Шафаревич", "Манин", "Арнольд", "Гельфанд", "Фаддеев", "Новиков", "Бернштейн",
    "Хан", "Вейерштрасс", "Рам", "Накаяма", "Шур", "Рисс", "Фубини", "Стинрод", "Кюннет", "Гаусс", "Бонне", "Стокс", "Бетти"
]
OBJECTS = ["векторное пространство", "топологическое пространство", "гладкое многообразие", "пучок", "когомологии", "цепной комплекс", "производная категория"]
OPERATIONS = ["дифференцирование", "интегрирование", "изоморфизм", "тензорное произведение", "пополнение", "проекция"]
ASSERTIONS = ["Лемма Цорна", "Теорема двойственности Серра", "Теорема Хана–Банаха", "Теорема Римана–Роха", "Теорема Стоуна–Вейерштрасса", "Теорема Тихонова"]

TEMPLATES = [
    "{assertion} устанавливает свойства, которыми обладает {object}.",
    "{person} доказал, что для {object} определено {operation}."
]

def main():
    generated_data = []

    # Генерируем 300 примеров
    for _ in range(300):
        text = random.choice(TEMPLATES).format(
            person=random.choice(PERSONS), 
            object=random.choice(OBJECTS),
            operation=random.choice(OPERATIONS), 
            assertion=random.choice(ASSERTIONS)
        )
        
        entities = []
        
        # Ищем точные совпадения базовых терминов
        for a in ASSERTIONS:
            for match in re.finditer(rf'\b{a}\b', text, flags=re.IGNORECASE):
                entities.append({"start": match.start(), "end": match.end(), "label": "ASSERTION"})
                
        for o in OBJECTS:
            for match in re.finditer(rf'\b{o}\b', text, flags=re.IGNORECASE):
                entities.append({"start": match.start(), "end": match.end(), "label": "OBJECT"})
                
        for op in OPERATIONS:
            for match in re.finditer(rf'\b{op}\b', text, flags=re.IGNORECASE):
                entities.append({"start": match.start(), "end": match.end(), "label": "OPERATION"})

        # Ищем фамилии с учетом русских падежей
        for p in PERSONS:
            for match in re.finditer(rf'\b{p}(?:а|у|е|ом|ой|и|ы)?\b', text, flags=re.IGNORECASE):
                entities.append({"start": match.start(), "end": match.end(), "label": "PERSON"})

        # Убираем дубликаты
        unique_entities = [dict(t) for t in {tuple(d.items()) for d in entities}]
        generated_data.append({"text": text, "entities": unique_entities})

    # Сохраняем результат
    with open("data/annotations.json", "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()