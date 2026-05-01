import spacy
from spacy import displacy

BASE_MODEL_NAME = "ru_core_news_lg"
CUSTOM_MODEL_PATH = r"d:\Диплом\models\my_spancat_model"
SPANS_KEY = "sc"

def main():
    """
    Основной пайплайн гибридной модели: извлечение вложенных сущностей 
    и семантических связей на основе синтаксического дерева.
    """
    print(f"Инициализация базовой лингвистической модели ({BASE_MODEL_NAME})...")
    nlp = spacy.load(BASE_MODEL_NAME)

    print(f"Подключение компонента SpanCat из ({CUSTOM_MODEL_PATH})...")
    custom_nlp = spacy.load(CUSTOM_MODEL_PATH)
    
    # Интеграция обученного модуля в базовый пайплайн
    nlp.add_pipe("spancat", source=custom_nlp)
    print("Гибридная архитектура успешно сформирована.\n")

    # Входные данные для анализа
    text = "Серр доказал, что для алгебраическое многообразие определено дифференцирование."
    doc = nlp(text)

    print(f"Текст для анализа: {text}\n")

    # 1. извлечение вложенных именованных сущностей
    print("--- Извлеченные сущности ---")
    spans = doc.spans.get(SPANS_KEY, [])
    for span in spans:
        print(f"Категория: {span.label_:<10} | Текст: {span.text}")

    # 2. извлечение семантических отношений
    print("\n--- Извлеченные синтаксические отношения ---")
    for token in doc:
        # Поиск глагола, задающего отношение
        if token.pos_ == "VERB":
            subject = None
            obj = None
            
            # Обход дочерних узлов для поиска субъекта и объекта действия
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubj:pass"): 
                    subject = child
                elif child.dep_ in ("obj", "obl"): 
                    obj = child
                    
            # Формирование триплета, если найдены оба компонента
            if subject and obj:
                print(f"Триплет: [Субъект: {subject.text}] -> [Предикат: {token.lemma_}] -> [Объект: {obj.text}]")

    # 3. Визуализация и экспорт графов
    html_spans = displacy.render(doc, style="span", options={"spans_key": SPANS_KEY})
    with open("diploma_entities.html", "w", encoding="utf-8") as f:
        f.write(html_spans)

    html_dep = displacy.render(doc, style="dep", options={"compact": True, "distance": 120})
    with open("diploma_relations.html", "w", encoding="utf-8") as f:
        f.write(html_dep)

    print("\nHTML READY!")

if __name__ == "__main__":
    main()