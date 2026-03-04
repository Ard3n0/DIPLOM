import spacy
import logging
from spacy.util import filter_spans
from typing import List

import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Тестовая выборка
TEST_SENTENCES: List[str] = [
    "Галуа использовал теорию групп.",
    "Атья доказал теорему об индексе для эллиптических операторов.",
    "Рассмотрим пучок, где дифференцирование не работает.",
    "Канторович предложил метод для банаховых пространств."
]

def main() -> None:
    """
    Загружает модель, выполняет инференс и применяет алгоритм 
    фильтрации для разрешения конфликтов перекрывающихся сущностей.
    """
    logger.info(f"Инициализация модели из {config.MODEL_OUTPUT_DIR}...")
    try:
        nlp = spacy.load(config.MODEL_OUTPUT_DIR)
    except OSError as e:
        logger.error(f"Ошибка загрузки весов модели: {e}. Запустите процесс обучения.")
        return

    logger.info("Модель успешно загружена. Запуск пакетного анализа.\n")

    for text in TEST_SENTENCES:
        doc = nlp(text)

        raw_spans = doc.spans.get(config.SPANS_KEY, [])
        
        # Постобработка: фильтрация дубликатов и пересечений
        filtered_spans = filter_spans(raw_spans)
        
        print(f"Текст: {text}")
        if not filtered_spans:
            print("  -> Сущности не обнаружены.\n")
        else:
            for span in filtered_spans:
                print(f"  🔹 {span.label_:<10} | {span.text}")
            print()

if __name__ == "__main__":
    main()