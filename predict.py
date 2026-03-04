import spacy
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class EntityPredictor:
    """
    Класс для загрузки обученной NLP-модели и проведения инференса 
    (распознавания вложенных сущностей в произвольном тексте).
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.nlp = self._load_model()

    def _load_model(self) -> spacy.Language | None:
        try:
            logger.info(f"Загрузка весов модели из директории: {self.model_path}")
            return spacy.load(self.model_path)
        except OSError:
            logger.error(f"Модель не найдена по пути: {self.model_path}. Требуется запуск обучения.")
            return None

    def predict(self, text: str) -> None:
        if not self.nlp:
            logger.warning("Инференс невозможен: модель не загружена.")
            return

        logger.info(f"Анализ текста: '{text}'")
        doc = self.nlp(text)
        
        spans = doc.spans.get(config.SPANS_KEY, [])
        
        if not spans:
            print("  -> Сущности не обнаружены.")
        else:
            print("  -> Найденные термины:")
            for span in spans:
                print(f"     [{span.label_:<10}] | {span.text}")
        print("-" * 50)

def main():
    predictor = EntityPredictor(config.MODEL_OUTPUT_DIR)
    
    if predictor.nlp:
        # Тестирование на новых данных
        test_text_1 = "Теорема Пифагора утверждает, что квадрат гипотенузы равен сумме квадратов катетов."
        predictor.predict(test_text_1)
        
        # Тестирование на данных из предметной области
        test_text_2 = "Серр доказал, что для алгебраическое многообразие определено дифференцирование."
        predictor.predict(test_text_2)

if __name__ == "__main__":
    main()