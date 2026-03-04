import spacy
import random
import os
import logging
from spacy.training import Example
from spacy.util import minibatch, compounding

import config
from data_loader import load_label_studio_data

# Настройка простого логирования для консоли
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def make_examples(nlp_model, data_list):
    """Преобразует текстовые данные в формат Example, понятный для spaCy."""
    examples = []
    for text, ann in data_list:
        pred_doc = nlp_model.make_doc(text)
        ref_doc = nlp_model.make_doc(text)
        spans = []
        
        for start, end, label in ann.get("entities", []):
            span = ref_doc.char_span(start, end, label=label)
            if span is not None:
                spans.append(span)
        
        ref_doc.spans[config.SPANS_KEY] = spans
        examples.append(Example(pred_doc, ref_doc))
    return examples

def train_model():
    logger.info("--- Шаг 1: Загрузка обучающих данных ---")
    data = load_label_studio_data(config.ANNOTATIONS_FILE)
    
    if not data:
        logger.info("Критическая ошибка: Файл с данными пуст или не найден.")
        return

    # Перемешиваем и делим данные на обучение и проверку
    random.seed(config.SEED) 
    random.shuffle(data)
    
    split_idx = int(len(data) * config.TRAIN_SPLIT)
    train_data = data[:split_idx]
    dev_data = data[split_idx:]
    
    logger.info(f"Всего предложений: {len(data)}. На обучение: {len(train_data)}, на проверку: {len(dev_data)}")

    logger.info("\n--- Шаг 2: Подготовка базовой языковой модели ---")
    try:
        nlp = spacy.load("ru_core_news_lg")
    except OSError:
        logger.info("Ошибка: Не найдена базовая модель. Выполните в консоли: python -m spacy download ru_core_news_lg")
        return
    
    # Очищаем модель от лишних функций
    for pipe in list(nlp.pipe_names):
        nlp.remove_pipe(pipe)
    
    # Добавляем наш модуль для поиска вложенных терминов
    if "spancat" not in nlp.pipe_names:
        nlp.add_pipe("spancat", config={
            "threshold": 0.2,
            "spans_key": config.SPANS_KEY,
            "suggester": {
                "@misc": "spacy.ngram_suggester.v1", 
                "sizes": [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
            }
        })
    
    spancat = nlp.get_pipe("spancat")

    # Регистрируем категории терминов
    for _, annotations in train_data:
        for ent in annotations.get("entities", []):
            spancat.add_label(ent[2])
            
    logger.info(f"Модель настроена. Категории: {', '.join(spancat.labels)}")

    train_examples = make_examples(nlp, train_data)
    
    logger.info("\n--- Шаг 3: Начало обучения нейросети ---")
    optimizer = nlp.begin_training()
    
    for i in range(config.N_ITER):
        random.shuffle(train_examples)
        losses = {}
        
        # Разбиваем данные на пакеты
        batches = minibatch(train_examples, size=compounding(
            config.BATCH_START, config.BATCH_STOP, config.BATCH_COMPOUND
        ))
        
        for batch in batches:
            nlp.update(batch, drop=config.DROPOUT, losses=losses, sgd=optimizer)
            
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Эпоха {i+1:3} из {config.N_ITER} | Ошибка модели: {losses.get('spancat', 0):.4f}")

    logger.info("\n--- Шаг 4: Проверка знаний (Экзамен модели) ---")
    total_gold = 0
    total_pred = 0
    correct = 0

    eval_examples = make_examples(nlp, dev_data)
    
    for ex in eval_examples:
        pred_doc = nlp(ex.reference.text)
        
        gold_spans = set((s.start_char, s.end_char, s.label_) for s in ex.reference.spans.get(config.SPANS_KEY, []))
        pred_spans = set((s.start_char, s.end_char, s.label_) for s in pred_doc.spans.get(config.SPANS_KEY, []))
        
        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        correct += len(gold_spans.intersection(pred_spans))

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"Точность (не берет лишнее): {precision:.3f}")
    logger.info(f"Полнота (находит всё нужное): {recall:.3f}")
    logger.info(f"Общая оценка (F1-Score):      {f1:.3f}")

    logger.info("\n--- Шаг 5: Наглядная проверка ---")
    for ex in eval_examples[:3]:
        text = ex.reference.text
        pred_doc = nlp(text)
        
        gold = [(s.text, s.label_) for s in ex.reference.spans.get(config.SPANS_KEY, [])]
        pred = [(s.text, s.label_) for s in pred_doc.spans.get(config.SPANS_KEY, [])]
        
        logger.info(f"\nТекст: {text[:80]}...")
        logger.info(f"Правильные ответы: {gold}")
        logger.info(f"Ответы модели:     {pred}")

    if not os.path.exists(config.MODEL_OUTPUT_DIR):
        os.makedirs(config.MODEL_OUTPUT_DIR)
    nlp.to_disk(config.MODEL_OUTPUT_DIR)
    logger.info(f"\nУспех! Обученная модель сохранена в папку: {config.MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()