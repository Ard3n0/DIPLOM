import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def check_hardware():
    print("--- Проверка аппаратного обеспечения ---")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Доступен GPU: {device_name}")
    else:
        print("GPU не обнаружен. Вычисления будут перенесены на CPU.")

def check_model_loading(model_name="cointegrated/rubert-tiny2"):
    print(f"\n--- Проверка загрузки модели: {model_name} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        print("Компоненты успешно инициализированы.")
        print(f"Количество токенов в словаре: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Ошибка при загрузке компонентов: {e}")

if __name__ == "__main__":
    check_hardware()
    check_model_loading()