import os
import re
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "saved_model"))

print(f"Попытка загрузки модели из: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    print("✅ Нейросеть успешно загружена и готова к работе!")
except Exception as e:
    print(f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")

id2label = {0: "O", 1: "B-TERM", 2: "I-TERM", 3: "B-FORMULA", 4: "I-FORMULA", 5: "B-NAME", 6: "I-NAME"}

def classify_sub_type(word, category):
    """Определяет точный подтип и фильтрует мусорные слова"""
    w = word.lower().strip(".,!?;:() ")
    
    trash = {
        "рассмотрим", "который", "которую", "часто", "называют", "устанавливает", 
        "связь", "между", "применяется", "используем", "выполняется", "стоит", 
        "упомянуть", "играют", "роль", "виде", "закона", "порядка", "проверке",
        "фундаментальную", "исследовании", "некоторой", "области"
    }
    
    if w in trash or len(w) < 3:
        return None

    if category == "NAME": return "УЧЕНЫЙ"
    if category == "FORMULA" or "$" in word: return "ФОРМУЛА"
    
    if any(x in w for x in ["теорем", "лемма", "аксиом", "следстви"]): return "ТЕОРЕМА"
    if any(x in w for x in ["метод", "алгоритм", "способ", "правило"]): return "МЕТОД"
    if any(x in w for x in ["пространств", "множеств", "поле", "групп"]): return "ОБЪЕКТ"
    if any(x in w for x in ["интеграл", "производн", "дифференц", "сумм", "операц"]): return "ОПЕРАЦИЯ"
    if any(x in w for x in ["аналитическ", "неравенство", "сходимост"]): return "СВОЙСТВО"
    
    return "ТЕРМИН"

def extract_entities(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    all_entities = []
    global_offset = 0

    for sentence in sentences:
        if not sentence.strip(): continue
        
        inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
        offsets = inputs.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        current_ent = None
        for idx, pred_id in enumerate(predictions):
            label_str = id2label.get(pred_id, "O")
            start, end = offsets[idx]
            if start == 0 and end == 0: continue 

            if label_str != "O":
                cat = label_str.split("-")[1]
                if current_ent and (cat == current_ent["category"]) and (start <= (current_ent["end"] - global_offset) + 1):
                    current_ent["end"] = global_offset + end
                else:
                    if current_ent: all_entities.append(current_ent)
                    current_ent = {"category": cat, "start": global_offset + start, "end": global_offset + end}
            else:
                if current_ent:
                    all_entities.append(current_ent)
                    current_ent = None
        
        if current_ent: all_entities.append(current_ent)
        global_offset += len(sentence) + 1

    final_res = []
    for ent in all_entities:
        s, e = ent["start"], ent["end"]
        while e < len(text) and not text[e].isspace() and text[e] not in ".,!?;:()":
            e += 1
            
        word = text[s:e].strip(".,!?;:() ")
        sub_type = classify_sub_type(word, ent["category"])
        
        if sub_type:
            final_res.append({"word": word, "type": sub_type, "start": s, "end": e})

    return {"text": text, "entities": final_res}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        return f"<h3>Ошибка: Файл index.html не найден по пути {index_path}</h3>"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/extract")
async def extract_api(text: str = Form(None), file: UploadFile = File(None)):
    target_text = ""
    if file and file.filename:
        content = await file.read()
        target_text = content.decode("utf-8")
    elif text:
        target_text = text
    
    if not target_text:
        return {"error": "Текст не предоставлен"}
        
    return extract_entities(target_text)