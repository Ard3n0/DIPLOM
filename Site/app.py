import json
import re
import datetime
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, BackgroundTasks, Form, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

MODEL_DIR = BASE_DIR / "math_ner_weighted" 
HIDDEN_DATA_FILE = BASE_DIR / "autonomous_dataset.jsonl"
FRONTEND_FILE = APP_DIR / "index.html"

CONFIDENCE_THRESHOLD = 0.50
RETRAIN_LIMIT = 50
MAX_FILE_SIZE_MB = 2 

TRUE_LABELS = [
    'B-FORMULA', 'B-NAME', 'B-OPERATOR', 'B-TERM', 'B-THEOREM', 'B-VAR', 
    'I-FORMULA', 'I-NAME', 'I-OPERATOR', 'I-TERM', 'I-THEOREM', 'I-VAR', 'O'
]

def heal_corrupted_config(model_path: Path):
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Директория не найдена: {model_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if config_data.get("num_labels") != len(TRUE_LABELS):
        print("Повреждение конфигурации. Восстановление")
        config_data["num_labels"] = len(TRUE_LABELS)
        config_data["id2label"] = {str(i): label for i, label in enumerate(TRUE_LABELS)}
        config_data["label2id"] = {label: i for i, label in enumerate(TRUE_LABELS)}
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print("Конфигурация успешно сделана")

heal_corrupted_config(MODEL_DIR)

print(f"Монтирование весов из: {MODEL_DIR}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)

ID2LABEL = model.config.id2label
print(f"Вычислительное ядро активно. Классов: {len(ID2LABEL)}. Устройство: {device}")

app = FastAPI(title="Math NER API", version="5.0.0 (Release Candidate 2)")
training_lock = threading.Lock()
def background_retrain_task():
    if not training_lock.acquire(blocking=False):
        return

    try:
        if not HIDDEN_DATA_FILE.exists():
            return
        
        tokens_list, tags_list = [], []
        with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        tokens_list.append(data["tokens"])
                        tags_list.append(data["ner_tags"])
                    except json.JSONDecodeError:
                        continue
                
        if len(tokens_list) < RETRAIN_LIMIT:
            return

        print(f"Фоновое дообучение стартовало")
        train_ds = Dataset.from_dict({"tokens": tokens_list, "ner_tags": tags_list})
        
        def tokenize_and_align(examples):
            tok = tokenizer(
                examples["tokens"], truncation=True, padding="max_length", 
                is_split_into_words=True, max_length=128
            )
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tok.word_ids(batch_index=i)
                prev_idx, label_ids = None, []
                for w_idx in word_ids:
                    if w_idx is None: label_ids.append(-100)
                    elif w_idx != prev_idx: label_ids.append(label[w_idx])
                    else: label_ids.append(-100)
                    prev_idx = w_idx
                labels.append(label_ids)
            tok["labels"] = labels
            return tok

        tds = train_ds.map(tokenize_and_align, batched=True)

        args = TrainingArguments(
            output_dir=str(MODEL_DIR),
            learning_rate=2e-5,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_strategy="no",
            report_to="none"
        )
        
        trainer = Trainer(model=model, args=args, train_dataset=tds, tokenizer=tokenizer)
        trainer.train()
        
        model.save_pretrained(MODEL_DIR, safe_serialization=False)
        tokenizer.save_pretrained(MODEL_DIR)
        
        with open(HIDDEN_DATA_FILE, "w", encoding="utf-8") as f: 
            f.truncate(0)
        print("Дообучение завершено. Память сброшена.")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        training_lock.release()

@app.get("/")
async def serve_frontend(): 
    if not FRONTEND_FILE.exists():
        return {"error": "Фронтенд не найден."}
    return FileResponse(FRONTEND_FILE)

@app.post("/extract")
async def analyze_text(bg_tasks: BackgroundTasks, text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if file:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Файл слишком велик.")
        text = file_bytes.decode("utf-8")
        
    if not text: 
        return {"error": "Пустой запрос"}

    text = text.replace("<", "＜").replace(">", "＞")

    priority_entities = []
    math_pattern = r'[A-Za-z0-9][A-Za-z0-9\(\)\{\}\.\,\s]*[\=\+\-\^\/＜＞\*\<\>][A-Za-z0-9\(\)\{\}\.\,\=\+\-\^\/＜＞\*\<\>\s]+'
    
    for match in re.finditer(math_pattern, text):
        word, s, e = match.group(0), match.start(), match.end()
        
        original_len = len(word)
        word = word.strip()
        s += original_len - len(word.lstrip())
        e = s + len(word)
        
        while word and word[-1] in ".,;!?": 
            word, e = word[:-1], e - 1
            
        if re.search(r'[\=\^\>\<\+\*\/＜＞]', word):
            priority_entities.append({"word": word, "type": "FORMULA", "start": s, "end": e})

    name_pattern = r'\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?(?:\s+[А-ЯЁ][а-яё]+)*\b'
    stop_words = {"Пусть", "Согласно", "В", "Вернемся", "Посмотрим", "Здесь", "Рассмотрим", "Для", "Если", "То", "Таким", "Образом"}
    
    for match in re.finditer(name_pattern, text):
        word, s, e = match.group(0), match.start(), match.end()
        
        if word not in stop_words:
            is_sentence_start = (s == 0 or text[max(0, s-2):s] in {". ", "? ", "! "})
            is_multi_word = " " in word or "-" in word
            
            
            if not is_sentence_start or is_multi_word:
                is_swallowed = any(f["start"] <= s and e <= f["end"] for f in priority_entities)
                if not is_swallowed:
                    priority_entities.append({"word": word, "type": "NAME", "start": s, "end": e})

    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = inputs.pop("offset_mapping")[0]
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0].cpu()
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    raw_entities, current, min_p = [], None, 1.0
    
    DYNAMIC_THRESHOLDS = {
        "NAME": 0.35,      
        "THEOREM": 0.40,   
        "DEFAULT": CONFIDENCE_THRESHOLD 
    }
    
    for idx, (pred, prob_t, off) in enumerate(zip(preds, probs, offsets)):
        s, e = off.tolist()
        if s == 0 and e == 0: continue
        
        label_id = pred.item()
        label = ID2LABEL.get(label_id) or ID2LABEL.get(str(label_id), "O")
        p = prob_t[label_id].item()
        
        base_type = label.split("-")[-1] if label != "O" else "O"
        current_threshold = DYNAMIC_THRESHOLDS.get(base_type, DYNAMIC_THRESHOLDS["DEFAULT"])
        
        if label != "O" and p < current_threshold: 
            label = "O"
            
        if label == "O":
            if current: raw_entities.append(current)
            current = None
        else:
            min_p = min(min_p, p)
            t = label.split("-")[-1]
            if current and current["type"] == t and (s - current["end"] <= 1):
                current["end"] = e
                current["word"] = text[current["start"]:e]
            else:
                if current: raw_entities.append(current)
                current = {"word": text[s:e], "type": t, "start": s, "end": e}
                
    if current: raw_entities.append(current)

    for ent in raw_entities:
        start, end = ent["start"], ent["end"]
        while start > 0 and (text[start-1].isalnum() or text[start-1] in "-"): start -= 1
        while end < len(text) and (text[end].isalnum() or text[end] in "-"): end += 1
        ent.update({"start": start, "end": end, "word": text[start:end]})

    final_entities = list(priority_entities)
    seen = set((f["start"], f["end"]) for f in priority_entities)

    for ent in raw_entities:
        if ent["type"] in ["FORMULA", "NAME"]:
            continue 
        is_swallowed = any(f["start"] <= ent["start"] and ent["end"] <= f["end"] for f in priority_entities)
        
        if is_swallowed and ent["type"] in ["VAR", "OPERATOR"]:
            continue 

        if (ent["start"], ent["end"]) not in seen:
            seen.add((ent["start"], ent["end"]))
            final_entities.append(ent)

    if min_p >= CONFIDENCE_THRESHOLD:
        t_list, g_list = [], []
        for idx, off in enumerate(offsets):
            s_off, e_off = off.tolist()
            if s_off == 0 and e_off == 0: continue
            t_list.append(text[s_off:e_off])
            g_list.append(preds[idx].item())
            
        with open(HIDDEN_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tokens": t_list, "ner_tags": g_list}, ensure_ascii=False) + "\n")
            
        if HIDDEN_DATA_FILE.exists():
            with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f:
                lines = sum(1 for line in f if line.strip())
            if lines >= RETRAIN_LIMIT and not training_lock.locked():
                bg_tasks.add_task(background_retrain_task)

    final_entities.sort(key=lambda x: x["start"])

    return {"status": "success", "text": text, "entities": final_entities}