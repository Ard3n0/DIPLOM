from fastapi import FastAPI, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import FileResponse
from typing import Optional
import json
import os
import torch
import torch.nn.functional as F
import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset

app = FastAPI()

DIPLOMA_PATH = "D:/Диплом"
BASE_MODEL_PATH = os.path.join(DIPLOMA_PATH, "saved_model")
HIDDEN_DATA_FILE = "autonomous_dataset.jsonl"

def get_latest_model_path():
    if not os.path.exists(DIPLOMA_PATH): return BASE_MODEL_PATH
    all_folders = [os.path.join(DIPLOMA_PATH, d) for d in os.listdir(DIPLOMA_PATH) 
                   if os.path.isdir(os.path.join(DIPLOMA_PATH, d))]
    checkpoints = [d for d in all_folders if "model_checkpoint_" in d]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else BASE_MODEL_PATH

CURRENT_MODEL_PATH = get_latest_model_path()
CONFIDENCE_THRESHOLD = 0.85 # Порог для записи в датасет
RETRAIN_LIMIT = 100

print(f">>> Загрузка модели из: {CURRENT_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(CURRENT_MODEL_PATH)

def background_retrain_task():
    print(">>> [ОБУЧЕНИЕ] Запуск...")
    if not os.path.exists(HIDDEN_DATA_FILE) or os.stat(HIDDEN_DATA_FILE).st_size == 0: return
    tokens_list, tags_list = [], []
    with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                tokens_list.append(data["tokens"])
                tags_list.append(data["ner_tags"])
    
    train_ds = Dataset.from_dict({"tokens": tokens_list, "ner_tags": tags_list})
    def tokenize_and_align(ex):
        tok = tokenizer(ex["tokens"], truncation=True, padding="max_length", is_split_into_words=True, max_length=128)
        labels = []
        for i, label in enumerate(ex["ner_tags"]):
            word_ids = tok.word_ids(batch_index=i)
            previous_idx, label_ids = None, []
            for w_idx in word_ids:
                if w_idx is None: label_ids.append(-100)
                elif w_idx != previous_idx: label_ids.append(label[w_idx])
                else: label_ids.append(-100)
                previous_idx = w_idx
            labels.append(label_ids)
        tok["labels"] = labels
        return tok

    tds = train_ds.map(tokenize_and_align, batched=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    NEW_PATH = os.path.join(DIPLOMA_PATH, f"model_checkpoint_{ts}")
    
    args = TrainingArguments(output_dir=NEW_PATH, num_train_epochs=3, per_device_train_batch_size=2, save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=tds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(NEW_PATH)
    tokenizer.save_pretrained(NEW_PATH)
    with open(HIDDEN_DATA_FILE, "w", encoding="utf-8") as f: f.truncate(0)
    print(f">>> Обучение завершено. Новая модель: {NEW_PATH}")

@app.get("/")
async def serve_frontend(): return FileResponse("index.html")

@app.post("/extract")
async def analyze_text(bg_tasks: BackgroundTasks, text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if file: text = (await file.read()).decode("utf-8")
    if not text: return {"error": "Нет текста"}
    
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = inputs.pop("offset_mapping")[0]
    with torch.no_grad(): outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits[0], dim=1)
    preds = torch.argmax(probs, dim=1)
    id2label = model.config.id2label
    
    final_entities, current, min_conf = [], None, 1.0
    for idx, (pred, prob_t, off) in enumerate(zip(preds, probs, offsets)):
        s, e = off.tolist()
        if s == 0 and e == 0: continue
        label = id2label[pred.item()]
        p = prob_t[pred.item()].item()
        
        if p < 0.95 and label != "O": continue
        min_conf = min(min_conf, p)
        
        if label == "O":
            if current: final_entities.append(current)
            current = None
        else:
            t = label.split("-")[-1]
            if label.startswith("B-") or not current or current["type"] != t:
                if current: final_entities.append(current)
                current = {"word": text[s:e], "type": t, "start": s, "end": e}
            else:
                current["word"] = text[current["start"]:e]; current["end"] = e
    if current: final_entities.append(current)

    if min_conf >= CONFIDENCE_THRESHOLD:
        t_l, g_l = [], []
        for idx, off in enumerate(offsets):
            s, e = off.tolist()
            if s == 0 and e == 0: continue
            t_l.append(text[s:e]); g_l.append(preds[idx].item())
        with open(HIDDEN_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tokens": t_l, "ner_tags": g_l}, ensure_ascii=False) + "\n")
            f.flush()
        with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f:
            if sum(1 for _ in f if _.strip()) >= RETRAIN_LIMIT: bg_tasks.add_task(background_retrain_task)

    return {"status": "success", "text": text, "entities": final_entities}