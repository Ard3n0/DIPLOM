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

CONFIDENCE_THRESHOLD = 0.25 
RETRAIN_LIMIT = 50
MAX_FILE_SIZE_MB = 2 

TRUE_LABELS = [
    'B-FORMULA', 'B-NAME', 'B-OPERATOR', 'B-TERM', 'B-THEOREM', 'B-VAR', 
    'I-FORMULA', 'I-NAME', 'I-OPERATOR', 'I-TERM', 'I-THEOREM', 'I-VAR', 'O'
]

def heal_corrupted_config(model_path: Path):
    config_file = model_path / "config.json"
    if not config_file.exists(): return
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config_data["num_labels"] = len(TRUE_LABELS)
    config_data["id2label"] = {str(i): label for i, label in enumerate(TRUE_LABELS)}
    config_data["label2id"] = {label: i for i, label in enumerate(TRUE_LABELS)}
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

heal_corrupted_config(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
ID2LABEL = model.config.id2label

app = FastAPI(title="Math NER API", version="5.0.0")

@app.get("/")
async def serve_frontend(): 
    if not FRONTEND_FILE.exists(): return {"error": "Фронтенд не найден."}
    return FileResponse(FRONTEND_FILE)

@app.post("/extract")
async def analyze_text(bg_tasks: BackgroundTasks, text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if file:
        file_bytes = await file.read()
        text = file_bytes.decode("utf-8")
    if not text: return {"error": "Пустой запрос"}

    text = text.replace("<", "＜").replace(">", "＞")

    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = inputs.pop("offset_mapping")[0]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0].cpu()
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    raw_entities, current = [], None
    
    for idx, (pred, prob_t, off) in enumerate(zip(preds, probs, offsets)):
        s, e = off.tolist()
        if s == 0 and e == 0: continue
        
        label_id = pred.item()
        label = ID2LABEL.get(label_id) or ID2LABEL.get(str(label_id), "O")
        p = prob_t[label_id].item()
        
        if label != "O" and p < CONFIDENCE_THRESHOLD: 
            label = "O"
            
        if label == "O":
            if current: raw_entities.append(current)
            current = None
        else:
            t = label.split("-")[-1].upper().strip()
            if current and current["type"] == t and (s - current["end"] <= 1):
                current["end"] = e
                current["word"] = text[current["start"]:e]
            else:
                if current: raw_entities.append(current)
                current = {"word": text[s:e], "type": t, "start": s, "end": e}
                
    if current: raw_entities.append(current)

    for ent in raw_entities:
        w = ent["word"]
        while w and w[0] in " \t\n.,;:!?$":
            w = w[1:]
            ent["start"] += 1
        while w and w[-1] in " \t\n.,;:!?$":
            w = w[:-1]
            ent["end"] -= 1
        ent["word"] = text[ent["start"]:ent["end"]].strip()

    final_entities = [e for e in raw_entities if e["word"]]
    final_entities.sort(key=lambda x: x["start"])

    return {"status": "success", "text": text, "entities": final_entities}