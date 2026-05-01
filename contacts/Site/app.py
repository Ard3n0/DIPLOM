import os
import json
import re
import datetime
import torch
import torch.nn.functional as F
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset

app = FastAPI()

DIPLOMA_PATH = "D:/Диплом"
BASE_MODEL_PATH = os.path.join(DIPLOMA_PATH, "saved_model")
HIDDEN_DATA_FILE = "autonomous_dataset.jsonl"

CONFIDENCE_THRESHOLD = 0.0
RETRAIN_LIMIT = 50

LABEL_LIST = ["O", "B-TERM", "I-TERM", "B-NAME", "I-NAME", "B-FORMULA", "I-FORMULA"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

def get_latest_model_path():
    if not os.path.exists(DIPLOMA_PATH): return BASE_MODEL_PATH
    folders = [os.path.join(DIPLOMA_PATH, d) for d in os.listdir(DIPLOMA_PATH) if "model_checkpoint_" in d]
    if folders:
        return max(folders, key=os.path.getmtime)
    return BASE_MODEL_PATH

CURRENT_MODEL_PATH = get_latest_model_path()

tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(
    CURRENT_MODEL_PATH, 
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True
)

def background_retrain_task():
    if not os.path.exists(HIDDEN_DATA_FILE): return
    
    tokens_list, tags_list = [], []
    with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    tokens_list.append(data["tokens"])
                    tags_list.append(data["ner_tags"])
                except: continue
                
    if not tokens_list: return

    train_ds = Dataset.from_dict({"tokens": tokens_list, "ner_tags": tags_list})
    
    def tokenize_and_align(examples):
        tok = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True, max_length=128)
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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    NEW_PATH = os.path.join(DIPLOMA_PATH, f"model_checkpoint_{ts}")

    args = TrainingArguments(
        output_dir=NEW_PATH,
        learning_rate=3e-5,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=tds, processing_class=tokenizer)
    trainer.train()
    
    model.save_pretrained(NEW_PATH, safe_serialization=False)
    tokenizer.save_pretrained(NEW_PATH)
    
    with open(HIDDEN_DATA_FILE, "w", encoding="utf-8") as f: f.truncate(0)

@app.get("/")
async def serve_frontend(): 
    return FileResponse("index.html")

@app.post("/extract")
async def analyze_text(bg_tasks: BackgroundTasks, text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if file: text = (await file.read()).decode("utf-8")
    if not text: return {"error": "Текст не предоставлен"}

    text = text.replace("<", "＜").replace(">", "＞")

    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = inputs.pop("offset_mapping")[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits[0], dim=1)
    preds = torch.argmax(probs, dim=1)
    
    raw_entities, current, min_p = [], None, 1.0
    
    for idx, (pred, prob_t, off) in enumerate(zip(preds, probs, offsets)):
        s, e = off.tolist()
        if s == 0 and e == 0: continue
        
        label = ID2LABEL[pred.item()]
        p = prob_t[pred.item()].item()
        
        if label != "O" and p < CONFIDENCE_THRESHOLD: label = "O"
            
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

    for ent in raw_entities:
        if ent["type"] == "NAME":
            pattern = r'[А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[А-ЯЁA-Z][а-яёa-z\-]+)+'
            for m in re.finditer(pattern, text):
                if m.start() <= ent["start"] <= m.end():
                    ent.update({"start": m.start(), "end": m.end(), "word": text[m.start():m.end()]})
                    break

    final_entities = []
    seen = set()
    for ent in [e for e in raw_entities if e["type"] != "FORMULA"]:
        if (ent["start"], ent["end"]) not in seen:
            seen.add((ent["start"], ent["end"]))
            final_entities.append(ent)

    math_pattern = r'[A-Za-z0-9][A-Za-z0-9\(\)\{\}\.\,]*[\=\+\-\^\/＜＞\*\<\>][A-Za-z0-9\(\)\{\}\.\,\=\+\-\^\/＜＞\*\<\>]+'
    for match in re.finditer(math_pattern, text):
        word, s, e = match.group(0), match.start(), match.end()
        while word and word[-1] in ".,;!?": 
            word, e = word[:-1], e - 1
        if re.search(r'[\=\^\>\<\+\*\/＜＞]', word):
            if not any(ent["start"] <= s < ent["end"] for ent in final_entities):
                final_entities.append({"word": word, "type": "FORMULA", "start": s, "end": e})

    if min_p >= CONFIDENCE_THRESHOLD:
        t_list, g_list = [], []
        for idx, off in enumerate(offsets):
            s_off, e_off = off.tolist()
            if s_off == 0 and e_off == 0: continue
            t_list.append(text[s_off:e_off])
            g_list.append(preds[idx].item())
            
        with open(HIDDEN_DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tokens": t_list, "ner_tags": g_list}, ensure_ascii=False) + "\n")
            f.flush()
            
        with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f:
            if sum(1 for _ in f if _.strip()) >= RETRAIN_LIMIT:
                bg_tasks.add_task(background_retrain_task)

    return {"status": "success", "text": text, "entities": final_entities}