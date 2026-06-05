import json, asyncio, threading
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODEL_DIR = BASE_DIR / "models" 
HIDDEN_DATA_FILE = BASE_DIR / "autonomous_dataset.jsonl"
FRONTEND_FILE = APP_DIR / "index.html"
CONFIDENCE_THRESHOLD, RETRAIN_LIMIT, MAX_FILE_SIZE_MB = 0.5, 50, 2 
TRUE_LABELS = ['B-FORMULA', 'B-NAME', 'B-OPERATOR', 'B-TERM', 'B-THEOREM', 'B-VAR', 'I-FORMULA', 'I-NAME', 'I-OPERATOR', 'I-TERM', 'I-THEOREM', 'I-VAR', 'O']

def heal_corrupted_config(model_path: Path):
    cf = model_path / "config.json"
    if not cf.exists(): return
    with open(cf, "r", encoding="utf-8") as f: c = json.load(f)
    c["num_labels"] = len(TRUE_LABELS)
    c.update({"id2label": {str(i): l for i, l in enumerate(TRUE_LABELS)}, "label2id": {l: i for i, l in enumerate(TRUE_LABELS)}})
    with open(cf, "w", encoding="utf-8") as f: json.dump(c, f, indent=2, ensure_ascii=False)

heal_corrupted_config(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
ID2LABEL = model.config.id2label

app = FastAPI(title="Math NER API", version="5.0.0")
d_lock, t_lock = threading.Lock(), threading.Lock()

def tr_task():
    if not t_lock.acquire(blocking=False): return
    try:
        with d_lock:
            if not HIDDEN_DATA_FILE.exists(): return
            with open(HIDDEN_DATA_FILE, "r", encoding="utf-8") as f: lines = f.readlines()
            if len(lines) < 200: return
            open(HIDDEN_DATA_FILE, "w").close()
        Trainer(model=model, args=TrainingArguments(output_dir=str(MODEL_DIR)), train_dataset=Dataset.from_dict({"text": lines})).train()
    except Exception: pass
    finally: t_lock.release()

def sv_task(t, e):
    if not e: return
    with d_lock:
        with open(HIDDEN_DATA_FILE, "a", encoding="utf-8") as f: f.write(json.dumps({"text": t, "entities": e}, ensure_ascii=False)+"\n")

def chk_sz():
    if not HIDDEN_DATA_FILE.exists(): return 0
    with d_lock: return sum(1 for _ in open(HIDDEN_DATA_FILE, "r", encoding="utf-8"))

def inf(text):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = inputs.pop("offset_mapping")[0]
    with torch.no_grad(): probs = F.softmax(model(**{k: v.to(device) for k, v in inputs.items()}).logits[0].cpu(), dim=1)
    preds = torch.argmax(probs, dim=1)
    r, c = [], None
    for (pred, prob_t, off) in zip(preds, probs, offsets):
        s, e = off.tolist()
        if s == 0 and e == 0: continue
        lbl = ID2LABEL.get(pred.item(), "O")
        if lbl != "O" and prob_t[pred.item()].item() < CONFIDENCE_THRESHOLD: lbl = "O"
        if lbl == "O":
            if c: r.append(c)
            c = None
        else:
            t = lbl.split("-")[-1].upper().strip()
            if c and c["type"] == t and (s - c["end"] <= 1): c["end"], c["word"] = e, text[c["start"]:e]
            else:
                if c: r.append(c)
                c = {"word": text[s:e], "type": t, "start": s, "end": e}
    if c: r.append(c)
    for ent in r:
        w = ent["word"]
        while w and w[0] in " \t\n.,;:!?$": w, ent["start"] = w[1:], ent["start"] + 1
        while w and w[-1] in " \t\n.,;:!?$": w, ent["end"] = w[:-1], ent["end"] - 1
        ent["word"] = text[ent["start"]:ent["end"]].strip()
    return [e for e in r if e["word"]]

@app.get("/")
async def serve_frontend(): 
    return FileResponse(FRONTEND_FILE) if FRONTEND_FILE.exists() else {"error": "Фронтенд не найден."}

@app.post("/extract")
async def analyze_text(bg_tasks: BackgroundTasks, text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if file: text = (await file.read()).decode("utf-8")
    if not text: return {"error": "Пустой запрос"}
    text = text.replace("<", "＜").replace(">", "＞")
    e, sz = await asyncio.gather(asyncio.to_thread(inf, text), asyncio.to_thread(chk_sz))
    if sz >= 200: bg_tasks.add_task(tr_task)
    else: bg_tasks.add_task(sv_task, text, e)
    return {"status": "success", "text": text, "entities": e}