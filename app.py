import os
import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import RAGEngine


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "queries.jsonl")
INDEX_DIR = "index"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI RAG KUSK")


class AskRequest(BaseModel):
    question: str
    strict: Optional[bool] = False
    k: Optional[int] = 5


class QueryLogger:
    """Jednoduchý logger dotazů do `logs/queries.jsonl`."""

    def __init__(self, path: str = LOG_FILE):
        self.path = path

    def log(self, question: str, answer: str, sources: List[dict], confidence: float):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Inicializace RAG engine (lazy load index)
engine = RAGEngine()
try:
    engine.load_index(INDEX_DIR)
    logger.info("Index načten.")
except Exception:
    logger.warning("Index není dostupný při startu. Spusťte build_index.py pro vytvoření indexu.")


@app.post('/ask')
def ask(req: AskRequest):
    if engine.index is None:
        raise HTTPException(status_code=503, detail="Index není dostupný. Spusťte build_index.py")

    retrieved = engine.retrieve(req.question, k=req.k)
    result = engine.answer_question(req.question, retrieved, strict=req.strict)

    qlogger = QueryLogger()
    qlogger.log(req.question, result.get('answer', ''), result.get('sources', []), result.get('confidence', 0.0))

    return result
