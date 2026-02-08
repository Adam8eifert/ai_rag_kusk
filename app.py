"""FastAPI aplikace pro RAG: HTTP vrstva na RAGEngine.

Poskytuje POST /ask endpoint pro zodpovídání otázek z PDF dokumentů.
Logging všech dotazů do logs/queries.jsonl.
"""

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

app = FastAPI(
    title="AI RAG KUSK",
    description="Retrieval-Augmented Generation systém pro českou dokumentaci"
)


class AskRequest(BaseModel):
    """Schéma požadavku na otázku.
    
    Parametry:
    - question: Otázka (povinné)
    - strict: Pokud True, vrací zkrácené odpovědi (max 3 věty bez halucinací)
    - k: Počet relevantních chunků k načtení (default 5)
    - use_llm: Pokud True, použije LLM pro syntetizaci, else vrátí syrové chunky
    """
    question: str
    strict: Optional[bool] = False
    k: Optional[int] = 5
    use_llm: Optional[bool] = True


class QueryLogger:
    """Audit trail: všechny dotazy a odpovědi se loggují do JSONL souboru.
    
    Formát:
    {"timestamp": "2024-...", "question": "...", "answer": "...", "sources": [...], "confidence": 0.85}
    
    Umožňuje zpětné analýzy, auditování a měření kvality.
    """

    def __init__(self, path: str = LOG_FILE):
        self.path = path

    def log(self, question: str, answer: str, sources: List[dict], confidence: float):
        """Zaloguje jeden dotaz se jeho odpovědí a metadaty."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Inicializace RAG engine (lazy loading indexu)
engine = RAGEngine()
try:
    engine.load_index(INDEX_DIR)
    logger.info("✓ FAISS index načten.")
except Exception:
    logger.warning("⚠ Index není dostupný. Spusťte: python build_index.py")


@app.post('/ask')
def ask(req: AskRequest):
    """Odpovídá na otázku pomocí RAG pipeline.
    
    Pipeline:
    1. **Retrieval**: FAISS semantic search vrátí top-k relevantních chunků
    2. **Synthesize**: LLM nebo syrové chunky se spojí v odpověď
    3. **Logging**: Dotaz, odpověď, zdroje a konfidence se zalogují
    
    Odpověď je vždy v jazyce dokumentu (česká PDF → česká odpověď).
    
    Request:
        {
            "question": "Jaká je doba plnění?",
            "strict": false,
            "k": 5,
            "use_llm": true
        }
    
    Response:
        {
            "answer": "Doba plnění je 30 dní...",
            "sources": [
                {"file": "smlouva.pdf", "page": 2, "chunk_id": 5}
            ],
            "confidence": 0.82
        }
    
    Errors:
    - 503: Index není dostupný (spusťte build_index.py)
    """
    # Kontrola: je vůbec index dostupný?
    if engine.index is None:
        raise HTTPException(
            status_code=503, 
            detail="Index není dostupný. Spusťte: python build_index.py"
        )

    # 1️⃣ Semantic retrieval z FAISS
    # Vrátí top-k chunků seřazených podle kosinusové podobnosti
    retrieved = engine.retrieve(req.question, k=req.k)

    # 2️⃣ Syntéza odpovědi
    # - Pokud use_llm=True: LLM projednotkuje kontext
    # - Pokud use_llm=False: spojíme chunky do syrového textu
    # - Pokud strict=True: zkrátíme na max 3 věty
    result = engine.synthesize_answer(
        req.question, 
        retrieved, 
        use_llm=req.use_llm,
        strict=req.strict
    )

    # 3️⃣ Audit logging: zalogujeme všechny dotazy
    qlogger = QueryLogger()
    qlogger.log(
        req.question, 
        result.get('answer', ''), 
        result.get('sources', []), 
        result.get('confidence', 0.0)
    )

    return result
