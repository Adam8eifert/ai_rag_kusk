# RAG pro PDF smlouvy

**Co program umí:**
- Odpovídá na dotazy pouze z obsahu PDF smluv (žádné halucinace)
- Odpovědi jsou vždy v češtině
- Vrací citace zdrojů (soubor, stránka, chunk)
- REST API (FastAPI, endpoint `/ask`), webové rozhraní (Swagger UI)

**Struktura projektu:**
```
data/         # PDF smlouvy
index/        # FAISS index a metadata
app.py        # FastAPI server
build_index.py# Indexace PDF
rag.py        # Jádro RAG (chunking, retrieval)
llm.py        # Wrapper pro LLM (OpenAI/FLAN-T5)
pdf_loader.py # Extrakce textu z PDF
rules.py      # Pravidla pro typy otázek
requirements.txt
README.md
```

**Instalace závislostí:**
```bash
pip install -r requirements.txt
```

**Použití:**
1. Vlož PDF smlouvy do složky `data/`
2. Vytvoř index:
   ```bash
   python build_index.py
   ```
3. Spusť API server:
   ```bash
   uvicorn app:app --reload
   ```
4. Otevři webové rozhraní: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Formát dotazů:**
- Dotazy pište v češtině jako prostý text (např. "Jaká je záruční doba?", "Jaké jsou platební podmínky?")

**API Request/Response:**

Request (POST /ask):
```json
{
   "question": "Jaká je doba plnění?",
   "strict": false,
   "k": 5,
   "use_llm": true
}
```

Response:
```json
{
   "answer": "Doba plnění je 30 dní...",
   "sources": [
      {"file": "smlouva.pdf", "page": 2, "chunk_id": 5}
   ],
   "confidence": 0.82
}
```

**Co by šlo zlepšit:**
- Přidat podporu pro více jazyků
- Vylepšit extrakci tabulek a čísel z PDF
- Zpřehlednit logování a audit dotazů
- Přidat jednoduché webové UI pro běžné uživatele
