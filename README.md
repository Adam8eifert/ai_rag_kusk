# AI RAG pipeline – dotazování nad PDF dokumenty (bez halucinací)

Tento projekt implementuje **robustní RAG (Retrieval‑Augmented Generation) systém** nad **lokálními PDF dokumenty** (např. smlouvami, interní dokumentací).

Hlavním cílem je:

* pracovat výhradně s uživatelem dodanými dokumenty,
* zabránit halucinacím,
* vracet pouze odpovědi podložené zdroji s citacemi,
* umožnit bezpečné použití v „enterprise" prostředí,
* **odpovídat výhradně v češtině** (bez překladu, v jazyce dokumentu).

Projekt je **hybridní**: lze jej používat jako:
* **Pure Extractive RAG** (bez LLM, deterministic, nejbezpečnější),
* **RAG s LLM syntézou** (LLM zkrátí/přeformuluje text, ale nikdy si nic nedomýšlí).

---

## Co tento projekt dělá

* načte PDF dokumenty z lokální složky `data/`,
* extrahuje text (stránku po stránce),
* rozdělí jej na overlapping chunky (sliding window),
* vytvoří embeddingy pomocí Sentence Transformers (multilingual),
* uloží vektorový index (FAISS IndexFlatIP),
* umožní dotazování přes REST API (FastAPI),
* **odpovídá výhradně z obsahu dokumentů** (bez domýšlení),
* **cituje source** (soubor, stránka, chunk_id) + confidence score.

---

## 🛡️ Striktní pravidla bez halucinací (Compliance-friendly)

Projekt implementuje **tři vrstvy ochrany** proti halucinacím a spekulacím:

### 1️⃣ Hard Factual Gate (threshold 0.72)

```
if top_score < 0.72:
    ❌ LLM se NESMÍ zavolat
    ✅ Vrát fallback odpověď
```

**Princip:** Pokud nejrelevantnější chunk má cosine similarity < 0.72 (tj. skóre < 72%), vrací se okamžitě fallback zpráva. LLM se volá **jen pokud** score >= 0.72.

```python
{
  "answer": "Požadovaná informace není v dokumentech.",
  "sources": [],
  "confidence": 0.68
}
```

### 2️⃣ Keyword Guard (relevance check)

Před zavoláním LLM se ověří, že:
- Otázka obsahuje alespoň 2 klíčová slova
- Kontext obsahuje alespoň 2 z těchto slov
- Pokud ne → fallback (otázka je mimo scope dokumentu)

**Příklady:**
- ✅ "Jaký je doba plnění?" + kontext s "doba" + "plnění" = OK
- ❌ "Kdo je skutečný vlastník?" + smlouva bez "vlastník" = Fallback
- ❌ "Jaké riziko smlouva představuje?" (evaluační, ne faktická) = Fallback

### 3️⃣ LLM Compression Mode (ne generátor)

LLM (pokud je zapnutý) je **JEN kompresor**:
- ✅ Zkrátí text na max 3 věty
- ✅ Zachovává faktické znění
- ❌ Nesmí generovat nové informace
- ❌ Nesmí odpovídat sám bez kontextu

**System prompt:**
```
"Nepřidávej žádné nové informace"
"Odpovídej POUZE z poskytnutého textu"
"Pokud nejsi si jistý, raději vynech"
```

---

## Architektura

```
PDF (data/)
  ↓
Extrakce textu (pdfplumber) - stránka po stránce
  ↓
Chunking (sliding window: chunk_size=200 slov, overlap=50)
  ↓
Embeddingy (SentenceTransformer: paraphrase-multilingual-MiniLM-L12-v2)
  ↓
Normalizace vektorů (L2 norm = jednotková délka)
  ↓
FAISS Index (IndexFlatIP: vnitřní součin = kosinusová podobnost)
  ↓
Semantic retrieval (top-k chunky seřazené podle similarity)
  ↓
┌─ Extractive mode (use_llm=False) → syrové chunky
└─ LLM Synthesis (use_llm=True) → GPT-3.5 (OpenAI) nebo FLAN-T5 (offline)
  ↓
[SYSTEM_PROMPT: "Odpovídej POUZE z kontextu, bez domýšlení"]
  ↓
Odpověď + sources + confidence score
  ↓
FastAPI /ask endpoint
  ↓
Logging do queries.jsonl (audit trail)
```

---

## Struktura projektu

```
.
├── data/               # Vstupní PDF dokumenty (sem vkládej smlouvy atd.)
├── index/              # FAISS index + metadata (automaticky generované)
├── logs/               # queries.jsonl (audit trail všech dotazů)
│
├── build_index.py      # Indexace: PDF → chunky → embeddingy → FAISS
├── rag.py              # RAGEngine (chunking, embedding, retrieval, synthesis)
├── app.py              # FastAPI aplikace s /ask endpointem
├── llm.py              # LLM wrapper (OpenAI API / FLAN-T5 fallback)
├── pdf_loader.py       # PDF text extraction (pdfplumber)
├── rules.py            # Optional: question type rules
│
├── requirements.txt    # Pip dependencies
└── README.md           # Tato dokumentace
```

---

## Popis hlavních souborů

### `build_index.py` – Vytvoření FAISS indexu

```bash
python build_index.py
```

**Procedura:**
1. Najde všechny `*.pdf` soubory v `data/`
2. Extrahuje text (pdfplumber, per page)
3. Rozdělí na chunky (sliding window: chunk_size=200 slov, overlap=50)
4. Vypočítá embeddingy (SentenceTransformer) po dávkách
5. Normalizuje vektory (vyžaduje IndexFlatIP)
6. Vytvoří FAISS index

**Výstup:**
```
index/
├── faiss.index         # Vektorový index (binární, IndexFlatIP)
└── documents.json      # Metadata: file, page, chunk_id, text
```

> Při změně dokumentů v `data/` je nutné znovu spustit `build_index.py`.

---

### `rag.py` – RAGEngine (jádro systému)

Třída `RAGEngine` zajišťuje:

* **chunking** (sliding window s overlappem),
* **embedding** (SentenceTransformer encoder, 384-dim),
* **retrieval** (FAISS semantic search, cosine similarity),
* **answer synthesis** (extractive nebo LLM-based),
* **confidence scoring** (avg similarity Retrieved chunks).

#### Dva módy odpovídání

**Mód 1: Extractive** (`use_llm=False`)
* Vrací raw text z nejrelevantnějších chunků
* Deterministické, bez LLM
* Fastest, nízká latence
* Ideální pro "fully auditable" answers

**Mód 2: LLM Synthesis** (`use_llm=True`)
* **Priorita:** OpenAI API (GPT-3.5-turbo, pokud je `OPENAI_API_KEY`)
* **Fallback:** Místní FLAN-T5 (offline, bez API klíče)
* LLM **zkrátí/přeformuluje** text ale **vždy jen z kontextu**
* System prompt: "Odpovídej POUZE z poskytnutého kontextu, bez domýšlení"

##### SYSTEM_PROMPT – Srdce bezpečnosti

```python
SYSTEM_PROMPT = """
Jsi extrakční asistent pro analýzu smluvních dokumentů.

Odpovídej výhradně v českém jazyce.

Odpovídej POUZE na základě poskytnutého kontextu.
Nic si nedomýšlej, neodvozuj a nepřidávej.

Pokud odpověď nelze jednoznačně najít v kontextu,
odpověz přesně touto větou:
"Požadovaná informace není v dokumentech explicitně uvedena."
"""
```

Tento prompt se **automaticky** předává:
* **OpenAI API:** `messages = [{"role": "system", "content": SYSTEM_PROMPT}, ...]`
* **FLAN-T5:** `prompt = "{SYSTEM_PROMPT}\n\nKONTEXT:\n{context}\n\nOTÁZKA:\n{question}"`

---

### `app.py` – FastAPI aplikace

Jediný endpoint:

```
POST /ask
```

#### Request

```json
{
  "question": "Jaká je doba plnění?",
  "strict": false,
  "k": 5,
  "use_llm": true
}
```

**Parametry:**
* `question` *(required)* – Dotaz v češtině
* `strict` *(optional, default=False)* – Pokud True: nižší práh, kratší odpovědi (2-3 věty)
* `k` *(optional, default=5)* – Počet chunků k načtení z FAISS
* `use_llm` *(optional, default=True)* – Pokud True: LLM synthesis; pokud False: raw text

#### Response

```json
{
  "answer": "Doba plnění je 30 dní od podpisu smlouvy.",
  "sources": [
    {
      "file": "smlouva_ABC.pdf",
      "page": 2,
      "chunk_id": 5
    }
  ],
  "confidence": 0.87
}
```

**Pole:**
* `answer` – Odpověď na otázku (Czech)
* `sources` – List zdrojů (file, page, chunk_id)
* `confidence` – Cosine similarity (0.0–1.0)

---

### `llm.py` – LLM wrapper (hibridní design)

```python
from llm import LLMWrapper

llm = LLMWrapper(use_openai=True)  # Nebo False
answer = llm.synthesize(question, context)
```

**Inicializace:**
1. Zkusí načíst `OPENAI_API_KEY` environment variable
2. Pokud existuje: inicializuje `openai.OpenAI()` client → **OpenAI mode**
3. Pokud ne: fallback na `google/flan-t5-base` → **FLAN-T5 mode** (offline)
4. Pokud ani to není dostupné: vrací syrový kontext

**Systémový prompt** je konsistentní v obou mode (viz `SYSTEM_PROMPT` výše).

---

### `pdf_loader.py` – PDF text extraction

```python
from pdf_loader import load_pdf

pages = load_pdf("data/smlouva.pdf")
# Vrátí: [{"text": "...", "page": 1, "source": "..."}, ...]
```

Extrahuje text stránku po stránce. Pokud je PDF obrázek bez OCR, vrátí `text=""`.

---

### `rules.py` – Optional question type filtering

```python
QUESTION_RULES = {
    "doba_plneni": {
        "question_keywords": ["doba plnění", "termín plnění"],
        "section_keywords": ["doba plnění", "plnění smlouvy"],
        "min_similarity": 0.75
    },
    ...
}
```

Používá se v `rag.answer_question()` (extractive mode s tvrdými pravidly).

---

## Instalace a spuštění

### 1️⃣ Vytvoření virtuálního prostředí

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# nebo
.venv\Scripts\activate         # Windows
```

### 2️⃣ Instalace závislostí

```bash
pip install -r requirements.txt
```

Obsah `requirements.txt`:
```
pdfplumber~=0.10
sentence-transformers~=2.2
faiss-cpu~=1.8
numpy~=1.23
fastapi~=0.104
uvicorn~=0.24
```

**OpenAI API (optional):**
```bash
pip install openai~=1.0
```

Bez OpenAI balíčku se projekt automaticky přepne na FLAN-T5 (offline).

### 3️⃣ Příprava dokumentů

Vložte PDF soubory do `data/`:

```
data/
├── smlouva_ABC.pdf
├── smlouva_DEF.pdf
└── interni_pravidla.pdf
```

### 4️⃣ Indexace dokumentů

```bash
python build_index.py
```

Očekávaný výstup:
```
🚀 Inicializace RAGEngine...
📁 Budování indexu z PDF souborů v 'data/' adresáři...
📄 Čtení: smlouva_ABC.pdf
📄 Čtení: smlouva_DEF.pdf
✓ Sbíráno 156 chunků
🔄 Enkódování 156 textů po dávkách x64...
✓ Embeddingy shape: (156, 384)
✓ Vektory normalizovány
✓ FAISS IndexFlatIP vytvořen (dim=384, n=156)
✅ Index uložen: index/faiss.index
✅ Metadata uložena: index/documents.json
```

### 5️⃣ Spuštění API

```bash
uvicorn app:app --reload
```

Nebo s customním hostem/portem:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

API dostupné na:
* **API root:** http://127.0.0.1:8000
* **Swagger UI:** http://127.0.0.1:8000/docs ← **Zde testuj**
* **ReDoc:** http://127.0.0.1:8000/redoc

---

## Praktické příklady

### Příklad 1: Extractive mode (bez LLM, pure deterministic)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Jaká je doba plnění?",
    "use_llm": false
  }'
```

**Odpověď:**
```json
{
  "answer": "Doba plnění je 30 dní od podpisu smlouvy. Plnění se musí uskutečnit v pracovních dnech. Dodavatel je povinen dodržovat dohodnuté termíny.",
  "sources": [
    {"file": "smlouva_ABC.pdf", "page": 2, "chunk_id": 3}
  ],
  "confidence": 0.92
}
```

### Příklad 2: LLM synthesis (zkrácená, přeformulovaná odpověď)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Jaká je doba plnění?",
    "use_llm": true,
    "strict": true
  }'
```

**Odpověď (zkrácená OpenAI/FLAN-T5):**
```json
{
  "answer": "Doba plnění smlouvy je 30 dní od podpisu.",
  "sources": [
    {"file": "smlouva_ABC.pdf", "page": 2, "chunk_id": 3}
  ],
  "confidence": 0.92
}
```

### Příklad 3: Swagger UI – Interaktivní testování

1. Otevři v prohlížeči: **http://127.0.0.1:8000/docs**
2. Klikni na `POST /ask`
3. Vyplň parametry:
   - `question`: "Jaká je doba plnění?"
   - `strict`: zaškrtni
   - `use_llm`: zaškrtni
4. Klikni **"Try it out"** → vidíš live odpověď

---

## Audit logging

Všechny dotazy jsou loggány do `logs/queries.jsonl`:

```json
{"timestamp": "2026-02-08T10:15:23.123456", "question": "Jaká je doba plnění?", "answer": "30 dní", "sources": [{"file": "smlouva.pdf", "page": 2, "chunk_id": 5}], "confidence": 0.92}
{"timestamp": "2026-02-08T10:16:45.654321", "question": "Kdo je objednatel?", "answer": "Firma ABC s.r.o.", "sources": [...], "confidence": 0.88}
```

Pro čtení Log souborů:
```python
import json

with open("logs/queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        print(f"Q: {record['question']}")
        print(f"A: {record['answer']}")
        print(f"Confidence: {record['confidence']}\n")
```

---

## Testování Compliance Pravidel

Spusť skript na ověření, že RAG dodržuje všechna striktní pravidla:

```bash
python test_compliance.py
```

**Kontroluje:**
- ✅ Hard factual gate (score < 0.72 → fallback bez LLM)
- ✅ Keyword guard (relevantní slova v kontextu)
- ✅ LLM jen jako kompresor (ne generátor)
- ✅ Fallback konzistence

---

## Parametry a konfigurace

### Confidence threshold a Hard Factual Gate

```python
# V rag.py - Hard factual gate (POVINNÉ)
if top_score < 0.72:
    return {"answer": "Požadovaná informace není v dokumentech.", ...}
    # LLM se NESMÍ zavolat!
```

Pokud je nejrelevantnější chunk má cosine similarity < 0.72, vrátí se fallback bez zavolání LLM. Toto je **kritické pravidlo** proti spekulacím.

### Chunk size a overlap

```python
# V build_index.py
engine.build_index(
    chunk_size=200,  # Počet slov v jednom chunku
    overlap=50       # Počet slov překryvu
)
```

Vyšší overlap = lepší seamless transition, ale je více chunků.

### Strict mode

```python
# request
{"question": "...", "strict": true}
```

V strict mode se:
* používá nižší confidence threshold,
* odpověď se zkrátí na max 2-3 věty,
* vrací "Nevím" místo nejistých odpovědí.

---

## Poznámky k návrhu

✅ **Bezpečnost:**
* Žádné scrapování webu
* Žádná externí data
* Žádné halucinace
* Plně auditovatelné odpovědi
* SYSTEM_PROMPT garantuje extrakci jen z kontextu

✅ **Vhodné pro:**
* Smlouvy a právní dokumenty
* Compliance dokumentaci
* Interní dokumenty
* Poznámky a příslušné
* Q&A systémy nad proprietary datou

❌ **Nevhodné pro:**
* Open-ended konverzace
* Otázky vyžadující externí znalosti
* Jazykové hry a humor
* Tvoření nového obsahu (creative writing)

---

## Troubleshooting

### Problém: "Index není dostupný"

```
HTTPException 503: Index není dostupný
```

**Řešení:**
```bash
python build_index.py
```

Musíš nejdřív vytvořit index.

### Problém: "ModuleNotFoundError: No module named 'openai'"

To je OK! Projekt fallbackuje na FLAN-T5. Pokud chceš OpenAI:
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

### Problém: Pomalá odpověď

Pravděpodobně FLAN-T5 běží na CPU. Řešení:
1. Instaluj OpenAI API (veel rychlejší)
2. Nebo použij `use_llm=false` (extractive mode)

### Problém: Odpověď je v angličtině (u Czech PDF)

SYSTEM_PROMPT říká "Odpovídej výhradně v českém jazyce". Pokud to nefunguje:
* Ověř že je SYSTEM_PROMPT korektně předán do LLM
* Zkus extractive mode (`use_llm=false`)
* Zktr ověř, že Python soubor má encoding `utf-8`

---

## Performance

| Faktor | Dopad |
|--------|-------|
| PDF velikost | Malý (chunking je offline) |
| Počet PDF | Malý (indexace je jedenkrát) |
| Query latence (bez LLM) | **< 100ms** (FAISS je velmi rychlý) |
| Query latence (s OpenAI) | **1-3s** (API latence) |
| Query latence (s FLAN-T5) | **2-5s** (CPU inference) |
| Embedding model size | ~460 MB (loaded once) |
| FAISS index size | ~1MB per 1000 vektory |

---

**Autor:** Adam Seifert  
**Poslední aktualizace:** 2026-02-08  
**License:** MIT
