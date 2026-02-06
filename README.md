# AI RAG pipeline – práce s PDF dokumenty

Tento projekt implementuje **jednoduchý, ale produkčně realistický RAG (Retrieval-Augmented Generation) systém** nad lokálními PDF dokumenty (např. smlouvy). Cílem je:

* načíst a zpracovat PDF dokumenty,
* rozdělit je na textové bloky (chunky),
* vytvořit vektorový index pomocí embedding modelu,
* umožnit dotazování přes API s důrazem na *strict mode* (odpovědi pouze z dat).

Projekt je navržen tak, aby byl **deterministický, snadno spustitelný a čitelný** – bez závislosti na scrapování webu.

---

## Architektura projektu

```
PDF (data/)
   ↓
Extrahování textu (pdfplumber)
   ↓
Chunking + embeddingy (Sentence Transformers)
   ↓
FAISS index
   ↓
RAG logika (retrieval + LLM prompt)
   ↓
FastAPI / QA služba
```

### Struktura souborů

```
.
├── data/                 # Vstupní PDF dokumenty
├── build_index.py        # Vytvoření FAISS indexu z PDF
├── rag.py                # Konsolidovaný RAG engine (retrieval + answer logic)
├── app.py                # FastAPI aplikace (QA endpoint)
├── requirements.txt      # Python závislosti
├── README.md
```

### Popis hlavních souborů

* **build_index.py**

  * načte PDF z `data/`
  * extrahuje text
  * rozdělí text na chunky
  * vytvoří embeddingy
  * uloží FAISS index + metadata

* **rag.py**

  * obsahuje `RAGEngine` — chunking, embeddingy, načtení FAISS indexu
  * vyhledá relevantní chunky v indexu a vrátí skóre + metadata
  * aplikační logika `answer_question` bez volání LLM: vrací buď zdrojové úryvky, nebo konzervativní odpovědi (např. `Nemám odpověď v dokumentech`, `Otázka se netýká obsahu dokumentů.` nebo v `strict` režimu `Nevím`)

* **app.py**

  * FastAPI aplikace
  * endpoint `/ask`
  * propojuje dotaz uživatele s RAG logikou

---

## Použité knihovny a modely

### Python knihovny

* `pdfplumber` – extrakce textu z PDF
* `sentence-transformers` – embedding model
* `faiss-cpu` – vektorový index
* `numpy` – práce s maticemi
* `fastapi` – API vrstva
* `uvicorn` – ASGI server

### Embedding model

* **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**

  * multijazyčný
  * vhodný pro češtinu
  * rychlý a lehký

---

## Instalace a spuštění – krok za krokem

### 1️⃣ Vytvoření a aktivace virtuálního prostředí

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 2️⃣ Instalace závislostí

```bash
pip install -r requirements.txt
```

Obsah `requirements.txt` (příklad):

```
pdfplumber
sentence-transformers
faiss-cpu
numpy
fastapi
uvicorn
```

---

### 3️⃣ Příprava dokumentů

* vytvořte složku `data/`
* vložte do ní **PDF dokumenty** (např. 5–10 smluv)

```
data/
├── smlouva_ABC.pdf
├── smlouva_DEF.pdf
└── ...
```

Názvy souborů mohou být **libovolné**.

---

### 4️⃣ Indexování dokumentů

```bash
python build_index.py
```

Výstup:

* vytvoření FAISS indexu
* informace o počtu chunků

---

### 5️⃣ Spuštění QA služby (API)

Spusťte pomocí `uvicorn` (doporučeno):

```bash
uvicorn app:app --reload
```

API poběží na:

```
http://127.0.0.1:8000
```

---

## Použití API – příklady dotazů

### Dotaz

```json
POST /ask
{
  "question": "Jaká je doba platnosti smlouvy?",
  "strict": true
}
```

### Očekávaná odpověď (pokud je v datech)

```json
{
  "answer": "Smlouva je uzavřena na dobu určitou do 31. 12. 2025.",
  "sources": [
    {
      "file": "smlouva_ABC.pdf",
      "page": 3
    }
  ]
}
```

### Odpovědi

- Pokud není v datech odpověď: `Nemám odpověď v dokumentech`.
- Pokud nejsou chunk-y dostatečně podobné (pod prahem), vrací se: `Otázka se netýká obsahu dokumentů.` (v `strict` režimu místo toho `Nevím`).

Příklad (pokud je v datech):

```json
{
  "answer": "(úryvek z relevantních chunků)",
  "sources": [
    {"file": "smlouva_ABC.pdf", "page": 3, "chunk_id": 1}
  ],
  "confidence": 0.87
}
```

---

## Poznámky k návrhu

* Projekt **záměrně nepoužívá scrapování** – ingest dat je lokální
* Cílem je stabilita, testovatelnost a kontrola nad vstupy
* Strict mode zabraňuje halucinacím LLM
* Architektura odpovídá běžnému enterprise RAG řešení

---

## Možná rozšíření

* UI (např. jednoduchý frontend)
* logování dotazů
* per-document metadata
* autentizace API

---

Autor: *Adam Seifert*
