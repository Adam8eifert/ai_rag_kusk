# AI RAG pipeline – dotazování nad PDF dokumenty (bez halucinací)

Tento projekt implementuje **deterministický, produkčně realistický RAG (Retrieval‑Augmented Generation) systém** nad **lokálními PDF dokumenty** (např. smlouvami, interní dokumentací).

Hlavním cílem je:

* pracovat výhradně s uživatelem dodanými dokumenty,
* zabránit halucinacím,
* vracet pouze odpovědi podložené zdroji,
* umožnit bezpečné použití v „enterprise“ prostředí.

Projekt **záměrně nepoužívá generativní LLM modely při odpovídání** – jedná se o **extractive RAG**.

---

## Co tento projekt dělá

* načte PDF dokumenty z lokální složky,
* extrahuje text,
* rozdělí jej na menší bloky (chunky),
* vytvoří embeddingy pomocí Sentence Transformers,
* uloží vektorový index (FAISS),
* umožní dotazování přes API,
* odpovídá **výhradně z obsahu dokumentů**.

---

## Co tento projekt NENÍ

* ❌ chatbot
* ❌ generativní AI
* ❌ model trénovaný na externích datech
* ❌ systém pro open‑ended otázky

---

## Architektura

```
PDF (data/)
  ↓
Extrakce textu (pdfplumber)
  ↓
Chunking + embeddingy (Sentence Transformers)
  ↓
FAISS index
  ↓
RAG logika (retrieval + extractive answering)
  ↓
FastAPI / QA služba
```

---

## Struktura projektu

```
.
├── data/               # Vstupní PDF dokumenty
├── index/              # FAISS index + metadata
├── build_index.py      # Indexace PDF dokumentů
├── rag.py              # RAGEngine (retrieval + odpovědi)
├── app.py              # FastAPI API (/ask)
├── llm.py              # Volitelná / nepoužitá LLM vrstva (placeholder)
├── requirements.txt
└── README.md
```

---

## Popis hlavních souborů

### `build_index.py`

* načte PDF soubory ze složky `data/`,
* extrahuje text,
* rozdělí text na chunky,
* vytvoří embeddingy,
* uloží:

  * FAISS index,
  * metadata (soubor, stránka, chunk).

> **Indexaci je nutné znovu spustit při každé změně dokumentů.**

---

### `rag.py`

Obsahuje třídu `RAGEngine`, která zajišťuje:

* chunking a embeddingy,
* načtení FAISS indexu,
* vyhledání relevantních chunků,
* **extraktivní odpovědi bez použití LLM**,
* konzervativní chování při nízké podobnosti.

#### Chování odpovědí

* žádná relevantní data → `Nemám odpověď v dokumentech`,
* nízká podobnost:

  * `strict = false` → „Otázka se netýká obsahu dokumentů.“,
  * `strict = true` → `Nevím`,
* vysoká podobnost → odpověď složená **ze zdrojových úryvků**.

V režimu `strict=true` je odpověď automaticky **zkrácena na 2–3 věty** a nikdy není domýšlena.

---

### `app.py`

* FastAPI aplikace,
* endpoint `POST /ask`,
* propojuje HTTP API s třídou `RAGEngine`.

---

### `llm.py`

* **není používán v runtime**,
* slouží jako architektonický placeholder,
* odděluje případnou budoucí generativní vrstvu od retrieval logiky.

> Projekt je navržen tak, aby **plně fungoval bez LLM** a bylo možné LLM doplnit později bez zásahu do RAG jádra.

---

## Použité knihovny a modely

### Knihovny

* `pdfplumber` – extrakce textu z PDF
* `sentence-transformers` – embeddingy
* `faiss-cpu` – vektorový index
* `numpy`
* `fastapi`
* `uvicorn`

### Embedding model

**sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**

* multijazyčný (vhodný pro češtinu),
* rychlý a lehký,
* vhodný pro produkční použití.

---

## Instalace a spuštění

### 1️⃣ Vytvoření virtuálního prostředí

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3️⃣ Příprava dokumentů

Vložte PDF soubory do složky `data/`:

```
data/
├── smlouva_ABC.pdf
├── smlouva_DEF.pdf
```

### 4️⃣ Indexace dokumentů

```bash
python build_index.py
```

### 5️⃣ Spuštění API

```bash
uvicorn app:app --reload
```

API poběží na:

* [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Použití API

### Dotaz

```bash
POST /ask
{
  "question": "Jaká je doba platnosti smlouvy?",
  "strict": true
}
```

### Odpověď

```json
{
  "answer": "Smlouva je uzavřena na dobu určitou do 31. 12. 2025.",
  "sources": [
    { "file": "smlouva_ABC.pdf", "page": 3, "chunk_id": 1 }
  ],
  "confidence": 0.87
}
```

---

## Význam `confidence`

Hodnota `confidence` odpovídá **cosine similarity** mezi embeddingem dotazu a nejrelevantnějším chunkem.

---

## Strict mode (doporučeno)

* vyšší práh podobnosti,
* kratší odpovědi (2–3 věty),
* žádné domýšlení,
* místo nejistých odpovědí vrací `Nevím`.

---

## Poznámky k návrhu

* žádné scrapování webu,
* žádná externí data,
* žádné halucinace,
* plně auditovatelné odpovědi,
* vhodné pro smlouvy, compliance a interní dokumenty.

---

**Autor:** Adam Seifert
