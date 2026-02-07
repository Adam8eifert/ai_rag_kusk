# LLM Integration: Kompletní průvodce

## Diagram: Jak to funguje

```
Dotaz → FAISS retrieve → LLM syntetizace → REST API response
                     ↓
              (kontext)
            top-k chunky
         se systém promptem
```

## Konfigurace: OpenAI vs Local

### 🟢 Varianta 1: OpenAI API (doporučeno)

```bash
# 1. Instalace (je v requirements.txt)
pip install openai>=1.0.0

# 2. Nastavení API key
export OPENAI_API_KEY="sk-..."

# 3. Test
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Jaká je doba plnění?", "use_llm": true}'
```

**Výhody:**
- Vysoká kvalita odpovědí
- Podpora češtiny
- Bezpečnost (bez stažení velkého modelu)

**Nevýhody:**
- Placené (0.0005 USD per 1K tokens)
- Závislost na OpenAI

### 🟡 Varianta 2: Local FLAN-T5 (fallback)

Automaticky se používá, pokud:
- `OPENAI_API_KEY` není nastavený
- `openai` není nainstalovaný

**Výhody:**
- Zdarma
- Offline
- Bez latence

**Nevýhody:**
- Kvalita odpovědí je nižší
- Vyžaduje ~2GB VRAM
- Generace může být pomalá

---

## Spuštění serveru

### Development (local FLAN-T5):
```bash
cd /home/adam/Dokumenty/projects/ai_rag_kusk
source .venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production (OpenAI):
```bash
export OPENAI_API_KEY="sk-..."
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## API Endpoint

### POST `/ask`

**Request:**
```json
{
  "question": "Jaká je doba plnění?",
  "k": 5,
  "use_llm": true
}
```

**Response:**
```json
{
  "answer": "Doba plnění je 30 dnů od podpisu smlouvy.",
  "sources": [
    {
      "file": "smlouva.pdf",
      "page": 3,
      "chunk_id": 1
    }
  ],
  "confidence": 0.785
}
```

**Chování:**
- `use_llm=true`: LLM syntetizuje odpověď (kvalita)
- `use_llm=false`: Vrátí syrové chunky (rychleji, bez syntézy)
- `k`: Počet retrievených chunků (default: 5)

---

## Prompts

### Systém:
```
Jsi extrakční asistent. Odpovídej pouze na základě poskytnutého kontextu. 
Pokud odpověď v kontextu není, řekni přesně: 
'Požadovaná informace není v dokumentech explicitně uvedena.'
Odpověz jednou nebo dvěma větami bez parafrází právního obsahu.
```

### User:
```
KONTEXT:
{retrieved_chunks_joined}

OTÁZKA:
{question}

Odpověz:
```

---

## Testování

```bash
# Test script
python test_llm_synthesis.py

# cURL
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Kdo je objednatelem smlouvy?",
    "k": 5,
    "use_llm": true
  }'
```

---

## Troubleshooting

| Problém | Řešení |
|---------|--------|
| `ModuleNotFoundError: openai` | `pip install openai>=1.0.0` |
| OPENAI_API_KEY error | `export OPENAI_API_KEY="sk-..."` |
| Pomalá odpověď | Snižte `k` parametr, nebo použijte `use_llm=false` |
| FLAN-T5 VRAM error | Redeploy s OpenAI API |
| Odpověď je v angličtině | Testujete s FLAN-T5; přepněte na OpenAI |

---

## Architektura

```
app.py (endpoint /ask)
  └─> RAGEngine.retrieve()           # FAISS
  └─> RAGEngine.synthesize_answer() # LLM wrapper
       └─> LLMWrapper.synthesize()
            ├─> OpenAI API (priority)
            └─> Local FLAN-T5 (fallback)
```

---

## Performance

| LLM | Latence | Kvalita | Náklady |
|-----|---------|---------|---------|
| OpenAI GPT-3.5 | ~500ms | ★★★★★ | 0.0005 USD/req |
| FLAN-T5 local | ~2s | ★★☆ | 0 USD |

---

## Bezpečnost

- **Context injection**: LLM je instrukován, aby ignoroval mimo-kontext
- **Token limits**: Max 200 tokens na odpověď
- **Fallback**: Pokud LLM selže, vrací se syrý kontext
