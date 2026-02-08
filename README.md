
pdfplumber~=0.10
data/

# AI RAG – dotazování nad PDF smlouvami

Rychlý systém pro odpovídání na otázky z PDF dokumentů (smlouvy, právní texty) bez halucinací.

**Klíčové vlastnosti:**
- Odpovědi pouze z obsahu PDF (žádné spekulace)
- Vždy v češtině
- Citace zdrojů (soubor, stránka, chunk)
- REST API (FastAPI, endpoint `/ask`)
- LLM (OpenAI/FLAN-T5) pouze komprimuje, negeneruje

## Rychlý start

1. Vlož PDF soubory do složky `data/`
2. Vytvoř virtuální prostředí a nainstaluj závislosti:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Vytvoř index:
   ```bash
   python build_index.py
   ```
4. Spusť API:
   ```bash
   uvicorn app:app --reload
   ```

## Použití

Dotaz přes API:
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Jaká je záruční doba?",
    "use_llm": true
  }'
```

**Odpověď:**
```json
{
  "answer": "Záruční doba je 24 měsíců a počíná plynout ode dne předání zboží.",
  "sources": [
    {"file": "smlouva.pdf", "page": 3, "chunk_id": 2}
  ],
  "confidence": 0.69
}
```

## Compliance ochrana
- Hard factual gate: pokud skóre < 0.50 → fallback
- Keyword guard: kontext musí obsahovat klíčová slova z otázky
- LLM pouze komprimuje, nikdy negeneruje

## Licence
MIT
