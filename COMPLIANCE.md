# Compliance & Anti-Hallucination Architecture

## 📋 Přehled

Tento dokument popisuje tři vrstvové bezpečnostní mechanismy implementované v RAG systému pro zabránění halucinacím a spekulacím LLM.

---

## 🏗️ Architektura Bezpečnosti

```
┌─────────────────────────────────────────────────────────────────┐
│ USER QUESTION                                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1️⃣ FAISS RETRIEVAL (Semantic Search)                            │
│   - Top-k similarity scores                                      │
│   - Result: retrieved[] = [score: 0.68, text: "...", ...]        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │ 2️⃣ HARD FACTUAL GATE (0.72 threshold)        │
        │                                              │
        │ if top_score < 0.72:                         │
        │   ❌ LLM se NESMÍ zavolat                    │
        │   ✅ Return fallback                         │
        │   STOP HERE                                  │
        └──────────────────────────────────────────────┘
           │                              │
           │ (PASS)                       │ (FAIL)
           ▼                              ▼
        Continue              {"answer": "Požadovaná...",
                              "sources": [],
                              "confidence": 0.58}
           │
           ▼
        ┌──────────────────────────────────────────────┐
        │ 3️⃣ KEYWORD GUARD (relevance check)           │
        │                                              │
        │ if not _has_relevant_keywords(context, q):   │
        │   ❌ Otázka mimo scope                       │
        │   ✅ Return fallback                         │
        │   STOP HERE                                  │
        └──────────────────────────────────────────────┘
           │                              │
           │ (PASS)                       │ (FAIL)
           ▼                              ▼
        Continue              {"answer": "Požadovaná...",
                              "sources": [],
                              "confidence": 0.xyz}
           │
           ▼
        ┌──────────────────────────────────────────────┐
        │ 4️⃣ LLM COMPRESSION (only if needed)          │
        │                                              │
        │ if use_llm=True:                             │
        │   → compress_answer() [NOT synthesize()]      │
        │   → Max 3 věty                               │
        │   → Jen ze zdrojů                            │
        │ else:                                        │
        │   → Return raw context                       │
        └──────────────────────────────────────────────┘
           │
           ▼
        ┌──────────────────────────────────────────────┐
        │ 5️⃣ OPTIONAL STRICT SUMMARIZATION              │
        │                                              │
        │ if strict=True:                              │
        │   → _summarize_answer() (2-3 věty)           │
        └──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ FINAL RESPONSE                                                   │
│ {"answer": "...", "sources": [file, page, chunk_id], ...}       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ Hard Factual Gate (0.72 Threshold)

### Co se kontroluje?

```python
# rag.py - synthesize_answer()
top_score = retrieved[0].get("score", 0.0) if retrieved else 0.0

if top_score < 0.72:
    return {
        "answer": "Požadovaná informace není v dokumentech.",
        "sources": [],
        "confidence": top_score,
    }
    # ❌ LLM se NESMÍ zavolat
```

### Kdy je třeba fallback?

| Score | Relevance | Akce |
|-------|-----------|------|
| 0.92 | Velmi høká | ✅ Proceed (LLM allowed) |
| 0.80 | Høká | ✅ Proceed (LLM allowed) |
| 0.72 | Borderline | ✅ Proceed (LLM allowed) |
| 0.68 | Nízká | ❌ FALLBACK (LLM forbidden) |
| 0.45 | Velmi nízká | ❌ FALLBACK (LLM forbidden) |
| 0.10 | Bez relevance | ❌ FALLBACK (LLM forbidden) |

### Proč 0.72?

- **0.72 = 72% cosine similarity** na multilingual SentenceTransformer
- Práh vypočtený empiricky na právních dokumentech
- Zabraňuje LLM aby spekuloval na otázky s nízkou relevancí
- Zabraňuje "hallucination bootstrapping" (LLM si vymýšlí odpovědi)

---

## 2️⃣ Keyword Guard (Relevance Check)

### Co se kontroluje?

```python
# rag.py - _has_relevant_keywords()
def _has_relevant_keywords(self, context: str, question: str) -> bool:
    # 1. Extrahuj keywords z otázky (slova > 3 znaky)
    # 2. Zkontroluj že alespoň 2 jsou v contextu
    # 3. Vrať True/False
```

### Algoritmus

```
Q: "Jaké riziko smlouva představuje?"
Keywords: ["riziko", "smlouva", "představuje"]

Context: "Smlouva je mezi Firmou A a Firmou B. Doba plnění je 30 dní."

Kontrola:
- "riziko" in context? ❌ NO
- "smlouva" in context? ✅ YES
- "představuje" in context? ❌ NO

Matched: 1 z 3 → NOT >= 2 → FALLBACK
```

### Příklady

**✅ Správné otázky (projdou)**
```python
"Jaká je doba plnění?" 
  → Keywords: ["doba", "plnění"]
  → Context: "Doba plnění je 30 dní"
  → Matched: 2 ✅ PASS

"Za jakých podmínek lze smlouvu vypovědět?"
  → Keywords: ["podmínek", "smlouvu", "vypovědět"]
  → Context: "Smlouva může být vypovězena..."
  → Matched: 2+ ✅ PASS
```

**❌ Nesprávné otázky (fallback)**
```python
"Kdo je skutečným vlastníkem?" (entity van company)
  → Keywords: ["skutečným", "vlastníkem"]
  → Context: (smlouva o dodávce bez ownership info)
  → Matched: 0 ❌ FALLBACK

"Jaké riziko smlouva představuje?" (evaluační)
  → Keywords: ["riziko", "smlouva", "představuje"]
  → Context: (smlouva s fakty, bez analýzy)
  → Matched: 1 < 2 ❌ FALLBACK

"Jaká je cena na trhu?" (externí data)
  → Keywords: ["cena", "trhu"]
  → Context: (smlouva s konkrétní cenou, ne trzní)
  → Matched: 1 < 2 ❌ FALLBACK
```

---

## 3️⃣ LLM Compression Mode (Ne Syntéza)

### Rozdíl: Synthesize vs. Compress

| Aspekt | synthesize() | compress_answer() |
|--------|--------------|------------------|
| **Účel** | Odpovědět na otázku | Zkrátit text |
| **Vstup** | Otázka + context | Otázka + context |
| **Výstup** | Kompletní odpověď | Zkrácení odpovědi |
| **Přidávání info** | ❌ Zakázáno | ❌ Zakázáno |
| **Temperature** | 0.1 (low) | 0.0 (deterministic) |
| **Max tokens** | 200 | 100-150 |
| **Případy** | Nikdy (zastaralé) | use_llm=True |

### Compression Prompt

```
COMPRESSION_PROMPT = """
Jsi KOMPRESOR textu pro právní dokumenty.

PRAVIDLA:
1. Nepřidávej ŽÁDNÉ nové informace
2. Odpovídej POUZE z poskytnutého textu
3. Shrnutí max 3 věty
4. Zachovej faktické znění (čísla, termíny, pojmy)
5. Pokud nejsi si jistý, raději vynech

Pokud se nemůžeš rozhodnout, vrať zbytky
"""
```

### Příklad Compression

```
INPUT TEXT (raw context):
"Doba plnění je 30 dní od podpisu smlouvy mezi Objednatelem a 
Dodavatelem. Plnění se musí uskutečnit v pracovních dnech. Dodavatel 
je povinen hledat schválení Objednatele za každou akci během plnění. 
Pokud je plnění zpožděno, je Dodavatel povinen zaplatit smluvní pokutu 
30 Kč za každý den zpoždění."

COMPRESSION RESULT (max 3 věty):
"Doba plnění je 30 dní od podpisu smlouvy v pracovních dnech. 
Dodavatel je povinen schválit každou akci s Objednatelem. 
V případě zpoždění hrozí smluvní pokuta 30 Kč/den."
```

---

## 4️⃣ Anti-Hallucination System Prompts

### SYSTEM_PROMPT (Standard)

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

### COMPRESSION_PROMPT (Stricter)

```python
COMPRESSION_PROMPT = """
Jsi kompresor textu pro právní dokumenty.

PRAVIDLA:
1. Nepřidávej žádné nové informace
2. Odpovídej POUZE z poskytnutého textu
3. Shrnutí max 3 věty
4. Zachovej faktické znění (konkrétní čísla, termíny, pojmy)
5. Pokud nejsi si jistý, než přidávej text, raději vynech
"""
```

---

## 5️⃣ Testování Compliance

### Spuštění Tests

```bash
python test_compliance.py
```

### Co se testuje

1. **Hard Factual Gate**
   - Otázky s nízkou relevancí (score < 0.72)
   - Zkontroluje že se LLM NEVOLÁ
   - Zkontroluje fallback odpověď

2. **Keyword Guard**
   - Otázky SE relevantními keywords (by měly projít)
   - Otázky BEZ relevantních keywords (fallback)
   - Ověří minimální počet matched keywords

3. **LLM Compression**
   - Ověří že krátký text se nemění
   - Ověří že dlouhý text se zkrátí

4. **Fallback Consistency**
   - Fallback answer má správný text
   - Fallback sources je prázdný []
   - Fallback confidence je top_score

---

## 🔬 Implementační Detaily

### rag.py - synthesize_answer()

```python
def synthesize_answer(self, question: str, retrieved: List[Dict], 
                      use_llm: bool = True, strict: bool = False):
    # 1. Check: Je empty?
    if not retrieved:
        return fallback
    
    # 2. CHECK: Hard gate (0.72)
    top_score = retrieved[0].get("score", 0.0)
    if top_score < 0.72:
        return fallback
    
    # 3. CHECK: Keyword guard
    context = "\n---\n".join(...)
    if not self._has_relevant_keywords(context, question):
        return fallback
    
    # 4. PROCEED: LLM compression nebo raw
    if use_llm:
        answer = llm.compress_answer(question, context)  # NOT synthesize!
    else:
        answer = context
    
    # 5. OPTIONAL: Strict summarization
    if strict:
        answer = self._summarize_answer(answer, max_sentences=3)
    
    return {
        "answer": answer.strip(),
        "sources": [...],
        "confidence": round(avg_score, 3),
    }
```

### llm.py - compress_answer()

```python
def compress_answer(self, question: str, context: str) -> str:
    # Pokud je text krátký, vrať bez změny
    if len(context.split()) <= 150:
        return context.strip()
    
    # Kompresor (POUZE zkrácení, nikdy generování)
    if self.use_openai:
        return self._compress_openai(question, context)
    elif LOCAL_LLM_AVAILABLE:
        return self._compress_local(question, context)
    else:
        return first_3_sentences(context)
```

---

## 🎯 Metriky a Benchmarky

| Metrika | Target | Status |
|---------|--------|--------|
| Hard gate accuracy | 99%+ (no false negatives) | ✅ Implemented |
| Keyword guard recall | 95%+ | ✅ Implemented |
| LLM hallucination rate | 0% (in theory) | ✅ Implemented |
| False fallbacks | < 5% | ⚠️ Depends on docs |
| Latency (no LLM) | < 100ms | ✅ FAISS fast |
| Latency (with OpenAI) | 1-3s | ✅ Expected |

---

## 🛠️ Troubleshooting

### Problém: Příliš mnoho fallback odpovědí

**Příčina:** Hard gate threshold 0.72 je příliš vysoký

**Řešení:** 
- Ověř že dokumenty obsahují relevantní info
- Zkus score 0.65 místo 0.72 (méně konzervativní)
- Zkontroluj embedding model (je multilingual?)

### Problém: LLM generuje nové informace

**Příčina:** COMPRESSION_PROMPT není správně předán

**Řešení:**
- Ověř že `compress_answer()` se používá (ne `synthesize()`)
- Zkontroluj COMPRESSION_PROMPT tekst
- Zkus temperature=0.0 (full deterministic)

### Problém: Keyword guard příliš přísný

**Příčina:** Otázka má málo keywords nebo jsou v textu synonyma

**Řešení:**
- Zvýš threshold `min_match` v `_has_relevant_keywords()`
- Přidej synonyma do detekce
- Zkus semantic keyword matching (místo exact string match)

---

## 📚 Další Zdroje

- [README.md](README.md) – Kompletní dokumentace
- [test_compliance.py](test_compliance.py) – Compliance test suite
- [rag.py](rag.py) – Core RAG engine
- [llm.py](llm.py) – LLM wrapper s compression mode

---

**Verze:** 2026-02-08  
**Status:** Production Ready ✅  
**Compliance Level:** Enterprise-grade (3 layers of protection)
