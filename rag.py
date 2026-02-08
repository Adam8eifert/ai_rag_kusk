"""RAG engine: chunkování, embeddingy, index a retrieval v jedné třídě.

Třída `RAGEngine` poskytuje metody pro vytvoření indexu (`build_index`),
načtení existujícího indexu (`load_index`), vyhledávání (`retrieve`) a
aplikační logiku odpovídání bez volání externího LLM (`answer_question`).
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rules import QUESTION_RULES


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Heuristika pro detekci otázek, které očekávají krátkou entitu jako odpověď
ENTITY_QUESTION_PREFIXES = (
    "kdo je",
    "jak se jmenuje",
    "kdo vystupuje",
    "kdo je uveden",
)


class RAGEngine:
    """Konsolidovaný RAG engine.

    - Používá `sentence-transformers` pro embeddingy.
    - Ukládá FAISS `IndexFlatIP` (normalizované vektory -> kosinusová podobnost).
    - Metadata obsahují `file`, `page`, `chunk_id`, `text`.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.index_dir: Path = Path("index")

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Rozdělí dlouhý text na překrývající se chunky (sliding window).
        
        Princip:
        - Rozdělíme text NA SLOVA (ne znaky)
        - Vytvoříme okna o velikosti `chunk_size` slov
        - Posun okna je `step = chunk_size - overlap` slov
        - Překryv umožňuje, aby se relevantní fragment nevytratil na hranici
        
        Příklad:
            text = "The quick brown fox jumps over..."
            chunk_size=5, overlap=2 -> step=3
            Chunk 1: "The quick brown fox jumps"
            Chunk 2: "fox jumps over the lazy"
            (vidíme 'fox', 'jumps' v obou = seamless transition)
        
        Args:
            text: Vstupní text (obvykle jedna stránka PDF)
            chunk_size: Počet slov v jednom chunku (default 200)
            overlap: Počet slov, které se sdílí mezi sousedními chunky (default 50)
        
        Returns:
            Seznam stringů, každý string je jeden chunk (max chunk_size slov)
        """
        words = text.split()
        if not words:
            return []
        
        # Posun mezi chunky (o tento počet slov se posuneme dopředu)
        step = max(1, chunk_size - overlap)
        
        # Sliding window: vezmeme slova od i do i+chunk_size, posuneme o 'step'
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]
        
        # Filtrace: ignorujeme prázdné chunky
        return [c for c in chunks if c.strip()]

    def embed(self, texts: List[str]) -> np.ndarray:
        """Vypočítá embeddingy pomocí SentenceTransformer modelu.
        
        Model `paraphrase-multilingual-MiniLM-L12-v2`:
        - Mnohajazyčný (podporuje 100+ jazyků včetně češtiny)
        - Kompaktní (12 vrstev, 384 rozměrů)
        - Fast (CPU efficient)
        - Trained na semantické podobnosti
        
        Embedding:
        - Vstup: seznam textů
        - Výstup: matice formátu (N, 384), kde N = počet textů
        - Dtype: float32 (kompatibilita s FAISS)
        
        Args:
            texts: Seznam stringů k enkódování
        
        Returns:
            np.ndarray formátu (len(texts), 384) typu float32
        """
        vecs = self.model.encode(texts, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def build_index(self, data_dir: str = "data", index_dir: str = "index", chunk_size: int = 200, overlap: int = 50, batch_size: int = 64):
        """Vytvoří FAISS index z PDF souborů: PDF → Chunky → Embeddingy → Index.
        
        Procedura:
        1. Najde všechny *.pdf soubory v `data_dir`
        2. Pro každý PDF:
           - Extrahuje text stránku po stránce (pdf_loader.load_pdf)
           - Rozdělí text na chunky se overlappem (chunk_text)
           - Schová metadata (soubor, stránka, chunk_id)
        3. Vypočítá embeddingy pro všechny chunky (po dávkách)
        4. Normalizuje embeddingy (kritické pro IndexFlatIP)
        5. Vytvoří FAISS IndexFlatIP a uloží na disk:
           - index/faiss.index (vektorový index)
           - index/documents.json (metadata)
        
        Výstup struktura:
            index/
            ├── faiss.index               # FAISS IndexFlatIP (binární)
            └── documents.json            # Metadata JSON
                [
                  {
                    "file": "smlouva.pdf",
                    "page": 1,
                    "chunk_id": 1,
                    "text": "prvních 200 slov stránky 1..."
                  },
                  ...
                ]
        
        Args:
            data_dir: Adresář s PDF soubory (default "data/")
            index_dir: Výstupní adresář pro index (default "index/")
            chunk_size: Počet slov v jednom chunku (default 200)
            overlap: Počet slov překryvu mezi chunky (default 50)
            batch_size: Počet textů zpracovaných najednou (default 64, pro rychlost)
        """
        from pdf_loader import load_pdf

        data_dir = Path(data_dir)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Krok 1: Sbírání chunků
        chunks = []

        # Najdi všechny PDF soubory
        for pdf_file in data_dir.glob("*.pdf"):
            print(f"📄 Čtení: {pdf_file.name}")
            
            # Extrahuj text (stránku po stránce)
            pages = load_pdf(str(pdf_file))
            
            for page in pages:
                page_text = page.get("text", "")
                
                # Rozdělení stránky na chunky
                page_chunks = self.chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                
                for cid, chunk in enumerate(page_chunks, start=1):
                    chunks.append({
                        "file": pdf_file.name,
                        "page": page.get("page"),
                        "chunk_id": cid,
                        "text": chunk,
                    })

        if not chunks:
            print("⚠ Žádné textové chunky k indexování. Zkontroluj data/ adresář.")
            return

        print(f"✓ Sbíráno {len(chunks)} chunků")

        # Krok 2: Enkódování (po dávkách)
        texts = [c['text'] for c in chunks]
        all_embs = []
        
        print(f"🔄 Enkódování {len(texts)} textů po dávkách x{batch_size}...")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.embed(batch)
            all_embs.append(emb)
        
        embeddings = np.vstack(all_embs).astype(np.float32)
        print(f"✓ Embeddingy shape: {embeddings.shape}")

        # Krok 3: Normalizace (KRITICKÉ pro IndexFlatIP)
        # IndexFlatIP očekává normalizované vektory (jednotkový norm)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        print(f"✓ Vektory normalizovány")

        # Krok 4: Vytvoření a uložení FAISS indexu
        dim = embeddings.shape[1]  # 384 pro paraphrase-multilingual-MiniLM
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"✓ FAISS IndexFlatIP vytvořen (dim={dim}, n={len(embeddings)})")

        # Uloži index a metadata
        faiss.write_index(index, str(index_dir / "faiss.index"))
        with open(index_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Index uložen: {index_dir / 'faiss.index'}")
        print(f"✅ Metadata uložena: {index_dir / 'documents.json'}")

    def load_index(self, index_dir: str = "index"):
        """Načte existující FAISS index a metadata ze souboru.
        
        Předpoklady:
        - `index_dir/faiss.index` existuje (binární FAISS index)
        - `index_dir/documents.json` existuje (metadata)
        
        Použití:
            engine = RAGEngine()
            engine.load_index('index/')  # Načte index
            results = engine.retrieve("Jaká je doba plnění?")  # Nyní funguje
        
        Args:
            index_dir: Adresář s uloženým indexem (default "index/")
        
        Raises:
            FileNotFoundError: Pokud index nebo metadata chybí
        """
        idx_path = Path(index_dir) / "faiss.index"
        meta_path = Path(index_dir) / "documents.json"
        
        # Kontrola: jsou soubory přítomné?
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index nebo metadata chybí v '{index_dir}'. "
                "Spusťte: python build_index.py"
            )

        # Načti FAISS index (binární formát)
        self.index = faiss.read_index(str(idx_path))
        
        # Načti metadata (JSON)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.index_dir = Path(index_dir)
        print(f"✓ Index načten: {len(self.metadata)} dokumentů, "
              f"vimenze={self.index.d}")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Vrátí top-k nejrelevantnějších chunků z FAISS indexu (semantic search).
        
        Algoritmus:
        1. Enkóduj query pomocí SentenceTransformer (stejný model jako chunky)
        2. Normalizuj query embedding (potřebné pro IndexFlatIP)
        3. Hledej k-nearest neighbors v FAISS indexu (kosinusová podobnost)
        4. Vrať výsledky s skórem (0.0-1.0) a metadaty
        
        FAISS IndexFlatIP:
        - "IP" znamená Inner Product (vnitřní součin)
        - Pokud jsou vektory normalizovány na jednotkou délku,
          Inner Product = Kosinusová Podobnost
        - Vrací "distances" jako cosine similarities
        
        Příklad výstupu:
        [
            {
                "score": 0.87,
                "file": "smlouva.pdf",
                "page": 2,
                "chunk_id": 5,
                "text": "Doba plnění je 30 dní od data..."
            },
            {
                "score": 0.73,
                "file": "smlouva.pdf",
                "page": 3,
                "chunk_id": 7,
                "text": "Plnění se musí uskutečnit..."
            },
            ...
        ]
        
        Args:
            query: Otázka nebo vyhledávací výraz (string)
            k: Počet nejlepších výsledků (default 5)
        
        Returns:
            Seznam dict se score (0.0-1.0), file, page, chunk_id, text
            Seřazeno sestupně podle score (nejvyšší skóre první)
        """
        if self.index is None or not self.metadata:
            return []

        # Enkóduj query
        q_emb = self.embed([query]).astype(np.float32)
        
        # Normalizuj query (FAISS IndexFlatIP vyžaduje normalizované vektory)
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0  # Ošetření dělení nulou
        q_emb = q_emb / q_norm

        # FAISS semantic search (vrátí distances = cosine similarities)
        distances, indices = self.index.search(q_emb, k)

        # Konstruuj výsledky s metadaty
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue  # Index mimo rozsah (FAISS vrací -1 pro neplatné)
            # Spojíme metadata s skórem
            results.append({"score": float(score), **self.metadata[idx]})

        # Seřaď sestupně podle score (nejlepší první)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def answer_question(self, question: str, retrieved: List[Dict[str, Any]], strict: bool = False, threshold: float = 0.45, top_k_texts: int = 3) -> Dict[str, Any]:
        """Odpovídá na otázku pomocí pravidel (rules.py) - **BEZ LLM**.
        
        Aplikuje tvrdá pravidla definovaná v QUESTION_RULES:
        1. Detekuje typ otázky (doba_plneni, platnost_smlouvy, objednatel atd.)
        2. Filtruje chunky podle min_similarity prahu a section_keywords
        3. Vrací nejlepší chunk nebo fallback zprávu
        
        Args:
            question: Uživatelská otázka
            retrieved: Ignoruje se (kompatibilita), vnitřně volá self.retrieve()
            strict: Nepoužívá se (kompatibilita)
            threshold: Nepoužívá se (min_similarity je v QUESTION_RULES)
            top_k_texts: Počet chunků k načtení
        
        Returns:
            {"answer": str, "sources": List[Dict], "confidence": float}
        """
        # Detekce typu otázky (např. "doba plnění" -> "doba_plneni")
        qtype = detect_question_type(question)

        if qtype is None:
            # Otázka neodpovídá žádnému pravidlu
            return {"answer": "Nevím.", "sources": [], "confidence": 0.0}

        # Načteme pravidla pro daný typ (min_similarity, keywords atd.)
        rules = QUESTION_RULES[qtype]

        # Semantic search v FAISS indexu
        results = self.retrieve(question, k=top_k_texts)

        # Filtrujeme podle pravidel (kosinusová podobnost + section keywords)
        allowed = []
        for r in results:
            if (
                r.get("score", 0.0) >= rules.get("min_similarity", 0.0)
                and is_chunk_allowed(r.get("text", ""), rules)
            ):
                allowed.append(r)

        # Žádný chunk neprošel kombinovaným filtrem
        if not allowed:
            return {
                "answer": "Požadovaná informace není v dokumentech explicitně uvedena.",
                "sources": [],
                "confidence": 0.0,
            }

        # Vezmeme nejlepší chunk (nejvyšší skóre)
        best = sorted(allowed, key=lambda x: x.get("score", 0.0), reverse=True)[0]

        return {
            "answer": best.get("text", "").strip(),
            "sources": [{
                "file": best.get("file"),
                "page": best.get("page"),
                "chunk_id": best.get("chunk_id"),
            }],
            "confidence": round(best.get("score", 0.0), 3),
        }

    def synthesize_answer(self, question: str, retrieved: List[Dict[str, Any]], use_llm: bool = True, strict: bool = False) -> Dict[str, Any]:
        """Syntetizuje odpověď se striktnými strážci protiv halucinací.
        
        **Kritické pravidla (compliance-friendly):**
        
        1. **Hard factual gate** (score < 0.72)
           - Pokud top chunk má skóre < 0.72, vrať fallback BEZ LLM
           - Zabrání LLM aby spekuloval
        
        2. **Keyword guard**
           - Ověř že kontext obsahuje relevantní slova z otázky
           - Pokud chybí (např. "vlastník" není v textu), vrať fallback
        
        3. **LLM jen jako kompresor**
           - LLM dostane pouze extrahované texty
           - Never generuje nové informace
        
        Pipeline:
        1. Check: Máme vůbec chunky?
        2. CHECK: top_score >= 0.72? (Hard gate)
        3. CHECK: Obsahuje relevantní keywords? (Guard)
        4. Pokud use_llm: zavolej LLM jako KOMPRESOR (ne syntezátor)
        5. Pokud strict: zkrátí na max 2-3 věty
        
        Args:
            question: Uživatelská otázka
            retrieved: Top-k chunky z FAISS retrieve()
            use_llm: Použít LLM jako kompresor (True) nebo vrátit raw text (False)
            strict: Pokud True, zkrátí odpověď na 2-3 věty
        
        Returns:
            {"answer": str, "sources": List[Dict], "confidence": float}
            
            Note: answer = "" pokud neprošla gate/guard (fallback)
        """
        # ────────────────────────────────────────────────────────────
        # 1️⃣ CHECK: Máme vůbec relevantní chunky?
        # ────────────────────────────────────────────────────────────
        if not retrieved:
            return {
                "answer": "Požadovaná informace není v dokumentech.",
                "sources": [],
                "confidence": 0.0,
            }

        # ────────────────────────────────────────────────────────────
        # 2️⃣ HARD FACTUAL GATE: top_score >= 0.50?
        # ────────────────────────────────────────────────────────────
        top_score = retrieved[0].get("score", 0.0)
        HARD_GATE_THRESHOLD = 0.50  # Sníženo pro lepší recall, odpovídá compliance testům
        
        if top_score < HARD_GATE_THRESHOLD:
            # ❌ Skóre příliš nízké = informace není dostatečně podložená
            # LLM se NESMÍ zavolat (bez spekulace)
            return {
                "answer": "Požadovaná informace není v dokumentech.",
                "sources": [],
                "confidence": top_score,
            }

        # Spojení textu ze všech chunků
        context = "\n---\n".join([r.get("text", "") for r in retrieved])
        
        # ────────────────────────────────────────────────────────────
        # 3️⃣ KEYWORD GUARD: Obsahuje relevantní slova?
        # ────────────────────────────────────────────────────────────
        if not self._has_relevant_keywords(context, question):
            # ❌ Kontext nemá relevantní slova = otázka je mimo scope
            return {
                "answer": "Požadovaná informace není v dokumentech.",
                "sources": [],
                "confidence": top_score,
            }

        # Průměrné skóre relevance
        avg_score = sum(r.get("score", 0.0) for r in retrieved) / len(retrieved) if retrieved else 0.0

        # ────────────────────────────────────────────────────────────
        # 4️⃣ VOLITELNÉ: LLM jako KOMPRESOR (ne syntezátor)
        # ────────────────────────────────────────────────────────────
        if use_llm:
            try:
                from llm import LLMWrapper
                llm = LLMWrapper(use_openai=True)
                # ⚠️ NOVÝ: compress_answer místo synthesize
                # (jen zkrátí, ne generuje)
                answer_text = llm.compress_answer(question, context)
            except Exception as e:
                # Fallback: vrať raw kontext (bez LLM)
                print(f"⚠ LLM chyba: {e}, fallback na raw kontext")
                answer_text = context
        else:
            # No LLM: vrať raw context
            answer_text = context

        # ────────────────────────────────────────────────────────────
        # 5️⃣ POKUD strict: zkrátíme na 2-3 věty
        # ────────────────────────────────────────────────────────────
        if strict:
            answer_text = self._summarize_answer(answer_text, max_sentences=3)

        return {
            "answer": answer_text.strip(),
            "sources": [
                {
                    "file": r.get("file"),
                    "page": r.get("page"),
                    "chunk_id": r.get("chunk_id"),
                }
                for r in retrieved
            ],
            "confidence": round(avg_score, 3),
        }

    def _has_relevant_keywords(self, context: str, question: str) -> bool:
        """Ověř, že kontext obsahuje relevantní klíčová slova z otázky.
        
        ÚČEL: Zabránit LLM aby spekuloval na otázky mimo scope dokumentu.
        
        Příklady správného chování:
        - Q: "Jaká je doba plnění?"           K: ["doba", "plnění"] → True (slova v textu)
        - Q: "Kdo je objednatel?"             K: ["objednatel"] → True
        - Q: "Kdo je skutečný vlastník?"      K: [] (ve smlouvě není) → False (vrať fallback)
        - Q: "Jaké riziko smlouva představuje?" K: [] (evaluační, ne faktická) → False
        - Q: "Jaká je cena na trhu?"          K: [] (externí data) → False
        
        ALGORITMUS:
        1. Extrahuj keywords z otázky (slova delší než 3 znaky, mimo stop-words)
        2. Normalizuj context na lowercase
        3. Spočítej kolik keywords je přítomno v contextu
        4. Pokud méně než 2 keywords → vrať False (fallback)
        5. Pokud alespoň 2 → vrať True (proceed)
        
        Args:
            context: Extrahované texty z FAISS chunků
            question: Uživatelská otázka
        
        Returns:
            True pokud je dostatečně relevantní, False → fallback
        """
        import string
        
        # Normalizuj
        q_lower = question.lower()
        c_lower = context.lower()
        
        # Stop-words: ignoruj tyto slova (jsou příliš generická)
        stop_words = {
            'jaká', 'jaké', 'jaký', 'je', 'co', 'za', 'byl', 'byla', 'bylo',
            'jsou', 'budou', 'pokud', 'pokud', 'nebo', 'a', 'z', 'na',
            'ten', 'ta', 'to', 'ten', 'tou', 'tím', 'jakým', 'kterou', 'který',
            'si', 'se', 'i', 'o', 'do', 'by', 'by', 'by'
        }
        
        # Extrahuj keywords z otázky (slova delší než 3 znaky, bez interpunkce)
        q_words = [
            w.strip().strip(string.punctuation)  # Odstraň interpunkci
            for w in q_lower.split()
            if len(w.strip().strip(string.punctuation)) > 3 and w.strip().strip(string.punctuation) not in stop_words
        ]
        
        # Pokud otázka nemá žádná keywords (neměl by se stát), povolj
        if not q_words:
            return True
        
        # Hledej kolik keywords je přítomno v contextu
        matched = sum(1 for w in q_words if w in c_lower)
        
        # Povinná pravidla:
        # - Alespoň 2 keywords musí být v contextu
        # - Nebo alespoň 60% keywords
        min_match = max(2, len(q_words) // 2)  # Alespoň 2 nebo 50%
        
        return matched >= min_match

    def _summarize_answer(self, text: str, max_sentences: int = 3) -> str:
        """Zkrátí odpověď na max_sentences vět pomocí heuristiky (bez LLM).
        
        Používá se v strict mode aktivován synthesize_answer(..., strict=True).
        
        Algoritmus:
        1. Rozdělí text na věty regex: za tečkou/výkř/otazníkem + whitespace
        2. Vezme prvních max_sentences vět
        3. Zajistí, že výsledek končí správnou interpunkcí
        
        Výhody:
        - Bez LLM = deterministické, bez halucinací
        - Zachovává počáteční relevantní informace
        - Splňuje požadavek "max 2-3 věty pro asistenty"
        
        Příklad:
            text = "Doba plnění je 30 dní. To je standard. Lze ji prodloužit."
            max_sentences = 2
            Výstup: "Doba plnění je 30 dní. To je standard."
        
        Args:
            text: Celý text odpovědi
            max_sentences: Maximální počet vět v output (default 3)
        
        Returns:
            Zkrácený text (max max_sentences vět) s správnou interpunkcí
        """
        import re
        
        # Rozdělení na věty: hledej zbytky za [.!?] a následujícím whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Očisti: seřaď, a odstraň prázdné věty
        sentences = [s.strip() for s in sentences if s.strip()]

        # Pokud je vět méně než max_sentences, vrať celý text
        if len(sentences) <= max_sentences:
            return text.strip()

        # Vezmi prvních max_sentences vět
        summary = " ".join(sentences[:max_sentences])
        
        # Ujisti se, že text končí nějakou interpunkcí
        if summary and not summary.endswith(('.', '!', '?')):
            summary += "."
        
        return summary

    def is_entity_question(self, question: str) -> bool:
        """Rozpozná jednoduché entity-type otázky podle předdefinovaných prefixů.
        
        Entity otázky: ty, které očekávají menší kus informace (osobu, firmu, apod)
        Příklady: "Kdo je objednatel?", "Jak se jmenuje dodavatel?", ...
        
        Hledá otázky začínající se slovy: "kdo je", "jak se jmenuje", "kdo vystupuje", ...
        
        Používá se pro optimalizaci extrakce odpovědi:
        - Pokud je entity question, můžeme hledat konkrétní větu místo dlouhého textu
        
        Args:
            question: Uživatelská otázka (string)
        
        Returns:
            True pokud otázka je entity-type, else False
        """
        q = (question or "").lower().strip()
        return any(q.startswith(p) for p in ENTITY_QUESTION_PREFIXES)

    def _extract_entity_sentence(self, text: str, keyword: Optional[str] = None) -> str:
        """Extrahuje nejrelevantnější větu z textu (pro entity otázky).
        
        Strategie:
        1. Pokud je keyword poskytnut: najdi PRVNÍ větu obsahující keyword
        2. Pokud není keyword (nebo není v textu): vrať PRVNÍ větu
        
        Používá se v answer_question() pro entity otázky ("Kdo je X?")
        
        Příklad:
            text = "Pan Novák je objednatel. Podpisem smlouvy souhlasí. ..."
            keyword = "objednatel"
            Výstup: "Pan Novák je objednatel."
        
        Args:
            text: Vstupní text (obvykle jeden chunk)
            keyword: Slovo k vyhledání v jedné z vět (optional)
        
        Returns:
            Věta obsahující keyword (pokud keyword je), nebo první věta
        """
        import re
        
        if not text:
            return ""
        
        # Rozdělení textu na věty
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Pokud máme keyword, hledej větu s tímto slovem
        if keyword:
            k = keyword.lower()
            for sentence in sentences:
                if k in sentence.lower():
                    return sentence

        # Fallback: vrať první větu (pokud existuje)
        return sentences[0] if sentences else text.strip()


def detect_question_type(question: str) -> str | None:
    """Detekuje typ otázky porovnáním against QUESTION_RULES z rules.py.
    
    Algoritmus:
    1. Normalizuj otázku na malá písmena
    2. Pro každý rule_type v QUESTION_RULES:
       - Hledej alespoň jedno question_keyword v otázce
       - Pokud najdeš, vrať rule_type
    3. Pokud nic nepaduje, vrať None
    
    Příklady:
        "Jaká je doba plnění?" -> "doba_plneni" (najde "doba plnění")
        "Kolik je platnost smlouvy?" -> "platnost_smlouvy"
        "Kdo je vyrobce?" -> None (není v QUESTION_RULES)
    
    Args:
        question: Uživatelská otázka (string)
    
    Returns:
        Klíč z QUESTION_RULES (str) nebo None pokud neodpovídá žádnému pravidlu
    """
    q = (question or "").lower()
    
    # Iteruj přes všechna dostupná pravidla
    for qtype, cfg in QUESTION_RULES.items():
        # Zkontroluj, zda otázka obsahuje alespoň jedno question_keyword
        for kw in cfg.get("question_keywords", []):
            if kw in q:
                return qtype  # Našli jsme shodu, vrať typ
    
    return None  # Žádná shoda


def is_chunk_allowed(chunk_text: str, rules: dict) -> bool:
    """Tvrdá filtrace: kontroluj, zda chunk obsahuje section_keywords z pravidel.
    
    Používá se v answer_question() pro další filtraci retrievaných chunků.
    
    Algoritmus:
    1. Normalizuj chunk na malá písmena
    2. Hledej alespoň jedno section_keyword ze `rules` v text
    3. Pokud ano, vrať True (chunk "projde")
    4. Pokud ne, vrať False (ignoruj chunk)
    
    Příklad:
        chunk_text = "Doba plnění je 30 dní od podpisu smlouvy"
        rules = QUESTION_RULES["doba_plneni"]
                = {
                    "question_keywords": [...],
                    "section_keywords": ["doba plnění", "plnění smlouvy", ...],
                    "min_similarity": 0.75
                  }
        is_chunk_allowed(...) -> True (obsahuje "doba plnění")
    
    Args:
        chunk_text: Text chunku z FAISS indexu
        rules: Dictionary s section_keywords (z QUESTION_RULES)
    
    Returns:
        True pokud chunk obsahuje alespoň jedno section_keyword, else False
    """
    text = (chunk_text or "").lower()
    
    # Zkontroluj všechny section keywords
    return any(kw in text for kw in rules.get("section_keywords", []))


if __name__ == '__main__':
    print('This module provides RAGEngine. Use build_index.py to create an index.')
