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


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
        """Rozdělí text na překrývající se chunky podle slov.
        """
        words = text.split()
        if not words:
            return []
        step = max(1, chunk_size - overlap)
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]
        return [c for c in chunks if c.strip()]

    def embed(self, texts: List[str]) -> np.ndarray:
        """Vypočítá embeddingy a vrátí `np.ndarray` float32.
        """
        vecs = self.model.encode(texts, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def build_index(self, data_dir: str = "data", index_dir: str = "index", chunk_size: int = 200, overlap: int = 50, batch_size: int = 64):
        """Vytvoří FAISS index z PDF souborů v `data_dir`.

        Uloží `index/faiss.index` a `index/documents.json` s metadata o chunkech.
        """
        from pdf_loader import load_pdf

        data_dir = Path(data_dir)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        chunks = []

        for pdf_file in data_dir.glob("*.pdf"):
            pages = load_pdf(str(pdf_file))
            for page in pages:
                page_text = page.get("text", "")
                page_chunks = self.chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                for cid, chunk in enumerate(page_chunks, start=1):
                    chunks.append({
                        "file": pdf_file.name,
                        "page": page.get("page"),
                        "chunk_id": cid,
                        "text": chunk,
                    })

        if not chunks:
            print("Žádné textové chunky k indexování.")
            return

        texts = [c['text'] for c in chunks]

        # Embeddingy po dávkách
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.embed(batch)
            all_embs.append(emb)
        embeddings = np.vstack(all_embs).astype(np.float32)

        # Normalizace (pro IndexFlatIP)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, str(index_dir / "faiss.index"))
        with open(index_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"✓ Index uložen: {index_dir / 'faiss.index'}")
        print(f"✓ Metadata uložena: {index_dir / 'documents.json'}")

    def load_index(self, index_dir: str = "index"):
        """Načte existující FAISS index a metadata.
        """
        idx_path = Path(index_dir) / "faiss.index"
        meta_path = Path(index_dir) / "documents.json"
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Index nebo metadata chybí. Spusťte build_index.py")

        self.index = faiss.read_index(str(idx_path))
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.index_dir = Path(index_dir)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Vrátí top-k výsledků s poli `score` (kosinusová podobnost) a metadaty.
        """
        if self.index is None or not self.metadata:
            return []

        q_emb = self.embed([query]).astype(np.float32)
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q_emb = q_emb / q_norm

        distances, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({"score": float(score), **self.metadata[idx]})

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def answer_question(self, question: str, retrieved: List[Dict[str, Any]], strict: bool = False, threshold: float = 0.45, top_k_texts: int = 3) -> Dict[str, Any]:
        """Aplikační logika odpovědí bez externího LLM.

        - Pokud nejsou retrieved: vrací `Nemám odpověď v dokumentech`.
        - Pokud max_similarity < threshold: v `strict` režimu vrací `Nevím`, jinak `Otázka se netýká obsahu dokumentů.`
        - Jinak vrátí konkatenaci nejrelevantnějších chunků jako odpověď.
        """
        if not retrieved:
            return {"answer": "Nemám odpověď v dokumentech", "sources": [], "confidence": 0.0}

        max_sim = max(r.get('score', 0.0) for r in retrieved)
        if max_sim < threshold:
            if strict:
                return {"answer": "Nevím", "sources": [], "confidence": float(max_sim)}
            return {"answer": "Otázka se netýká obsahu dokumentů.", "sources": [], "confidence": float(max_sim)}

        # Sestavíme odpověď jako spojení top-N chunků (bez halucinací)
        top = retrieved[:top_k_texts]
        answer_text = "\n\n---\n\n".join([t.get('text', '') for t in top])
        sources = [{"file": t.get('file'), "page": t.get('page'), "chunk_id": t.get('chunk_id')} for t in top]

        # Ořez pro bezpečnost
        if len(answer_text) > 2000:
            answer_text = answer_text[:2000] + "..."

        return {"answer": answer_text, "sources": sources, "confidence": float(max_sim)}


if __name__ == '__main__':
    print('This module provides RAGEngine. Use build_index.py to create an index.')
