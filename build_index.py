import os
import json
import faiss
import numpy as np
from tqdm import tqdm

from pdf_loader import load_pdf
from rag import RAGIndexer


DATA_DIR = "data"
INDEX_DIR = "index"


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    rag = RAGIndexer()
    all_chunks = []
    metadata = []

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(DATA_DIR, file)
        pages = load_pdf(path)

        for page in pages:
            chunks = rag.chunk_text(page["text"])
            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "source": file,
                    "page": page["page"]
                })
                if not all_chunks:
                    print("❌ Nebyly nalezeny žádné textové části k indexaci.")
                    print("➡️ Zkontrolujte, že složka data/ obsahuje PDF soubory.")
                    return

    embeddings = rag.embed(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

    with open(f"{INDEX_DIR}/documents.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"text": t, **m} for t, m in zip(all_chunks, metadata)],
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Index vytvořen, počet chunků: {len(all_chunks)}")


if __name__ == "__main__":
    main()
