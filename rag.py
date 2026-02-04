from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGIndexer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def chunk_text(self, text: str, chunk_size=500, overlap=100):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
