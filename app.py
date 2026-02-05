import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

from llm import LocalLLM

MAX_DISTANCE = 1.2
INDEX_DIR = "index"
LOG_FILE = "logs/queries.jsonl"

os.makedirs("logs", exist_ok=True)

app = FastAPI()

embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
llm = LocalLLM()

index = faiss.read_index(f"{INDEX_DIR}/faiss.index")

with open(f"{INDEX_DIR}/documents.json", encoding="utf-8") as f:
    documents = json.load(f)


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(question: Question):
    query_embedding = embed_model.encode([question.question]).astype("float32")
    distances, indices = index.search(query_embedding, k=5)

    retrieved_chunks = []
    references = []

    for idx in indices[0]:
        doc = documents[idx]
        retrieved_chunks.append(doc["text"])
        references.append({
            "source": doc["source"],
            "page": doc["page"]
        })
       
valid_indices = [
    idx for dist, idx in zip(distances[0], indices[0])
    if dist < MAX_DISTANCE
]

    if not valid_indices:
        return {
        "question": question.question,
        "answer": "V dostupných dokumentech nebyla nalezena relevantní informace.",
        "references": []
    }

    }

    context = "\n\n".join(retrieved_chunks)

    answer = llm.answer(question.question, context)

    response = {
        "question": question.question,
        "answer": answer,
        "references": references
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question.question,
            "answer": answer,
            "references": references
        }, ensure_ascii=False) + "\n")

    return response
