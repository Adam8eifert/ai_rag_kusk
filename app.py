import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime


INDEX_DIR = "index"
LOG_FILE = "logs/queries.jsonl"

app = FastAPI()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index(f"{INDEX_DIR}/faiss.index")

with open(f"{INDEX_DIR}/documents.json", encoding="utf-8") as f:
    documents = json.load(f)


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(question: Question):
    query_embedding = model.encode([question.question]).astype("float32")
    distances, indices = index.search(query_embedding, k=5)

    answers = []
    for idx in indices[0]:
        doc = documents[idx]
        answers.append({
            "text": doc["text"],
            "source": doc["source"],
            "page": doc["page"]
        })

    result = {
        "question": question.question,
        "answers": answers
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question.question,
            "answers": answers
        }, ensure_ascii=False) + "\n")

    return result
