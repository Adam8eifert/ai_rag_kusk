import pdfplumber
from typing import List, Dict


def load_pdf(path: str) -> List[Dict]:
    documents = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                documents.append({
                    "text": text,
                    "page": page_number,
                    "source": path
                })

    return documents
