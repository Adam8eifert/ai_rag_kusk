import pdfplumber
from typing import List, Dict


def load_pdf(path: str) -> List[Dict]:
    """Načte PDF soubor a vrátí seznam stránek s textem.

    Každý záznam obsahuje pole `text`, `page` (1-based) a `source` (cesta).
    """
    documents = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            documents.append({
                "text": text,
                "page": page_number,
                "source": path
            })

    return documents
