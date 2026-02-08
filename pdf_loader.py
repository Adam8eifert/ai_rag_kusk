"""PDF extrakce: jednotný interface pro čtení PDF a získávání textu.

Používá `pdfplumber` library, která poskytuje:
- Přesné extrakce textu ze stránek
- Metadata (číslo stránky, zdroj)
- Podporu pro tabelární data (není zde použito)
"""

import pdfplumber
from typing import List, Dict


def load_pdf(path: str) -> List[Dict]:
    """Načte PDF soubor a vrátí seznam stránek jako 'dokumenty'.
    
    Funkce procházíme všechny stránky PDF a extrahuje text.
    Pokud je text v OCR bloku (ne-textový PDF), `pdfplumber.extract_text()` vrátí "".
    
    Args:
        path: Cesta k PDF souboru (relativní nebo absolutní)
    
    Returns:
        Seznam záznamů, kde každý záznam představuje JEDNU STRÁNKU:
        [
            {
                "text": "Obsah stránky 1...",
                "page": 1,
                "source": "/cesta/k/dokumentu.pdf"
            },
            {
                "text": "Obsah stránky 2...",
                "page": 2,
                "source": "/cesta/k/dokumentu.pdf"
            },
            ...
        ]
    
    Poznámka:
    - `page` je 1-based (první stránka = 1, ne 0)
    - Pokud stránka je obrázek bez OCR, vrátí se prázdný string ""
    - Tento seznam se později předává RAGEngine.build_index() k chunkování
    """
    documents = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Extrakce textu ze stránky (vrátí string, nebo "" pokud je obrázek)
            text = page.extract_text() or ""
            
            # Jeden záznam per stránku
            documents.append({
                "text": text,
                "page": page_number,
                "source": path
            })

    return documents
