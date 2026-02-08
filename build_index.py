"""Entry point: Vytvoření FAISS indexu z PDF souborů.

Spusťte tento skript JEDNOU, aby se inicializoval index:
    python build_index.py

Očekávaná vstupní struktura:
    data/
    ├── dokument1.pdf
    ├── dokument2.pdf
    └── ...

Výstup:
    index/
    ├── faiss.index (FAISS IndexFlatIP - vektorový index)
    └── documents.json (metadata: file, page, chunk_id, text)

Po spuštění lze spustit aplikaci:
    uvicorn app:app --reload
"""

from rag import RAGEngine


def main():
    """Hlavní funkce: inicializace RAGEngine a build indexu.
    
    Proces:
    1. Inicializuje RAGEngine (načítá sentence-transformers model)
    2. Najde všechny *.pdf soubory v adresáři 'data/'
    3. Extrahuje text z každé stránky (pdf_loader.load_pdf)
    4. Rozdělí text na chunky (overlap=50 slov, chunk_size=200)
    5. Vypočítá embeddingy (SentenceTransformer encoder)
    6. Normalizuje vektory (IndexFlatIP vyžaduje normalizaci)
    7. Vytvoří FAISS IndexFlatIP a uloží:
       - index/faiss.index (vektorový index)
       - index/documents.json (metadata)
    """
    print("🚀 Inicializace RAGEngine...")
    engine = RAGEngine()
    
    print("📁 Budování indexu z PDF souborů v 'data/' adresáři...")
    engine.build_index(data_dir='data', index_dir='index')
    
    print("✅ Index byl úspěšně vytvořen!")
    print("   - Vektory: index/faiss.index")
    print("   - Metadata: index/documents.json")
    print("\n💡 Nyní můžete spustit aplikaci:")
    print("   uvicorn app:app --reload")


if __name__ == '__main__':
    main()
