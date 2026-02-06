AI RAG Pipeline – Dotazování nad PDF dokumenty (bez halucinací)Tento projekt implementuje deterministický, produkčně realistický RAG (Retrieval-Augmented Generation) systém nad lokálními PDF dokumenty (např. smlouvami, interní dokumentací).Hlavní cíle projektuData Sovereignty: Pracuje výhradně s lokálně dodanými dokumenty.Zero Hallucination: Architektura navržena tak, aby AI nemohla lhát.Auditovatelnost: Každá odpověď je přímo podložená zdrojem (soubor + strana).Enterprise Ready: Bezpečné pro citlivá data a interní compliance.[!IMPORTANT]Projekt záměrně nepoužívá generativní LLM modely při odpovídání. Jedná se o tzv. extractive RAG, který vrací přesné citace místo generovaného textu.🛠 Co projekt dělá a co NENÍCo projekt děláCo projekt NENÍ✅ Extrahuje text z PDF (pdfplumber)❌ Chatbot pro volný pokec✅ Tvoří sémantické embeddingy❌ Generativní AI (nepíše básně)✅ Ukládá vektorový index (FAISS)❌ Model trénovaný na externích datech✅ Odpovídá výhradně z obsahu dokumentů❌ Systém pro "open-ended" otázky🏗 Architektura systémuFragment kódugraph TD
    PDF[PDF Data] --> EXT[Extrakce textu]
    EXT --> CHUNK[Chunking + Embedding]
    CHUNK --> FAISS[(FAISS Vector Index)]
    QUERY[Uživatel /ask] --> SEARCH[Similarity Search]
    SEARCH --> FAISS
    FAISS --> ANS[Extraktivní odpověď + zdroje]
📁 Struktura projektuPlaintext.
├── data/               # Vstupní PDF dokumenty
├── index/              # FAISS index + metadata
├── build_index.py      # Script pro indexaci PDF
├── rag.py              # RAGEngine (logika vyhledávání)
├── app.py              # FastAPI API endpointy
├── llm.py              # Architektonický placeholder
├── requirements.txt    # Závislosti
└── README.md           # Dokumentace
📄 Popis komponentbuild_index.pyZpracovává surová data. Rozdělí text na menší bloky (chunky), vytvoří embeddingy a uloží je do FAISS indexu společně s metadaty (název souboru, strana).Poznámka: Indexaci je nutné spustit znovu při každém přidání dokumentu.rag.py (Srdce systému)Obsahuje třídu RAGEngine. Při dotazu vyhledá nejrelevantnější úryvky textu.Strict Mode: Pokud je zapnutý, systém vrací odpovědi pouze při vysoké shodě.Confidence: Vrací míru podobnosti (cosine similarity).llm.pySlouží jako placeholder. Projekt je navržen tak, aby fungoval bez LLM, ale tato vrstva umožňuje budoucí napojení generativního modelu bez nutnosti měnit logiku vyhledávání.🚀 Instalace a spuštění1. Příprava prostředíBash# Vytvoření virtuálního prostředí
python -m venv .venv
source .venv/bin/activate # (Na Windows: .venv\Scripts\activate)

# Instalace závislostí
pip install -r requirements.txt
2. Indexace datVložte PDF do složky /data a spusťte:Bashpython build_index.py
3. Spuštění APIBashuvicorn app:app --reload
API: http://127.0.0.1:8000Dokumentace (Swagger): http://127.0.0.1:8000/docs🔌 Použití APIPOST /askRequest:JSON{
  "question": "Jaká je doba platnosti smlouvy?",
  "strict": true
}
Response:JSON{
  "answer": "Smlouva je uzavřena na dobu určitou do 31. 12. 2025.",
  "sources": [
    { "file": "smlouva_ABC.pdf", "page": 3, "chunk_id": 1 }
  ],
  "confidence": 0.87
}
💡 Poznámky k návrhuBezpečí: Žádné scrapování webu ani odesílání dat do cloudu.Model: Používá paraphrase-multilingual-MiniLM-L12-v2 (skvělý pro češtinu).Vhodné pro: Právní smlouvy, compliance, interní firemní směrnice.Autor: Adam Seifert