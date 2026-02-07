#!/usr/bin/env python3
"""
Quick test skript: načte index a zkoušej LLM syntetizaci na několika otázkách.

Spuštění:
    python test_llm_synthesis.py
"""

import json
from rag import RAGEngine

# Otázky z test_questions.json
TEST_QUESTIONS = [
    {
        "question": "Jaká je doba plnění uvedená ve smlouvě?",
        "category": "doba_plneni",
    },
    {
        "question": "Jaké sankce jsou uvedeny v případě porušení smlouvy?",
        "category": "sankce",
    },
    {
        "question": "Jakým způsobem může být smlouva ukončena?",
        "category": "ukonceni",
    },
]


def main():
    # 1️⃣ Inicializace RAG engine
    engine = RAGEngine()
    try:
        engine.load_index("index")
        print("✓ Index načten.\n")
    except Exception as e:
        print(f"✗ Chyba při načtení indexu: {e}")
        return

    # 2️⃣ Test na každé otázce
    for q_obj in TEST_QUESTIONS:
        question = q_obj["question"]
        print(f"\n{'='*80}")
        print(f"OTÁZKA: {question}")
        print('='*80)

        # Retrieve
        retrieved = engine.retrieve(question, k=5)
        print(f"\nHledáno: {len(retrieved)} chunků z FAISS")
        for i, r in enumerate(retrieved, 1):
            print(f"  [{i}] score={r.get('score', 0):.3f} | {r.get('file')} str.{r.get('page')}")

        # Synthesis
        result = engine.synthesize_answer(question, retrieved, use_llm=True)
        
        print(f"\n📝 ODPOVĚĎ (Confidence: {result.get('confidence', 0)}):")
        print(f"{result.get('answer', '(prázdná)')}")
        
        print(f"\nZDROJE: {len(result.get('sources', []))} dokumentů")
        for src in result.get('sources', []):
            print(f"  - {src.get('file')} str.{src.get('page')}")


if __name__ == "__main__":
    main()
