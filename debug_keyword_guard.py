#!/usr/bin/env python3
"""
Debug script pro Keyword Guard - podrobná analýza co se děje.
"""

import sys
sys.path.insert(0, '/home/adam/Dokumenty/projects/ai_rag_kusk')

from rag import RAGEngine
import json

# Inicializuj engine
print("\n" + "="*70)
print("LOADING RAG ENGINE")
print("="*70)
engine = RAGEngine(use_openai=False)

# Testovací otázky
test_questions = [
    "Jaká je doba plnění?",
    "Jaká je výpovědní lhůta?",
    "Za jakých podmínek může být smlouva vypovězena?",
    "Jakým dnem začíná běžet záruční doba?",
]

for question in test_questions:
    print(f"\n{'━'*70}")
    print(f"OTÁZKA: {question}")
    print(f"{'━'*70}")
    
    # 1. Retrieve
    retrieved = engine.retrieve_relevant_chunks(question, top_k=3)
    
    if not retrieved:
        print("❌ Žádné dokumenty!")
        continue
    
    top_chunk = retrieved[0]
    top_score = top_chunk.get("score", 0.0)
    
    print(f"\n📊 FAISS SCORE: {top_score:.3f}")
    print(f"   Threshold: 0.72")
    print(f"   Status: {'✅ PASS' if top_score >= 0.72 else '❌ FAIL (< 0.72)'}")
    
    # 2. Context
    context = "\n---\n".join([c["text"] for c in retrieved])
    print(f"\n📄 CONTEXT (first 200 chars):")
    print(f"   {context[:200]}...")
    
    # 3. Keyword Guard
    print(f"\n🔍 KEYWORD GUARD ANALYSIS:")
    
    # Extrahuj keywords (dari rag.py logiku)
    STOP_WORDS = {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "be",
        "je", "jsou", "se", "za", "do", "na", "v", "ve", "z", "ze", "od", "s", "se", "při",
        "by", "bylo", "bylo by", "by měl", "by měla", "být", "bude", "budou", "měl", "měla",
        "měli", "měly", "by byl", "by byla"
    }
    
    words = [w.lower() for w in question.split() if len(w) > 3]
    keywords = [w for w in words if w not in STOP_WORDS]
    
    print(f"   Všechna slova: {words}")
    print(f"   Keywords (bez stop-slov): {keywords}")
    
    context_lower = context.lower()
    matches = []
    
    for kw in keywords:
        if kw in context_lower:
            matches.append(kw)
            print(f"   ✅ '{kw}' nalezeno v kontextu")
        else:
            print(f"   ❌ '{kw}' NENALEZENO v kontextu")
    
    print(f"\n   VÝSLEDEK: {len(matches)} z {len(keywords)} keywords nalezeno")
    print(f"   Threshold: >= 2")
    print(f"   Status: {'✅ PASS' if len(matches) >= 2 else '❌ FAIL (< 2)'}")
    
    # 4. Final decision
    print(f"\n🎯 FINAL DECISION:")
    
    if top_score < 0.72:
        print(f"   ❌ Hard gate BLOCKED (score {top_score:.3f} < 0.72)")
        print(f"      → FALLBACK (keyword guard se nevolá)")
    elif len(matches) < 2:
        print(f"   ❌ Keyword guard BLOCKED ({len(matches)} < 2)")
        print(f"      → FALLBACK")
    else:
        print(f"   ✅ OBĚ brány PROŠLY")
        print(f"      → LLM se BUDE volat")

print(f"\n{'='*70}")
print("DEBUG KONEC")
print(f"{'='*70}\n")
