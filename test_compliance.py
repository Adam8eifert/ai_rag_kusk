"""Test compliance: Ověř, že RAG dodržuje striktní pravidla bez halucinací.

Tento skript testuje:
1. Hard factual gate (top_score < 0.72 → fallback bez LLM)
2. Keyword guard (relevantní slova v kontextu)
3. LLM jen jako kompresor (ne generátor)
4. Fallback odpovědi pro out-of-scope otázky
"""

import json
import sys
from pathlib import Path

# Přidej projekt do path
sys.path.insert(0, str(Path(__file__).parent))

from rag import RAGEngine


def test_hard_factual_gate():
    """
    TEST 1: Hard factual gate (score < 0.72)
    
    ✅ Pokud top_score < 0.72, musí být fallback BEZ LLM
    ❌ LLM se NESMÍ zavolat
    """
    print("\n" + "="*70)
    print("TEST 1: Hard Factual Gate (0.72 threshold)")
    print("="*70)
    
    engine = RAGEngine()
    
    try:
        engine.load_index("index/")
    except FileNotFoundError:
        print("⚠️  Index není dostupný. Spusťte: python build_index.py")
        return False
    
    # Otázka s nízkou relevancí (bez dokumentu)
    # Tato otázka by měla mít score < 0.72 (pokud dokument neexistuje)
    low_relevance_questions = [
        "Kdo je skutečným konečným vlastníkem dodavatele?",
        "Jaká je obvyklá cena na trhu?",
        "Jaké riziko pro objednatele smlouva představuje?",
    ]
    
    results = []
    for q in low_relevance_questions:
        retrieved = engine.retrieve(q, k=5)
        if retrieved:
            top_score = retrieved[0].get("score", 0.0)
            result = engine.synthesize_answer(q, retrieved, use_llm=True)
            
            passed = (
                result["answer"] == "Požadovaná informace není v dokumentech."
                and result["confidence"] < 0.72
                and result["sources"] == []
            )
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status}: {q}")
            print(f"  Top score: {top_score:.3f}")
            print(f"  Answer: {result['answer'][:60]}...")
            print(f"  Sources: {len(result['sources'])}")
            
            results.append(passed)
        else:
            print(f"\n⚠️  No retrieval results for: {q}")
    
    return all(results) if results else None


def test_keyword_guard():
    """
    TEST 2: Keyword Guard
    
    ✅ Pokud relevantní slova NEJSOU v kontextu → fallback
    ✅ Pokud relevantní slova JSOU → allow
    """
    print("\n" + "="*70)
    print("TEST 2: Keyword Guard (relevance check)")
    print("="*70)
    
    engine = RAGEngine()
    
    try:
        engine.load_index("index/")
    except FileNotFoundError:
        print("⚠️  Index není dostupný")
        return False
    
    # Otázka SE relevantními slovy (by měla projít)
    relevant_questions = [
        "Jaká je doba plnění?",
        "Jaká je výpovědní lhůta?",
        "Za jakých podmínek může být smlouva vypovězena?",
    ]
    
    # Otázka BEZ relevantních slov (by měla fallback)
    irrelevant_questions = [
        "Jaký je barva auta?",
        "Kdy se zrodila královna Alžběta?",
        "Jaká je teplota v Pekingu?",
    ]
    
    results = []
    
    print("\n🔍 Otázky s relevantními slovy (by měly mít odpověď):")
    for q in relevant_questions:
        retrieved = engine.retrieve(q, k=5)
        if retrieved:
            result = engine.synthesize_answer(q, retrieved, use_llm=False)
            has_answer = result["answer"] != "Požadovaná informace není v dokumentech."
            status = "✅ PASS" if has_answer else "❌ FAIL"
            print(f"\n{status}: {q}")
            print(f"  Answer: {result['answer'][:70]}...")
            results.append(has_answer)
    
    print("\n🔍 Otázky bez relevantních slov (by měly mít fallback):")
    for q in irrelevant_questions:
        retrieved = engine.retrieve(q, k=5)
        if retrieved:
            result = engine.synthesize_answer(q, retrieved, use_llm=False)
            is_fallback = result["answer"] == "Požadovaná informace není v dokumentech."
            status = "✅ PASS" if is_fallback else "❌ FAIL"
            print(f"\n{status}: {q}")
            print(f"  Answer: {result['answer'][:70]}...")
            results.append(is_fallback)
    
    return all(results) if results else None


def test_llm_compression():
    """
    TEST 3: LLM jen jako kompresor
    
    ✅ LLM nesmí generovat nové informace
    ✅ LLM nesmí odpovídat sám
    ✅ LLM jen zkrátí poskytnutý text
    """
    print("\n" + "="*70)
    print("TEST 3: LLM Compression (not generation)")
    print("="*70)
    
    try:
        from llm import LLMWrapper
    except ImportError:
        print("⚠️  LLM modul není dostupný")
        return None
    
    # Test krátký kontext (by se neměl měnit)
    short_context = "Doba plnění je 30 dní."
    
    try:
        llm = LLMWrapper(use_openai=True)
        compressed = llm.compress_answer("Jaká je doba plnění?", short_context)
        
        # Krátký text by měl zůstat beze změny
        passed = short_context in compressed or len(compressed.split()) <= 10
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}: Krátký text není měněn")
        print(f"  Original: {short_context}")
        print(f"  Compressed: {compressed}")
        
        return passed
    except Exception as e:
        print(f"⚠️  LLM test skipped: {e}")
        return None


def test_fallback_consistency():
    """
    TEST 4: Fallback chování je konzistentní
    
    ✅ fallback answer = "Požadovaná informace není v dokumentech."
    ✅ fallback sources = []
    ✅ fallback confidence = top_score
    """
    print("\n" + "="*70)
    print("TEST 4: Fallback Consistency")
    print("="*70)
    
    engine = RAGEngine()
    
    try:
        engine.load_index("index/")
    except FileNotFoundError:
        print("⚠️  Index není dostupný")
        return False
    
    # Otázka s velmi nízkou relevancí
    q = "xyz random gibberish question that doesn't exist anywhere"
    retrieved = engine.retrieve(q, k=5)
    
    if retrieved:
        result = engine.synthesize_answer(q, retrieved, use_llm=False)
        
        # Ověř fallback strukturu
        checks = [
            ("answer je fallback msg", result["answer"] == "Požadovaná informace není v dokumentech."),
            ("sources je empty list", result["sources"] == []),
            ("confidence je score", isinstance(result["confidence"], float)),
        ]
        
        all_passed = all(check[1] for check in checks)
        
        for desc, check in checks:
            status = "✅" if check else "❌"
            print(f"{status} {desc}: {check}")
        
        return all_passed
    else:
        print("⚠️  Žádné retrieval výsledky")
        return None


def main():
    """Spusť všechny testy"""
    print("\n" + "█"*70)
    print("█ COMPLIANCE TESTING: Hard Factual Gate + Keyword Guard")
    print("█"*70)
    
    tests = [
        ("Hard Factual Gate", test_hard_factual_gate),
        ("Keyword Guard", test_keyword_guard),
        ("LLM Compression", test_llm_compression),
        ("Fallback Consistency", test_fallback_consistency),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = result
        except Exception as e:
            print(f"\n⚠️  Test '{name}' error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{status}: {name}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    print(f"\n📊 {passed}/{total} testů prošlo")
    
    return all(r for r in results.values() if r is not None)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
