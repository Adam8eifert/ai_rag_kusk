"""Pravidla pro detekci a filtraci otázek.

Tento modul definuje sady otázek a odpovídajících klíčových slov,
které se používají k filtraci relevantních chunků z indexu.

Struktura pravidla:
- question_keywords: Slova, která signalizují typ otázky
- section_keywords: Klíčová slova, která musí být obsažena v relevantním chunku
- min_similarity: Minimální prahová hodnota kosinusové podobnosti
"""

QUESTION_RULES = {
    # Otázky týkající se doby plnění (lhůt, termínů)
    "doba_plneni": {
        "question_keywords": [
            "doba plnění",
            "termín plnění",
            "lhůta plnění"
        ],
        "section_keywords": [
            "doba plnění",
            "plnění smlouvy",
            "termín plnění",
            "lhůta plnění"
        ],
        "min_similarity": 0.75  # Vyšší práh pro specifické termíny
    },

    # Otázky týkající se platnosti smlouvy
    "platnost_smlouvy": {
        "question_keywords": [
            "platnost smlouvy",
            "doba platnosti"
        ],
        "section_keywords": [
            "platnost smlouvy",
            "ukončení smlouvy"
        ],
        "min_similarity": 0.7
    },

    # Otázky na objednatele/subjekt
    "objednatel": {
        "question_keywords": [
            "objednatel",
            "kdo je objednatelem"
        ],
        "section_keywords": [
            "objednatel",
            "smluvní strany"
        ],
        "min_similarity": 0.7
    }
}
