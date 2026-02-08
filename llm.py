"""LLM wrapper: hybridní rozhraní pro OpenAI API + fallback na local FLAN-T5.

Tento modul poskytuje jednotné rozhraní pro syntézu odpovědí.
Preferuje OpenAI API (GPT-3.5), ale fallbacks na místní FLAN-T5 bez internetu.

Princip:
- Systémový prompt instruuje LLM, aby odpovídal "pouze z kontextu"
- Explicitně definujeme fallback zprávu ("Požadovaná informace není...")
- Nízká teplota (0.1) = zaměření se na vstupní data bez kreativity
"""

import os
from typing import Optional

# Jednotný systémový prompt pro všechny LLM implementace
SYSTEM_PROMPT = """
Jsi extrakční asistent pro analýzu smluvních dokumentů.

Odpovídej výhradně v českém jazyce.

Odpovídej POUZE na základě poskytnutého kontextu.
Nic si nedomýšlej, neodvozuj a nepřidávej.

Pokud odpověď nelze jednoznačně najít v kontextu,
odpověz přesně touto větou:
"Požadovaná informace není v dokumentech explicitně uvedena."
"""

# LLM compression prompt: explicitně bez halucinací
COMPRESSION_PROMPT = """
Jsi kompresor textu pro právní dokumenty.

PRAVIDLA:
1. Nepřidávej žádné nové informace
2. Odpovídej POUZE z poskytnutého textu
3. Shrnutí max 3 věty
4. Zachovej faktické znění (konkrétní čísla, termíny, pojmy)
5. Pokud nejsi si jistý, než přidávej text, raději vynech

VSTUP: Relevantní texty z dokumentu
VÝSTUP: Stručné shrnutí (max 3 věty) v českém jazyce

Pokud se nemůžeš rozhodnout, vrať """.strip()

# Místní LLM: importujme s graceful fallback
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    # Pokud transformers není nainstalován, jen log a pokračuj
    LOCAL_LLM_AVAILABLE = False

MODEL_NAME = "google/flan-t5-base"


class LLMWrapper:
    """Hybridní LLM wrapper: OpenAI (online) > FLAN-T5 (offline).
    
    Inicializace se pokusí:
    1. Načíst OpenAI API klíč z OPENAI_API_KEY
    2. Inicializovat openai.OpenAI client
    3. Pokud selže, fallback na místní FLAN-T5 (pokud je dostupný)
    4. Pokud ani to není dostupné, vrací syrový kontext
    
    Každá syntetizace používá systémový prompt, který:
    - Instruuje LLM, aby odpovídal "jen z poskytnutého kontextu"
    - Definuje explicitní fallback zprávu
    - Nastavuje nízkou teplotu (0.1) pro fokus na data
    """

    def __init__(self, use_openai: bool = True, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Inicializuje LLM wrapper (OpenAI nebo fallback na FLAN-T5).
        
        Args:
            use_openai: Pokusit se použít OpenAI API (default True).
                Přeskoči, pokud je api_key None a OPENAI_API_KEY není nastaven.
            api_key: Explicitní OpenAI API klíč.
                Pokud None, čte se z environment proměnné OPENAI_API_KEY.
            model: Model name pro OpenAI (default: "gpt-3.5-turbo").
                           Lze změnit na "gpt-4" pro lepší kvalitu (ale dražší).
        """
        # Rozhodnutí: můžeme vůbec použít OpenAI?
        self.use_openai = use_openai and (api_key or os.getenv("OPENAI_API_KEY"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        self.local_llm = None

        # Iniciální pokus o OpenAI
        if self.use_openai:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.api_key)
                print("✓ OpenAI API inicializován (gpt-3.5-turbo).")
            except ImportError:
                print("⚠ Balíček 'openai' není nainstalován, fallback na místní FLAN-T5...")
                self.use_openai = False
            except Exception as e:
                print(f"⚠ OpenAI inicializace selhala: {e}, zkusím FLAN-T5...")
                self.use_openai = False
        
        # Fallback na místní FLAN-T5 (pokud je dostupný)
        if not self.use_openai and LOCAL_LLM_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
                print(f"✓ Místní FLAN-T5 inicializován ({MODEL_NAME}).")
            except Exception as e:
                print(f"⚠ FLAN-T5 inicializace selhala: {e}")
                LOCAL_LLM_AVAILABLE = False

    def synthesize(self, question: str, context: str) -> str:
        """Syntetizuje odpověď na otázku na základě kontextu.
        
        Strategie:
        1. Pokud je OpenAI dostupné: zavolej _synthesize_openai()
        2. Pokud je FLAN-T5 dostupné: zavolej _synthesize_local()
        3. Fallback: vrať první řádek kontextu (nebo chybovou zprávu)
        
        Systémový prompt v obou případech instruuje LLM:
        - Odpovídej "pouze z provided kontextu"
        - Pokud info chybí, vrať přesně tuto zprávu
        - Nehallucidu, nezačínej vymýšlet věci
        
        Args:
            question: Otázka od uživatele
            context: Relevantní textový kontext (spojené chunky z FAISS)
        
        Returns:
            Odpověď od LLM jako string (nebo fallback zpráva)
        """
        if self.use_openai and hasattr(self, 'openai_client'):
            return self._synthesize_openai(question, context)
        elif LOCAL_LLM_AVAILABLE:
            return self._synthesize_local(question, context)
        else:
            # Poslední zastávka: vrať syrový kontext nebo chybu
            return context.split('\n')[0] if context else "Chyba: žádný LLM ani kontext dostupný."

    def compress_answer(self, question: str, context: str) -> str:
        """Kompresor textu: zkrátí text bez generování nových informací.
        
        ⚠️ KRITICKY ROZDÍLNÉ OD synthesize()
        
        - synthesize(): Odpovídá na otázku (může vytvořit odpověď z kontextu)
        - compress_answer(): Jen ZKRÁTÍ poskytnutý text (NEPŘIDÁ nic nového)
        
        PRAVIDLA KOMPRESE:
        1. Zachovej faktické znění (čísla, termíny, pojmy)
        2. Vrať max 3 věty
        3. Nikdy negeneruj (nedomýšlej si)
        4. Pokud je text krátký (< 3 věty): vrať jej beze změny
        
        Použití: RAGEngine.synthesize_answer() když use_llm=True
        
        Args:
            question: Uživatelská otázka (pro kontext)
            context: Relevantní texty z FAISS chunků
        
        Returns:
            Zkrácený text (max 3 věty, fakticky přesný)
        """
        # Pokud je text už krátký, vrať jej beze změny
        sentences = [s for s in context.replace("---", ".").split(".") if s.strip()]
        if len(sentences) <= 3:
            return context.strip()
        
        # Jinak zavolej LLM jako kompresor
        if self.use_openai and hasattr(self, 'openai_client'):
            return self._compress_openai(question, context)
        elif LOCAL_LLM_AVAILABLE:
            return self._compress_local(question, context)
        else:
            # Fallback: vrať první 3 věty
            return " ".join(sentences[:3])

    def _compress_openai(self, question: str, context: str) -> str:
        """OpenAI kompresor pomocí GPT-3.5-turbo."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": COMPRESSION_PROMPT.strip()},
                    {
                        "role": "user",
                        "content": f"Zhrnuj tento text max 3 věty (jen zkrácení, bez přidávání):\n\n{context}",
                    },
                ],
                max_tokens=150,
                temperature=0.0,  # Deterministic: žádná kreativita
            )
            return response.choices[0].message.content.strip() or context
        except Exception as e:
            print(f"⚠ OpenAI kompresor chyba: {e}, fallback na raw text")
            return context

    def _compress_local(self, question: str, context: str) -> str:
        """FLAN-T5 kompresor (offline mode)."""
        prompt = (
            f"{COMPRESSION_PROMPT.strip()}\n\n"
            f"Text k shrnutí (max 3 věty):\n{context}\n\nZhrnuti:"
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=100)
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True) or context

    def _synthesize_openai(self, question: str, context: str) -> str:
        """Volá OpenAI API s instrukcí pro extrakci bez halucinací.
        
        Parametry OpenAI call:
        - model: gpt-3.5-turbo (default, levný a fast)
        - temperature: 0.1 (velmi nízká = fokus a konzistence)
        - max_tokens: 200 (cca 3-4 věty)
        - systemPrompt: SYSTEM_PROMPT konstanta (zajistí konzistenci)
        
        If OpenAI API selže, vrací chybovou zprávu.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {
                        "role": "user",
                        "content": f"KONTEXT:\n{context}\n\nOTÁZKA:\n{question}\n\nOdpověz na základě kontextu:",
                    },
                ],
                max_tokens=200,
                temperature=0.1,  # Nízká teplota = determinisitické odpovědi bez kreativnosti
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠ OpenAI API chyba: {e}")
            return f"Chyba LLM: {str(e)}"

    def _synthesize_local(self, question: str, context: str) -> str:
        """Fallback syntéza pomocí místního FLAN-T5 modelu (bez internetu).
        
        FLAN-T5 je seq2seq model (otázka -> odpověď), který si všimá sekvencí.
        Používá SYSTEM_PROMPT konstantu (stejný prompt jako OpenAI).
        
        Výhody: offline, bez závislostí, levý compute
        Nevýhody: horší kvalita než GPT-3.5, pomalejší na CPU
        """
        # Sestav prompt s jednotným systémovým promptem
        prompt = f"{SYSTEM_PROMPT.strip()}\n\nKONTEXT:\n{context}\n\nOTÁZKA:\n{question}\n\nOdpověz:"

        # Tokenizace s truncation (FLAN-T5 má limit na input)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Generování bez gradientů (inference mód)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=150)

        # Dekódování a čištění special tokens
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# Backward compatibility
class LocalLLM(LLMWrapper):
    """Zpětná kompatibilita: pokud někdo používá LocalLLM přímo."""
    def __init__(self):
        super().__init__(use_openai=False)
