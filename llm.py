"""
LLM wrapper: hybridní rozhraní pro OpenAI API + fallback na local FLAN-T5.
"""

import os
from typing import Optional

# Local LLM (fallback)
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

MODEL_NAME = "google/flan-t5-base"


class LLMWrapper:
    """Hybridní wrapper: preferuje OpenAI, fallback na local LLM."""

    def __init__(self, use_openai: bool = True, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Args:
            use_openai: Pokusit se použít OpenAI API.
            api_key: OpenAI API key (pokud None, čte z OPENAI_API_KEY).
            model: Model name pro OpenAI (default: gpt-3.5-turbo).
        """
        self.use_openai = use_openai and (api_key or os.getenv("OPENAI_API_KEY"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        self.local_llm = None

        if self.use_openai:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.api_key)
                print("✓ OpenAI API inicializován.")
            except ImportError:
                print("⚠ openai nebyl nalezen, padáme na local LLM.")
                self.use_openai = False
        
        # Fallback na local LLM
        if not self.use_openai and LOCAL_LLM_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            print("✓ Local FLAN-T5 LLM inicializován.")

    def synthesize(self, question: str, context: str) -> str:
        """Syntetizuje odpověď na základě otázky a kontextu.
        
        Používá systémový prompt s instrukcí, aby LLM odpovídal jen
        z poskytnutého kontextu a explicitně řekl, pokud odpověď tam není.
        """
        if self.use_openai and hasattr(self, 'openai_client'):
            return self._synthesize_openai(question, context)
        elif LOCAL_LLM_AVAILABLE:
            return self._synthesize_local(question, context)
        else:
            # Fallback: syrový text
            return context.split('\n')[0] if context else "Chyba: LLM nedostupný."

    def _synthesize_openai(self, question: str, context: str) -> str:
        """Volání OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Jsi extrakční asistent. Odpovídej pouze na základě poskytnutého kontextu. "
                            "Pokud odpověď v kontextu není, řekni přesně: "
                            "'Požadovaná informace není v dokumentech explicitně uvedena.' "
                            "Odpověz jednou nebo dvěma větami bez parafrází právního obsahu."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"KONTEXT:\n{context}\n\nOTÁZKA:\n{question}\n\nOdpověz:",
                    },
                ],
                max_tokens=200,
                temperature=0.1,  # Low temp = fokus na kontext
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI chyba: {e}")
            return f"Chyba LLM: {str(e)}"

    def _synthesize_local(self, question: str, context: str) -> str:
        """Fallback na local FLAN-T5."""
        prompt = (
            "Jsi extrakční asistent. Odpovídej pouze na základě poskytnutého kontextu. "
            "Pokud odpověď v kontextu není, řekni přesně: 'Požadovaná informace není v dokumentech explicitně uvedena.' "
            f"\n\nKONTEXT:\n{context}\n\nOTÁZKA:\n{question}\n\nOdpověz:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=150)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# Legacy compatibility
class LocalLLM(LLMWrapper):
    """Backward compatibility."""
    def __init__(self):
        super().__init__(use_openai=False)
