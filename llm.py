from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


MODEL_NAME = "google/flan-t5-base"


class LocalLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def answer(self, question: str, context: str) -> str:
        prompt = (
            "Odpověz na otázku pouze na základě uvedeného textu.\n\n"
            f"TEXT:\n{context}\n\n"
            f"OTÁZKA:\n{question}\n\n"
            "ODPOVĚĎ:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
