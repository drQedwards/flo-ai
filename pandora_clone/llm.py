from functools import lru_cache
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

_DEFAULT_MODEL = os.getenv("PANDORA_MODEL", "gpt2")

@lru_cache()
def _get_pipeline(model_name: str = _DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", torch_dtype="auto")


def generate(prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
    pipe = _get_pipeline()
    outputs = pipe(prompt, max_length=len(prompt.split()) + max_tokens, do_sample=True, temperature=temperature, num_return_sequences=1)
    # HuggingFace pipeline returns list of dicts with 'generated_text'
    text = outputs[0]["generated_text"]
    # Remove the prompt prefix to return only completion part
    completion = text[len(prompt):]
    return completion.strip()