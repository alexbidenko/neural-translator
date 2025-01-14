import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import torch

app = FastAPI()

# Модель по умолчанию (на случай, если переменная окружения не задана)
DEFAULT_MODEL_NAME = "some-llama-model"  # <-- замените на реальный репозиторий/путь

# Считываем имя модели (или путь) из переменной окружения
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

# Опциональные настройки генерации
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель
print(f"Loading model: {MODEL_NAME}, device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,               # Иногда для LLaMA нужно отключить fast-токенизацию
    trust_remote_code=True        # Если модель кастомная
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
)

# Создаем pipeline для удобства
generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1
)

# Pydantic модель для входящих запросов
class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: GenerateRequest):
    prompt = req.prompt
    # Генерация
    outputs = generation_pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        num_return_sequences=1
    )

    # outputs — это список, берем первый сгенерированный текст
    # Пример: [{'generated_text': 'Your full text...'}]
    result_text = outputs[0]["generated_text"]
    return {"generated_text": result_text}


@app.get("/health")
def health():
    return {"status": "ok"}
