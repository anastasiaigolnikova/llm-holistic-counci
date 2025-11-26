"""
LLM Council — консилиум нейросетей для проверки гипотез и фактов
Упрощённая версия на базе идеи Andrej Karpathy
"""

import asyncio
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
# langdetect больше не используется - простая проверка кириллицы
import re

load_dotenv()

app = FastAPI(title="LLM Council")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ КОНФИГУРАЦИЯ ============
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Модели совета через OpenRouter
# ВАЖНО: сюда входят все модели, которые могут участвовать в консилиуме И/ИЛИ быть судьёй.
COUNCIL_MODELS = {
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "name": "GPT-4.1",
        "provider": "OpenAI",
    },
    "claude-sonnet": {
        "model": "anthropic/claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "provider": "Anthropic",
    },
    "perplexity": {
        "model": "perplexity/sonar-pro",
        "name": "Sonar Reasoning Pro",
        "provider": "Perplexity",
    },
    "llama": {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "provider": "Meta",
    },
}

# Председатель совета по умолчанию
CHAIRMAN_MODEL = "claude-sonnet"

# Простая проверка наличия кириллицы
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]")


def determine_language(text: str, client_hint: Optional[str] = None) -> str:
    """
    Простое определение языка:
    - Если в тексте есть кириллица -> русский
    - Если нет кириллицы -> английский
    - client_hint используется только как fallback для пустых текстов
    """
    if not text or not text.strip():
        return client_hint if client_hint in ("ru", "en") else "en"
    
    # Проверяем наличие кириллицы
    has_cyrillic = bool(CYRILLIC_PATTERN.search(text))
    
    if has_cyrillic:
        return "ru"
    else:
        return "en"


# ============ ПРОМПТЫ ============
# Функция для получения языковой инструкции - МАКСИМАЛЬНО КОРОТКОЙ И ЯВНОЙ
def get_language_instruction(lang_code: str) -> str:
    if lang_code == "ru":
        return "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. ВСЕ СЛОВА ДОЛЖНЫ БЫТЬ НА РУССКОМ."
    else:
        return "RESPOND ONLY IN ENGLISH. EVERY WORD MUST BE IN ENGLISH. NO RUSSIAN."

# Базовый системный промпт для членов совета
# ВАЖНО: {language_instruction} должна быть ПЕРВОЙ СТРОКОЙ!
COUNCIL_SYSTEM_PROMPT_BASE = """{language_instruction}

You are an advanced, structured assistant. Answer the user's question thoroughly and precisely."""

# Системный промпт для независимого судьи  
CHAIRMAN_SYSTEM_PROMPT_BASE = """{language_instruction}

You are an impartial judge. Create TWO sections:
## Analysis of Council Responses (short)
## Final Synthesis (main section - follow user's question structure)"""

# Системный промпт для peer‑review
PEER_REVIEW_SYSTEM_PROMPT_BASE = """{language_instruction}

You are a peer reviewer. Briefly review each answer (A, B, C)."""


# ============ API КЛИЕНТЫ ============
async def call_openrouter(model: str, prompt: str, system: str, max_tokens: int = 2000) -> str:
    """Запрос к OpenRouter API"""
    if not OPENROUTER_API_KEY:
        return "❌ Ошибка: OPENROUTER_API_KEY не настроен"
    
    # Увеличенный таймаут для больших ответов
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Ошибка {model}: {str(e)}"


def build_system_prompt(base_prompt: str, lang_code: str) -> str:
    """Строит системный промпт с языковой инструкцией"""
    lang_instruction = get_language_instruction(lang_code)
    return base_prompt.replace("{language_instruction}", lang_instruction)


async def call_model(model_key: str, prompt: str, lang_code: str = "en", max_tokens: int = 3000) -> dict:
    """Универсальный вызов модели"""
    model_info = COUNCIL_MODELS.get(model_key)
    if not model_info:
        return {"model": model_key, "response": "❌ Неизвестная модель", "error": True}
    
    # Строим системный промпт с языковой инструкцией
    system_prompt = build_system_prompt(COUNCIL_SYSTEM_PROMPT_BASE, lang_code)
    
    response = await call_openrouter(model_info["model"], prompt, system_prompt, max_tokens)
    
    return {
        "model": model_key,
        "name": model_info["name"],
        "response": response,
        "error": response.startswith("❌")
    }


# ============ ЭНДПОИНТЫ ============
class QueryRequest(BaseModel):
    query: str
    skip_chairman: bool = False
    judge: Optional[str] = None
    query_language: str = "en"


class CouncilResponse(BaseModel):
    query: str
    individual_responses: list[dict]
    chairman_response: Optional[str]
    reviews: list[dict] = []
    timestamp: str


class TranslateRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    """Отдаём HTML интерфейс"""
    return FileResponse("index.html")


@app.get("/api/status")
async def status():
    """Проверка статуса API ключей"""
    return {
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "council_models": list(COUNCIL_MODELS.keys()),
        "chairman_model": CHAIRMAN_MODEL,
    }


@app.get("/api/models")
async def get_models():
    """
    Список доступных моделей для фронтенда:
    - models: [{ key, name, provider }]
    - default_judge: ключ модели судьи по умолчанию
    """
    models = [
        {
            "key": key,
            "name": info["name"],
            "provider": info.get("provider", ""),
        }
        for key, info in COUNCIL_MODELS.items()
    ]
    return {
        "models": models,
        "default_judge": CHAIRMAN_MODEL,
    }


@app.post("/api/council", response_model=CouncilResponse)
async def council_query(request: QueryRequest):
    """Основной запрос к консилиуму"""
    
    # Определяем язык по тексту запроса
    lang_code = determine_language(request.query, request.query_language)
    lang_instruction = get_language_instruction(lang_code)
    
    print(f"🔍 LANG DEBUG: query_language from client = '{request.query_language}'")
    print(f"🔍 LANG DEBUG: determined lang_code = '{lang_code}'")
    print(f"🔍 LANG DEBUG: instruction preview = '{lang_instruction[:50]}...'")
    
    # Определяем, кто сейчас судья
    judge_key = request.judge or CHAIRMAN_MODEL
    if judge_key not in COUNCIL_MODELS:
        judge_key = CHAIRMAN_MODEL
    
    # Модели совета — все, кроме судьи
    council_keys = [key for key in COUNCIL_MODELS.keys() if key != judge_key]
    
    # ========== ЭТАП 1: индивидуальные ответы ==========
    # Языковая инструкция В САМОМ НАЧАЛЕ user prompt
    query_with_lang = f"[{lang_instruction}]\n\n{request.query}"
    
    tasks = [
        call_model(model_key, query_with_lang, lang_code=lang_code)
        for model_key in council_keys
    ]
    individual_responses = await asyncio.gather(*tasks)
    
    # Подготавливаем ответы с анонимными метками A, B, C...
    valid_answers = [r for r in individual_responses if not r.get("error")]
    labels = [chr(ord("A") + i) for i in range(len(valid_answers))]
    labelled_answers_text = "\n\n".join(
        f"Answer {label}:\n{resp['response']}"
        for label, resp in zip(labels, valid_answers)
    )
    
    # ========== ЭТАП 2: peer‑review от членов совета ==========
    peer_reviews: list[dict] = []
    if valid_answers:
        review_prompt = f"""[{lang_instruction}]

Question: {request.query}

Answers:
{labelled_answers_text}

Review each answer briefly."""
        
        peer_review_system = build_system_prompt(PEER_REVIEW_SYSTEM_PROMPT_BASE, lang_code)
        
        review_tasks = []
        for model_key in council_keys:
            model_info = COUNCIL_MODELS[model_key]
            review_tasks.append(
                call_openrouter(
                    model_info["model"],
                    review_prompt,
                    peer_review_system,
                )
            )
        
        review_texts = await asyncio.gather(*review_tasks)
        for model_key, review_text in zip(council_keys, review_texts):
            peer_reviews.append({
                "reviewer": COUNCIL_MODELS[model_key]["name"],
                "review": review_text,
            })
    
    # ========== ЭТАП 3: независимый судья ==========
    chairman_response = None
    if not request.skip_chairman:
        reviews_block = "\n\n".join(
            f"Review by {r['reviewer']}:\n{r['review']}"
            for r in peer_reviews
        ) if peer_reviews else ""
        
        chairman_prompt = f"""[{lang_instruction}]

QUESTION: {request.query}

ANSWERS:
{labelled_answers_text}

{f"REVIEWS: {reviews_block}" if reviews_block else ""}

Create: ## Analysis of Council Responses (short) ## Final Synthesis (main)

[{lang_instruction}]"""
        
        chairman_system = build_system_prompt(CHAIRMAN_SYSTEM_PROMPT_BASE, lang_code)
        chairman_info = COUNCIL_MODELS[judge_key]
        
        chairman_response = await call_openrouter(
            chairman_info["model"],
            chairman_prompt,
            chairman_system,
            max_tokens=4000
        )
    
    return CouncilResponse(
        query=request.query,
        individual_responses=individual_responses,
        chairman_response=chairman_response,
        reviews=peer_reviews,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """Перевод текста на английский через GPT"""
    try:
        translation = await call_openrouter(
            "openai/gpt-4o-mini",
            f"Translate the following text to English. Only output the translation, no explanations:\n\n{request.text}",
            "You are a professional translator."
        )
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("🏛️ LLM Council запускается...")
    print(f"   OpenRouter API: {'✅' if OPENROUTER_API_KEY else '❌'}")
    print(f"   Открой http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

