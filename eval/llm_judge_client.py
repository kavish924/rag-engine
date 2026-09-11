from groq import Groq
from openai import OpenAI

from app.config import settings


def call_judge_llm(system_prompt: str, user_prompt: str) -> str:
    """Single entry point for all eval judge calls.
    Reads settings.eval_judge_provider — independent of settings.llm_provider,
    which controls production generation."""
    provider = settings.eval_judge_provider.lower()

    if provider == "groq":
        client = Groq(api_key=settings.groq_api_key)
        model = settings.generation_model
    elif provider == "gemini":
        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = settings.gemini_model
    elif provider == "ollama":
        client = OpenAI(api_key="ollama", base_url=settings.ollama_base_url)
        model = settings.ollama_judge_model
    else:
        raise ValueError(f"Unknown eval judge provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # judges should be deterministic, not creative
    )
    return response.choices[0].message.content.strip()