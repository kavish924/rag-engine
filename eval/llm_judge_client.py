from openai import OpenAI

from app.config import settings


def call_judge_llm(system_prompt: str, user_prompt: str) -> str:
    """Single entry point for eval judge calls. Ollama-only — production
    generation (settings.llm_provider) is handled separately in generator.py."""
    client = OpenAI(api_key="ollama", base_url=settings.ollama_base_url)

    response = client.chat.completions.create(
        model=settings.ollama_judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()